"""SnapCount API: POST /count and GET /health. Photos are processed in memory and never stored."""

from __future__ import annotations

import io
import logging
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings, get_settings
import asyncio
from contextlib import asynccontextmanager

from .consensus import combine
from .stack import is_nestable
from .providers import (AnthropicProvider, CountProvider, CountResult, MockProvider,
                        ProviderError, ProviderTimeout)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snapcount")

SERVER_VERSION = "2026-11-04.stack-sides"
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
Image.MAX_IMAGE_PIXELS = 60_000_000  # guard against decompression bombs


def prepare_image(data: bytes, max_side: int) -> bytes:
    """Validate the upload is a real image, fix rotation, shrink it and re-encode as JPEG."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
        raise HTTPException(415, "That file isn't a supported image. Please use a JPEG or PNG photo.") from e
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, "That file isn't a supported image. Please use a JPEG or PNG photo.")

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            # Transparent areas would turn black; put the image on white instead.
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=85, optimize=True)
        return out.getvalue()
    except (Image.DecompressionBombError, OSError) as e:
        raise HTTPException(415, "That image couldn't be read. Please try another photo.") from e


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := await upload.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, f"Photo is too large. The limit is {limit // (1024 * 1024)} MB.")
    return bytes(buf)


MODEL_INFO = {
    "owlv2": ("OWLv2", "free", "Google's open detector. Good on small, dense items."),
    "grounding_dino": ("Grounding DINO", "free", "Good at matching the item name."),
    "omdet_turbo": ("OmDet-Turbo", "free", "The fastest free model."),
    "stack": ("Stacked items", "free", "For nested stacks (chairs, plates, buckets): one per edge."),
    "claude": ("Claude", "paid", "Anthropic's AI. Uses your API key; about US$0.01 per count."),
    "mock": ("Test model", "test", "Always answers 12. For testing only."),
}


def build_registry(settings: Settings) -> tuple[dict[str, CountProvider], dict[str, str]]:
    """Create every configured model. Returns (models, reasons-unavailable)."""
    wanted = ("mock",) if settings.provider == "mock" else settings.models
    models: dict[str, CountProvider] = {}
    problems: dict[str, str] = {}
    detector_ids = [m for m in wanted if m in ("owlv2", "grounding_dino", "omdet_turbo")]
    if detector_ids:
        try:
            from .detectors import make_detectors
            models.update(make_detectors(detector_ids))
        except ImportError as e:  # torch/transformers not installed
            for m in detector_ids:
                problems[m] = f"free models need torch and transformers installed ({e.name})"
    if "stack" in wanted:
        try:
            from .stack import StackProvider
            locator = models.get("grounding_dino") or models.get("owlv2") or models.get("omdet_turbo")
            models["stack"] = StackProvider(locator)
        except ImportError as e:
            problems["stack"] = f"needs numpy ({e.name})"
    if "claude" in wanted:
        if settings.anthropic_api_key:
            models["claude"] = AnthropicProvider(settings)
        else:
            problems["claude"] = "ANTHROPIC_API_KEY is not set"
    if "mock" in wanted:
        models["mock"] = MockProvider()
    return models, problems


def result_json(model_id: str, r: CountResult, seconds: float) -> dict:
    name = MODEL_INFO.get(model_id, (model_id,))[0]
    return {
        "model_id": model_id,
        "model_name": name,
        "count": r.count,
        "confidence": r.confidence,
        "notes": r.notes,
        "points": r.points,
        "seconds": round(seconds, 1),
        "error": None,
    }


def create_app(settings: Settings | None = None, models: dict[str, CountProvider] | None = None) -> FastAPI:
    settings = settings or get_settings()

    problems: dict[str, str] = {}
    if models is None:
        models, problems = build_registry(settings)
    auto_ids = [m for m in settings.auto_models if m in models] or list(models)[:3]
    for m, why in problems.items():
        log.warning("model %s unavailable: %s", m, why)

    def load_all():
        for mid, prov in models.items():
            warm = getattr(prov, "warm_up", None)
            if warm:
                try:
                    warm()
                except Exception:
                    log.exception("could not preload %s", mid)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload:
            # Load in the background so the server answers /health straight away.
            asyncio.get_running_loop().run_in_executor(None, load_all)
        yield

    app = FastAPI(title="SnapCount API", version="2.0.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        # Always return {"error": "..."} so the app can show the message as-is.
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse({"error": "The request was missing the photo. Please try again."}, status_code=422)

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if models else "misconfigured",
            "version": SERVER_VERSION,
            "models": list(models),
            "auto_models": auto_ids,
            "unavailable": problems,
        }

    @app.get("/models")
    async def list_models():
        out = []
        if len(auto_ids) > 1:
            names = ", ".join(MODEL_INFO.get(m, (m,))[0] for m in auto_ids)
            out.append({"id": "auto", "name": "Auto", "kind": "free",
                        "description": f"Runs {names} and combines their counts."})
        for mid in models:
            name, kind, desc = MODEL_INFO.get(mid, (mid, "free", ""))
            out.append({"id": mid, "name": name, "kind": kind, "description": desc})
        return {"default": "auto" if len(auto_ids) > 1 else next(iter(models), None), "models": out}

    async def run_one(mid: str, jpeg: bytes, item: str | None,
                      line_coords=None) -> tuple[str, CountResult | Exception, float]:
        t = time.perf_counter()
        try:
            if mid == "stack" and line_coords is not None:
                r = await models[mid].count(jpeg, "image/jpeg", item, line=line_coords)
            else:
                r = await models[mid].count(jpeg, "image/jpeg", item)
            return mid, r, time.perf_counter() - t
        except Exception as e:  # noqa: BLE001 - reported per model
            return mid, e, time.perf_counter() - t

    @app.post("/count")
    async def count(
        image: UploadFile = File(...),
        item: str = Form(""),
        model: str = Form("auto"),
        line: str = Form(""),
        x_app_token: str | None = Header(default=None),
    ):
        if settings.app_token and x_app_token != settings.app_token:
            raise HTTPException(401, "This app isn't authorised to use the server.")
        if not models:
            raise HTTPException(503, "The counting service isn't configured yet.")

        model = (model or "auto").strip().lower()
        item_name = " ".join(item.split())[:80] or None
        line_coords = None
        if line.strip():
            try:
                line_coords = tuple(float(v) for v in line.split(","))
                assert len(line_coords) == 4 and all(-0.05 <= v <= 1.05 for v in line_coords)
            except (ValueError, AssertionError):
                raise HTTPException(400, "The line couldn't be read. Please draw it again.")
            if "stack" not in models:
                raise HTTPException(400, "Counting along a line isn't available on this server.")
            model = "stack"
        if model == "auto":
            run_ids = list(auto_ids)
            # Nested stacks look like one object to a detector; also count the edges.
            if "stack" in models and "stack" not in run_ids and is_nestable(item_name):
                run_ids.append("stack")
        elif model in models:
            run_ids = [model]
        else:
            raise HTTPException(400, f"Unknown model '{model}'. Pick one from the list.")

        raw = await read_limited(image, settings.max_upload_bytes)
        if not raw:
            raise HTTPException(400, "No photo was received.")
        jpeg = prepare_image(raw, settings.max_image_side)

        started = time.perf_counter()
        outcomes = await asyncio.gather(*(run_one(m, jpeg, item_name, line_coords) for m in run_ids))
        latency = time.perf_counter() - started

        results, ok = [], {}
        for mid, r, secs in outcomes:
            if isinstance(r, Exception):
                log.warning("model %s failed: %s", mid, r)
                timeout = isinstance(r, ProviderTimeout)
                results.append({"model_id": mid, "model_name": MODEL_INFO.get(mid, (mid,))[0],
                                "count": None, "confidence": "low", "notes": "", "points": [],
                                "seconds": round(secs, 1),
                                "error": "Took too long." if timeout else "This model had a problem."})
            else:
                ok[mid] = r
                results.append(result_json(mid, r, secs))

        if not ok:
            if all(isinstance(r, ProviderTimeout) for _, r, _ in outcomes):
                raise HTTPException(504, "Counting took too long. Please try again.")
            raise HTTPException(502, "The counting service had a problem. Please try again.")

        if len(run_ids) == 1:
            mid = run_ids[0]
            r = ok[mid]
            top = {"count": r.count, "confidence": r.confidence, "notes": r.notes,
                   "retake_tip": r.retake_tip, "points": r.points, "best_model": mid,
                   "sections": [{"label": s.label, "count": s.count} for s in r.sections]}
        else:
            stack = ok.get("stack")
            detectors = {m: r.count for m, r in ok.items() if m != "stack"}
            combo = combine(detectors) if detectors else {"count": None, "confidence": "low", "best_model": None, "notes": ""}
            looks_like_stack = (stack is not None and stack.count and stack.count >= 4
                                and (combo["count"] is None or combo["count"] <= 3))
            if looks_like_stack:
                # The detectors saw the stack as one or two objects; trust the edge count.
                combo = {"count": stack.count, "confidence": stack.confidence, "best_model": "stack",
                         "notes": "The detectors saw this as a stack, so it was counted edge by edge. "
                                  + stack.notes}
            best = combo["best_model"]
            tip = "" if combo["confidence"] == "high" else (
                "Check the dots on the photo, or take a closer photo of a smaller section.")
            top = {"count": combo["count"], "confidence": combo["confidence"], "notes": combo["notes"],
                   "retake_tip": (stack.retake_tip or tip) if looks_like_stack else tip,
                   "points": ok[best].points if best else [], "best_model": best,
                   "sections": []}

        # Log usage only. The photo itself is never logged or stored.
        tokens_in = sum(r.input_tokens for r in ok.values())
        tokens_out = sum(r.output_tokens for r in ok.values())
        cost = (tokens_in * settings.price_in_per_mtok + tokens_out * settings.price_out_per_mtok) / 1_000_000
        log.info("count ok item=%r model=%s count=%s conf=%s per_model=%s cost_usd=%.4f latency=%.1fs upload_kb=%d",
                 item_name, model, top["count"], top["confidence"],
                 {m: r.count for m, r in ok.items()}, cost, latency, len(raw) // 1024)

        first = next(iter(ok.values()))
        return {
            "item": first.item if item_name is None else item_name,
            "model": model,
            "count": top["count"],
            "confidence": top["confidence"],
            "notes": top["notes"],
            "retake_suggested": top["confidence"] != "high",
            "retake_tip": top["retake_tip"],
            "sections": top["sections"],
            "points": top["points"],
            "best_model": top["best_model"],
            "results": results,
        }

    return app


app = create_app()
