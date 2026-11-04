import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app, prepare_image
from app.providers import AnthropicProvider, MockProvider, normalize


def jpeg_bytes(w=800, h=600, color=(120, 80, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def settings(**kw) -> Settings:
    base = dict(provider="mock", anthropic_api_key="test-key", app_token="", max_upload_mb=10)
    base.update(kw)
    return Settings(**base)


def client(provider=None, **kw) -> TestClient:
    return TestClient(create_app(settings(**kw), {"mock": provider or MockProvider()}))


def post(c, data=None, item="plastic chairs", headers=None, ctype="image/jpeg"):
    files = {"image": ("photo.jpg", data if data is not None else jpeg_bytes(), ctype)}
    return c.post("/count", files=files, data={"item": item}, headers=headers or {})


# ---------- API ----------

def test_health():
    r = client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["models"] == ["mock"]


def test_count_success_shape():
    r = post(client())
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 12
    assert body["confidence"] == "medium"
    assert body["retake_suggested"] is True
    assert body["item"] == "plastic chairs"
    assert [s["count"] for s in body["sections"]] == [7, 5]


def test_count_without_item():
    r = post(client(), item="")
    assert r.status_code == 200
    assert r.json()["item"] == "items"


def test_rejects_non_image():
    r = post(client(), data=b"this is not an image", ctype="text/plain")
    assert r.status_code == 415
    assert "supported image" in r.json()["error"]


def test_rejects_empty_upload():
    r = post(client(), data=b"")
    assert r.status_code == 400


def test_rejects_too_large():
    big = b"\xff" * (1024 * 1024 + 10)
    r = post(client(max_upload_mb=1), data=big)
    assert r.status_code == 413
    assert "1 MB" in r.json()["error"]


def test_missing_image_field():
    r = client().post("/count", data={"item": "logs"})
    assert r.status_code == 422
    assert "photo" in r.json()["error"]


def test_app_token_required_when_set():
    c = client(app_token="secret")
    assert post(c).status_code == 401
    assert post(c, headers={"X-App-Token": "wrong"}).status_code == 401
    assert post(c, headers={"X-App-Token": "secret"}).status_code == 200


def test_misconfigured_provider_returns_503():
    app = create_app(settings(provider="", models=("claude",), anthropic_api_key=""))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "misconfigured"
    r = post(c)
    assert r.status_code == 503


# ---------- image preparation ----------

def test_prepare_image_downscales_and_reencodes():
    png = io.BytesIO()
    Image.new("RGB", (4000, 3000), (10, 200, 10)).save(png, "PNG")
    out = prepare_image(png.getvalue(), 1600)
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert max(img.size) == 1600


def test_prepare_image_applies_exif_rotation():
    img = Image.new("RGB", (400, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    out = Image.open(io.BytesIO(prepare_image(buf.getvalue(), 1600)))
    assert out.size == (200, 400)


# ---------- result normalisation ----------

def test_normalize_uses_section_sum_when_total_disagrees():
    r = normalize(
        {"item": "logs", "sections": [{"label": "a", "count": 10}, {"label": "b", "count": 8}],
         "countable": True, "count": 20, "confidence": "high", "notes": "", "retake_tip": ""},
        "logs",
    )
    assert r.count == 18
    assert r.retake_tip == ""


def test_normalize_uncountable_forces_null_and_low():
    r = normalize(
        {"item": "chairs", "sections": [], "countable": False, "count": 40,
         "confidence": "high", "notes": "", "retake_tip": ""},
        "chairs",
    )
    assert r.count is None
    assert r.confidence == "low"
    assert r.notes
    assert r.retake_tip


def test_normalize_bad_values_are_safe():
    r = normalize(
        {"item": "", "sections": [{"label": "x", "count": "nope"}, {"label": "y", "count": -3}],
         "countable": True, "count": "abc", "confidence": "certain", "notes": "", "retake_tip": ""},
        None,
    )
    assert r.count is None
    assert r.confidence == "low"
    assert r.sections == []


# ---------- Anthropic provider (HTTP mocked, no real API calls) ----------

def tool_response(inp: dict) -> dict:
    return {
        "model": "claude-sonnet-5-5",
        "content": [{"type": "tool_use", "id": "t1", "name": "report_count", "input": inp}],
        "usage": {"input_tokens": 1800, "output_tokens": 250},
    }


GOOD = {"item": "eucalyptus logs", "sections": [{"label": "pile", "count": 47}], "countable": True,
        "count": 47, "confidence": "medium", "notes": "Some ends overlap.", "retake_tip": "Shoot the ends straight on."}


def anthropic_client(handler):
    s = settings(provider="anthropic")
    provider = AnthropicProvider(s, transport=httpx.MockTransport(handler))
    return TestClient(create_app(s, {"claude": provider}))


def test_anthropic_request_and_parse():
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=tool_response(GOOD))

    r = post(anthropic_client(handler), item="eucalyptus logs")
    assert r.status_code == 200
    assert r.json()["count"] == 47
    body = seen["body"]
    assert seen["headers"]["x-api-key"] == "test-key"
    assert body["tool_choice"] == {"type": "tool", "name": "report_count"}
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert "eucalyptus logs" in content[1]["text"]


def test_anthropic_retries_once_on_overload():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, json={"error": "overloaded"})
        return httpx.Response(200, json=tool_response(GOOD))

    r = post(anthropic_client(handler))
    assert r.status_code == 200
    assert calls["n"] == 2


def test_anthropic_client_error_maps_to_502():
    r = post(anthropic_client(lambda req: httpx.Response(400, json={"error": "bad"})))
    assert r.status_code == 502
    assert "try again" in r.json()["error"]


def test_anthropic_timeout_maps_to_504():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    r = post(anthropic_client(handler))
    assert r.status_code == 504


def test_anthropic_missing_tool_use_maps_to_502():
    resp = {"content": [{"type": "text", "text": "about 40"}], "usage": {}}
    r = post(anthropic_client(lambda req: httpx.Response(200, json=resp)))
    assert r.status_code == 502


# ---------- multiple models and Auto ----------

class FixedProvider(MockProvider):
    def __init__(self, n, fail=False):
        self.n, self.fail = n, fail

    async def count(self, image, media_type, item):
        from app.providers import CountResult, ProviderError
        if self.fail:
            raise ProviderError("boom")
        r = CountResult(item=item or "items", count=self.n, confidence="medium", notes="", retake_tip="")
        r.points = [[0.1 * i, 0.5] for i in range(self.n)]
        return r


def multi_client(**providers):
    s = settings(provider="", auto_models=tuple(providers))
    return TestClient(create_app(s, providers))


def test_models_endpoint_lists_auto_first():
    c = multi_client(owlv2=FixedProvider(10), grounding_dino=FixedProvider(11), omdet_turbo=FixedProvider(30))
    body = c.get("/models").json()
    assert body["default"] == "auto"
    assert [m["id"] for m in body["models"]] == ["auto", "owlv2", "grounding_dino", "omdet_turbo"]


def test_auto_uses_median_and_returns_each_result():
    c = multi_client(owlv2=FixedProvider(10), grounding_dino=FixedProvider(11), omdet_turbo=FixedProvider(30))
    body = post(c).json()
    assert body["count"] == 11
    assert body["best_model"] == "grounding_dino"
    assert len(body["points"]) == 11
    assert body["confidence"] == "low"  # 10..30 is a wide disagreement
    assert {r["model_id"]: r["count"] for r in body["results"]} == {"owlv2": 10, "grounding_dino": 11, "omdet_turbo": 30}


def test_auto_high_confidence_when_models_agree():
    c = multi_client(owlv2=FixedProvider(40), grounding_dino=FixedProvider(41), omdet_turbo=FixedProvider(42))
    body = post(c).json()
    assert body["count"] == 41
    assert body["confidence"] == "high"
    assert body["retake_suggested"] is False


def test_auto_survives_one_model_failing():
    c = multi_client(owlv2=FixedProvider(20), grounding_dino=FixedProvider(0, fail=True), omdet_turbo=FixedProvider(22))
    body = post(c).json()
    assert body["count"] in (20, 22)
    errors = {r["model_id"]: r["error"] for r in body["results"]}
    assert errors["grounding_dino"] and errors["owlv2"] is None


def test_single_model_choice():
    c = multi_client(owlv2=FixedProvider(20), omdet_turbo=FixedProvider(25))
    r = c.post("/count", files={"image": ("p.jpg", jpeg_bytes(), "image/jpeg")}, data={"item": "logs", "model": "omdet_turbo"})
    body = r.json()
    assert body["count"] == 25 and [x["model_id"] for x in body["results"]] == ["omdet_turbo"]


def test_unknown_model_rejected():
    c = multi_client(owlv2=FixedProvider(20))
    r = c.post("/count", files={"image": ("p.jpg", jpeg_bytes(), "image/jpeg")}, data={"model": "gpt"})
    assert r.status_code == 400


def test_all_models_failing_returns_502():
    c = multi_client(owlv2=FixedProvider(0, fail=True), omdet_turbo=FixedProvider(0, fail=True))
    assert post(c).status_code == 502


def test_singular_prompts():
    from app.detectors import _prompt
    assert _prompt("tree logs") == "tree log"
    assert _prompt("bags of cement") == "bag of cement"
    assert _prompt("Boxes") == "box"
    assert _prompt("glass") == "glass"
    assert _prompt("") == "object"


def test_consensus_edge_cases():
    from app.consensus import combine
    assert combine({"a": None, "b": 0})["count"] is None
    one = combine({"a": 7, "b": None})
    assert one["count"] == 7 and one["confidence"] == "low"


def test_stack_counter_counts_repeated_edges():
    """A synthetic side-on stack: 9 light slats separated by dark gaps."""
    import asyncio
    import io as _io
    from PIL import Image, ImageDraw
    from app.stack import StackProvider, is_nestable

    img = Image.new("L", (400, 600), 230)
    d = ImageDraw.Draw(img)
    for i in range(9):
        y = 80 + i * 50
        d.rectangle([60, y, 340, y + 36], fill=245)   # seat lip
        d.rectangle([60, y + 36, 340, y + 50], fill=70)  # dark gap under it
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    r = asyncio.run(StackProvider(None).count(buf.getvalue(), "image/jpeg", "chairs"))
    assert r.count == 9
    assert len(r.points) == r.count
    assert is_nestable("plastic chairs") and is_nestable("stacked buckets") and not is_nestable("logs")


def test_auto_switches_to_stack_when_detectors_see_one_object():
    s = settings(provider="", auto_models=("owlv2", "grounding_dino", "omdet_turbo"))
    models = {"owlv2": FixedProvider(1), "grounding_dino": FixedProvider(1),
              "omdet_turbo": FixedProvider(2), "stack": FixedProvider(14)}
    c = TestClient(create_app(s, models))
    body = post(c, item="plastic chairs").json()
    assert body["count"] == 14 and body["best_model"] == "stack"
    # Not a nestable item: stack counter isn't run.
    body = post(c, item="logs").json()
    assert body["best_model"] != "stack" and "stack" not in [r["model_id"] for r in body["results"]]


def test_auto_ignores_stack_when_detectors_find_many():
    s = settings(provider="", auto_models=("owlv2", "grounding_dino", "omdet_turbo"))
    models = {"owlv2": FixedProvider(11), "grounding_dino": FixedProvider(12),
              "omdet_turbo": FixedProvider(11), "stack": FixedProvider(38)}
    body = post(TestClient(create_app(s, models)), item="chairs").json()
    assert body["count"] == 11


def test_transparent_png_is_put_on_white():
    rgba = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    out = Image.open(io.BytesIO(prepare_image(buf.getvalue(), 1600))).convert("RGB")
    assert min(out.getpixel((10, 10))) > 240


def test_pile_box_is_broken_up_only_when_few_found():
    from app.detectors import SPECS, DetectorProvider
    p = DetectorProvider(SPECS["owlv2"])
    pile = [100, 100, 500, 400]
    ends = [[110 + 90 * i, 300, 180 + 90 * i, 370] for i in range(4)]
    boxes = [pile] + ends
    scores = [0.6] + [0.1] * 4      # the ends are weak, the pile box is strong
    kept = p.filter(boxes, scores, 600, 500, threshold=0.15)
    assert len(kept) == 4           # pile replaced by the 4 ends inside it


def test_ends_only_for_logs():
    from app.detectors import SPECS, DetectorProvider
    p = DetectorProvider(SPECS["owlv2"])
    ends = [[10 + 60 * i, 10, 60 + 60 * i, 60] for i in range(5)]
    sides = [[10, 100 + 40 * i, 400, 130 + 40 * i] for i in range(5)]  # long thin boxes
    kept = p.filter(ends + sides, [0.5] * 10, 600, 400, item="logs")
    assert len(kept) == 5
    assert len(p.filter(ends + sides, [0.5] * 10, 600, 400, item="boxes")) == 10


def test_low_counts_never_high_confidence():
    from app.consensus import combine
    assert combine({"a": 1, "b": 1, "c": 1})["confidence"] != "high"
    assert combine({"a": 14, "b": 1, "c": 13})["count"] in (13, 14)


def _stack_jpeg(n=6):
    from PIL import ImageDraw
    img = Image.new("L", (400, 600), 230)
    d = ImageDraw.Draw(img)
    for i in range(n):
        y = 120 + i * 60
        d.rectangle([60, y, 340, y + 44], fill=245)
        d.rectangle([60, y + 44, 340, y + 60], fill=70)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def test_count_along_a_drawn_line():
    from app.stack import StackProvider
    c = TestClient(create_app(settings(provider=""), {"stack": StackProvider(None)}))
    r = c.post("/count", files={"image": ("s.jpg", _stack_jpeg(6), "image/jpeg")},
               data={"item": "chairs", "model": "auto", "line": "0.5,0.15,0.5,0.85"})
    body = r.json()
    assert r.status_code == 200, body
    assert body["count"] == 6 and len(body["points"]) == 6
    assert all(abs(p[0] - 0.5) < 0.01 for p in body["points"])


def test_bad_line_rejected():
    from app.stack import StackProvider
    c = TestClient(create_app(settings(provider=""), {"stack": StackProvider(None)}))
    r = c.post("/count", files={"image": ("s.jpg", _stack_jpeg(), "image/jpeg")},
               data={"item": "chairs", "line": "a,b"})
    assert r.status_code == 400
