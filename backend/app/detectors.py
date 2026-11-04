"""Free, open-source counting models that run on your own server.

Each model is a zero-shot object detector: you give it a text prompt ("plastic chairs")
and it returns a box for every instance it finds. The count is the number of boxes left
after removing low-scoring ones and duplicates. Because every box has a position, the app
can mark each counted item on the photo so a worker can check the result.

All three models are Apache-2.0, published as safetensors (pure weights, no code), and
run through the standard Hugging Face `transformers` library. They are downloaded once on
first use and cached.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass

from PIL import Image

from .providers import CountProvider, CountResult, ProviderError

log = logging.getLogger("snapcount")


@dataclass(frozen=True)
class DetectorSpec:
    id: str
    name: str
    hf_repo: str
    description: str
    threshold: float          # minimum score to count a detection
    nms_iou: float = 0.5      # boxes overlapping more than this are treated as one item


# Thresholds were tuned on half of a 35-photo FSC-147 sample and checked on the other half
# (see eval/benchmark_fsc147.py). Override per model with e.g. OWLV2_THRESHOLD=0.2.
SPECS: dict[str, DetectorSpec] = {
    "owlv2": DetectorSpec(
        id="owlv2",
        name="OWLv2",
        hf_repo="google/owlv2-base-patch16-ensemble",
        description="Google's open-vocabulary detector. Best on small, dense items.",
        threshold=0.15,
    ),
    "grounding_dino": DetectorSpec(
        id="grounding_dino",
        name="Grounding DINO",
        hf_repo="IDEA-Research/grounding-dino-tiny",
        description="Strong at matching the item name to what's in the photo.",
        threshold=0.15,
    ),
    "omdet_turbo": DetectorSpec(
        id="omdet_turbo",
        name="OmDet-Turbo",
        hf_repo="omlab/omdet-turbo-swin-tiny-hf",
        description="The fastest of the three.",
        threshold=0.10,
    ),
}

MAX_DETECTIONS = 600
PILE_MIN_INSIDE = int(os.getenv("PILE_MIN_INSIDE", "3"))

# Long items photographed end-on: count the cut ends, not the long sides.
ELONGATED = ("log", "pipe", "pole", "rod", "tube", "timber", "firewood", "bamboo", "post", "beam")


def counts_ends(item: str | None) -> bool:
    t = (item or "").lower()
    return any(w in t for w in ELONGATED)


def pick_device() -> str:
    # On Apple Silicon, run any operation MPS doesn't support on the CPU instead of failing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    forced = os.getenv("TORCH_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child",
              "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
              "sheep": "sheep", "fish": "fish", "glasses": "glass", "boxes": "box",
              "knives": "knife", "leaves": "leaf", "shelves": "shelf", "loaves": "loaf",
              "tomatoes": "tomato", "potatoes": "potato", "mangoes": "mango"}


def singular(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def drop_parts(kept: list, inside: float = 0.85) -> list:
    """Drop a box that lies almost entirely inside a bigger kept box: it's usually part of
    that item (a chair's seat inside the whole chair), not a separate item."""
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    out = []
    order = sorted(range(len(kept)), key=lambda i: -area(kept[i][0]))  # big boxes first
    for i in order:
        b = kept[i][0]
        a = area(b)
        part = False
        for j in out:
            o = kept[j][0]
            ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0]))
            iy = max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
            # Only when the bigger box isn't hugely bigger (a pile-sized box shouldn't
            # swallow the items in it).
            if a > 0 and ix * iy / a > inside and area(o) < 6 * a:
                part = True
                break
        if not part:
            out.append(i)
    return [kept[i] for i in sorted(out, key=lambda i: -kept[i][1])]


def _prompt(item: str | None) -> str:
    """Detectors find single things, so ask for one of the item: asking for 'bricks'
    makes them box the whole pile as one object. 'bags of cement' -> 'bag of cement',
    'tree logs' -> 'tree log'."""
    text = " ".join((item or "object").strip().lower().split())
    words = text.split(" ")
    if " of " in f" {text} ":
        i = words.index("of")
        if i > 0:
            words[i - 1] = singular(words[i - 1])
    else:
        words[-1] = singular(words[-1])
    return " ".join(words)


class DetectorProvider(CountProvider):
    """Loads one Hugging Face detector on first use and keeps it in memory."""

    def __init__(self, spec: DetectorSpec):
        self.spec = spec
        self.name = spec.id
        env = os.getenv(f"{spec.id.upper()}_THRESHOLD")
        self.threshold = float(env) if env else spec.threshold
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()

    # ---- loading -------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch  # noqa: F401  (import here so the app starts fast without models)
            from transformers import AutoProcessor

            log.info("loading %s (%s)", self.spec.name, self.spec.hf_repo)
            kwargs = dict(use_safetensors=True)
            if self.spec.id == "owlv2":
                from transformers import Owlv2ForObjectDetection as M
            elif self.spec.id == "grounding_dino":
                from transformers import AutoModelForZeroShotObjectDetection as M
            else:
                from transformers import OmDetTurboForObjectDetection as M
            self._processor = AutoProcessor.from_pretrained(self.spec.hf_repo)
            model = M.from_pretrained(self.spec.hf_repo, **kwargs)
            self._device = pick_device()
            self._model = model.to(self._device).eval()
            log.info("%s ready on %s", self.spec.name, self._device)

    def warm_up(self) -> None:
        self._load()

    # ---- inference -----------------------------------------------------------

    def detect(self, img: Image.Image, item: str | None) -> tuple[list[list[float]], list[float]]:
        """Return (boxes in pixels [x0,y0,x1,y1], scores), before thresholding/NMS."""
        import torch

        self._load()
        text = _prompt(item)
        size = torch.tensor([[img.height, img.width]])
        proc, model, dev = self._processor, self._model, self._device

        with torch.inference_mode():
            if self.spec.id == "owlv2":
                inputs = proc(text=[[f"a photo of {text}"]], images=img, return_tensors="pt").to(dev)
                out = model(**inputs)
                # OWLv2 pads the image to a square; boxes come back relative to that square.
                side = max(img.width, img.height)
                res = proc.post_process_grounded_object_detection(
                    out, threshold=0.05, target_sizes=torch.tensor([[side, side]])
                )[0]
            elif self.spec.id == "grounding_dino":
                inputs = proc(images=img, text=f"{text}.", return_tensors="pt").to(dev)
                out = model(**inputs)
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=size
                )[0]
            else:
                inputs = proc(images=img, text=[text], return_tensors="pt").to(dev)
                out = model(**inputs)
                res = proc.post_process_grounded_object_detection(
                    out, text_labels=[[text]], threshold=0.05, nms_threshold=0.5, target_sizes=size
                )[0]

        boxes = res["boxes"].detach().float().cpu().tolist()
        scores = res["scores"].detach().float().cpu().tolist()
        return boxes, scores

    def detect_tiled(self, img: Image.Image, item: str | None, grid: int = 2, overlap: float = 0.2):
        """Run on the whole photo plus a grid of overlapping zoomed-in tiles, so small items
        in a dense pile are seen larger. Returns boxes in whole-image pixels."""
        boxes, scores = self.detect(img, item)
        W, H = img.width, img.height
        tw, th = W / grid, H / grid
        ox, oy = tw * overlap, th * overlap
        for gy in range(grid):
            for gx in range(grid):
                x0 = max(0, int(gx * tw - ox)); y0 = max(0, int(gy * th - oy))
                x1 = min(W, int((gx + 1) * tw + ox)); y1 = min(H, int((gy + 1) * th + oy))
                tb, ts = self.detect(img.crop((x0, y0, x1, y1)), item)
                for b, s in zip(tb, ts):
                    bx = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
                    # Skip boxes cut off by an inner tile edge; the neighbouring tile or the
                    # whole-photo pass sees that item complete.
                    cut = ((b[0] <= 2 and x0 > 0) or (b[1] <= 2 and y0 > 0)
                           or (b[2] >= (x1 - x0) - 2 and x1 < W) or (b[3] >= (y1 - y0) - 2 and y1 < H))
                    if not cut:
                        boxes.append(bx)
                        scores.append(s)
        return boxes, scores

    def filter(self, boxes, scores, width: int, height: int, threshold: float | None = None,
               item: str | None = None):
        """Threshold, drop pile boxes, and merge duplicates. Returns kept (box, score).

        The pile rule (a box with several smaller detections inside it is the pile, not one
        item) only runs when the normal pass finds three or fewer items: that's the "boxed
        the whole pile as one log" failure. In busy photos it would break up good boxes."""
        kept = self._filter(boxes, scores, width, height, threshold, item, piles=False)
        if len(kept) <= 3:
            broken_up = self._filter(boxes, scores, width, height, threshold, item, piles=True)
            if len(broken_up) > len(kept):
                return broken_up
        return kept

    def _filter(self, boxes, scores, width, height, threshold, item, piles: bool):
        import torch
        from torchvision.ops import nms

        thr = self.threshold if threshold is None else threshold

        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        if counts_ends(item):
            # Keep only roughly square boxes: the round ends, not the long sides.
            pairs = [(b, s) for b, s in zip(boxes, scores)
                     if max((b[2] - b[0]) / max(1.0, b[3] - b[1]), (b[3] - b[1]) / max(1.0, b[2] - b[0])) <= 1.8]
            boxes, scores = [p[0] for p in pairs], [p[1] for p in pairs]

        # Weaker candidates are used only to spot "pile" boxes and fill them in.
        weak = []
        for b, s in zip(boxes, scores):
            if s < thr * 0.5:
                continue
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(width, b[2]), min(height, b[3])
            if x1 > x0 and y1 > y0:
                weak.append(([x0, y0, x1, y1], s))

        kept, group_fill = [], []
        for b, s in weak:
            if s < thr:
                continue
            # A box covering most of the photo is the whole pile, not one item.
            if area(b) > 0.5 * width * height:
                continue
            # A box with several much smaller detections inside it is a pile (e.g. a stack of
            # logs boxed as "log"): drop it and count what's inside instead.
            inside = [(ib, isc) for ib, isc in weak
                      if area(ib) < 0.35 * area(b)
                      and b[0] <= (ib[0] + ib[2]) / 2 <= b[2] and b[1] <= (ib[1] + ib[3]) / 2 <= b[3]]
            if piles and len(inside) >= PILE_MIN_INSIDE:
                group_fill += inside
                continue
            kept.append((b, s))
        # Add the items found inside pile boxes (deduplicated by NMS below).
        kept += [(b, s) for b, s in group_fill if area(b) <= 0.5 * width * height]
        if not kept:
            return []
        bt = torch.tensor([k[0] for k in kept], dtype=torch.float32)
        st = torch.tensor([k[1] for k in kept], dtype=torch.float32)
        idx = nms(bt, st, self.spec.nms_iou).tolist()[:MAX_DETECTIONS]
        kept = [kept[i] for i in idx]
        return drop_parts(kept)

    async def count(self, image: bytes, media_type: str, item: str | None) -> CountResult:
        import asyncio

        img = Image.open(io.BytesIO(image)).convert("RGB")
        # DETECT_TILED=1 also looks at zoomed-in tiles: ~5x slower, sometimes better on dense piles.
        run = self.detect_tiled if os.getenv("DETECT_TILED", "0") == "1" else self.detect
        try:
            boxes, scores = await asyncio.to_thread(run, img, item)
        except Exception as e:  # model load or inference failure
            log.exception("%s failed", self.spec.name)
            raise ProviderError(f"{self.spec.name} failed: {e}") from e

        kept = self.filter(boxes, scores, img.width, img.height, item=item)
        n = len(kept)
        points = [
            [round((b[0] + b[2]) / 2 / img.width, 4), round((b[1] + b[3]) / 2 / img.height, 4)]
            for b, _ in kept
        ]
        borderline = sum(1 for _, s in kept if s < self.threshold * 1.5)
        confidence, notes, tip = self._judge(n, borderline)
        if counts_ends(item) and n:
            notes = "Counted the cut ends. Photograph the pile end-on so every end is visible. " + notes
        if n >= MAX_DETECTIONS:
            notes = f"Stopped at {MAX_DETECTIONS}; there may be more. " + notes

        result = CountResult(
            item=(item or "items").strip(),
            count=n if n > 0 else None,
            confidence=confidence,
            notes=notes,
            retake_tip=tip,
        )
        result.model = self.spec.name
        result.points = points
        return result

    @staticmethod
    def _judge(n: int, borderline: int) -> tuple[str, str, str]:
        """A detector can't judge hidden items, so confidence reflects how clear-cut its
        detections are, and is never 'high' on its own. Agreement between models (Auto)
        is what can raise it."""
        if n == 0:
            return ("low", "No matching items were found. Check the item name, or the items may be too small or hidden.",
                    "Move closer and name the item simply, e.g. 'chairs' or 'logs'.")
        share = borderline / n
        if share > 0.4:
            return ("low", f"Many of the {n} detections were uncertain, so some may be wrong or missed.",
                    "Take a sharper, closer photo with even light.")
        return ("medium", "Each marked dot is one counted item. Items hidden behind others can't be seen.",
                "Check the dots on the photo for missed or doubled items.")


def make_detectors(enabled: list[str]) -> dict[str, DetectorProvider]:
    out = {}
    for mid in enabled:
        if mid in SPECS:
            out[mid] = DetectorProvider(SPECS[mid])
        else:
            log.warning("unknown model id %r ignored", mid)
    return out
