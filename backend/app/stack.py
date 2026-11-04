"""Count nested or stacked items (plastic chairs, plates, buckets, basins, cups, trays).

In a stack, every item hides the one below it, so a detector sees one object. What
*is* visible is a repeated edge: one seat lip, rim or plate edge per item, with a dark gap
under each. This counter:

1. finds the stack(s) with a detector (or uses the middle of the photo if none is found),
2. inside each stack, reads brightness down several vertical strips and counts the
   repeated dark gaps,
3. takes the median across strips, so one strip crossing a logo or shadow doesn't
   decide the answer.

It assumes the stack is roughly upright and photographed from the side, with the edges
facing the camera. Each counted edge gets a dot, so the result can be checked.
"""

from __future__ import annotations

import io
import statistics

import numpy as np
from PIL import Image

from .providers import CountProvider, CountResult

NESTABLE = ("chair", "plate", "bowl", "bucket", "basin", "cup", "tray", "crate", "stool",
            "tyre", "tire", "pot", "pan", "dish", "glass", "tumbler", "container", "jerrycan",
            "saucepan", "sufuria", "mattress", "pallet", "sheet", "board")


def is_nestable(item: str | None) -> bool:
    t = (item or "").lower()
    return "stack" in t or any(w in t for w in NESTABLE)


def _strip_gaps(gray: np.ndarray, x0: int, x1: int, min_gap: int) -> list[int]:
    """Row indices of the repeated dark gaps down one vertical strip."""
    h = gray.shape[0]
    prof = gray[:, x0:x1].mean(axis=1)
    k = max(5, int(h * 0.06)) | 1
    trend = np.convolve(prof, np.ones(k) / k, mode="same")
    # Smooth over ~1% of the height so JPEG ripples don't split one gap into two.
    sk = max(3, int(h * 0.012)) | 1
    d = np.convolve(prof - trend, np.ones(sk) / sk, mode="same")
    cut = -0.35 * float(np.std(d))
    out: list[int] = []
    for i in range(2, h - 2):
        if d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] < cut:
            if not out or i - out[-1] >= min_gap:
                out.append(i)
            elif d[i] < d[out[-1]]:
                out[-1] = i
    return out


def evenly_spaced_run(rows: list[int]) -> list[int]:
    """Keep the longest run of evenly spaced edges.

    Stacked items repeat at a steady spacing; a strip that also crosses a backrest, a logo
    or the floor picks up extra gaps at odd distances. Those fall outside the run."""
    if len(rows) < 3:
        return rows
    diffs = [b - a for a, b in zip(rows, rows[1:])]
    m = statistics.median(diffs)
    best, cur = [rows[0]], [rows[0]]
    for prev, nxt, d in zip(rows, rows[1:], diffs):
        if 0.55 * m <= d <= 1.6 * m:
            cur.append(nxt)
        else:
            cur = [nxt]
        if len(cur) > len(best):
            best = list(cur)
    return best


def _gap_depth(prof: np.ndarray, r: int, win: int) -> float:
    """How much darker the gap at row r is than its surroundings."""
    lo, hi = max(0, r - win), min(len(prof), r + win + 1)
    return float(np.median(prof[lo:hi]) - prof[r])


def _strip_count(gray: np.ndarray, x0: int, x1: int, min_gap: int) -> tuple[int, list[int], float]:
    """Items counted down one strip: (count, edge rows, spacing).

    Each item in a stack shows one deep dark gap under its seat lip or rim. The top
    item's seat surface and the floor below the bottom item give extra, shallow gaps:
    those are dropped, except that one shallow gap right below the last deep one, at the
    same spacing, is the bottom item and is counted."""
    rows = _strip_gaps(gray, x0, x1, min_gap)
    if len(rows) < 2:
        return len(rows), rows, 0.0
    prof = gray[:, x0:x1].mean(axis=1)
    win = max(3, int(gray.shape[0] * 0.03))
    depth = [_gap_depth(prof, r, win) for r in rows]
    ref = float(np.median(sorted(depth)[len(depth) // 2:]))   # typical depth of the strong gaps
    strong = [r for r, d in zip(rows, depth) if d >= 0.45 * ref]
    run = evenly_spaced_run(strong) if len(strong) >= 3 else strong
    if len(run) < 2:
        return len(run), run, 0.0
    spacing = float(statistics.median([b - a for a, b in zip(run, run[1:])]))
    # A shallow gap right after the last deep one, at the same spacing: the bottom item.
    below = [r for r in rows if r > run[-1] and 0.6 * spacing <= r - run[-1] <= 1.5 * spacing]
    if below:
        run = run + [below[0]]
    return len(run), run, spacing


def count_layers(gray: np.ndarray, strips: int = 16) -> tuple[int, list[int], float, int]:
    """Return (items, edge rows of a strip with that count, agreement 0..1, its column).

    Reads many vertical strips. Strips through the parts that repeat once per item
    (seat fronts, seat sides, legs) share the stack's spacing; strips through a backrest
    or a logo pick up details at a different spacing. Only strips at the most common
    spacing vote, so the backrest isn't double counted."""
    h, w = gray.shape
    if h < 40 or w < 20:
        return 0, [], 0.0, w // 2
    min_gap = max(3, int(h * 0.02))
    sw = max(3, int(w * 0.05))
    results = []
    for s in range(strips):
        x0 = int(w * (0.06 + 0.88 * s / max(1, strips - 1))) - sw // 2
        x0 = max(0, min(w - sw, x0))
        n, rows, spacing = _strip_count(gray, x0, x0 + sw, min_gap)
        if n >= 2 and spacing > 0:
            results.append((n, rows, spacing, x0 + sw // 2))
    if not results:
        return 0, [], 0.0, w // 2

    # The stack's spacing: the one most strips agree on (within 15%).
    def neighbours(sp):
        return [r for r in results if abs(r[2] - sp) <= 0.15 * sp]
    stack_spacing = max((r[2] for r in results), key=lambda sp: (len(neighbours(sp)), sp))
    voters = neighbours(stack_spacing)

    tally: dict[int, int] = {}
    for r in voters:
        tally[r[0]] = tally.get(r[0], 0) + 1
    top = max(tally.values())
    best = min(c for c, v in tally.items() if v == top)  # ties: the smaller, safer count
    agree = top / len(voters)
    chosen = min((r for r in voters if r[0] == best), key=lambda r: abs(r[3] - w / 2))
    return best, chosen[1], agree, chosen[3]


def _band_along(gray: np.ndarray, p0, p1, offset: float, half: int) -> np.ndarray:
    """Pixels in a band `2*half+1` wide following the line p0->p1, shifted sideways by
    `offset` pixels. Rows run along the line."""
    h, w = gray.shape
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(2, int(length))
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    nx, ny = -(p1[1] - p0[1]) / length, (p1[0] - p0[0]) / length  # unit normal
    cols = []
    for o in range(-half, half + 1):
        xx = np.clip((xs + nx * (offset + o)).astype(int), 0, w - 1)
        yy = np.clip((ys + ny * (offset + o)).astype(int), 0, h - 1)
        cols.append(gray[yy, xx])
    return np.stack(cols, axis=1)


def count_along_line(gray: np.ndarray, p0, p1) -> tuple[int, list[float], float]:
    """Count the repeated edges crossed by a line the worker drew along one side of a stack.

    Reads seven bands beside the line (so a small wobble in where the finger went doesn't
    matter) and takes the most common count. Returns (count, edge positions as 0..1 along
    the line, agreement 0..1)."""
    h, w = gray.shape
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if length < 30:
        return 0, [], 0.0
    half = max(3, int(0.02 * max(w, h)))
    min_gap = max(3, int(length * 0.04))
    runs = []
    for off in range(-3 * half, 3 * half + 1, half):
        band = _band_along(gray, p0, p1, off, half)
        runs.append(evenly_spaced_run(_strip_gaps(band, 0, band.shape[1], min_gap)))
    counts = [len(r) for r in runs]
    tally: dict[int, int] = {}
    for c in counts:
        tally[c] = tally.get(c, 0) + 1
    top = max(tally.values())
    best = min(c for c, v in tally.items() if v == top) if top > 1 else int(round(statistics.median(counts)))
    # Positions from the band closest to the drawn line that has the chosen count.
    order = sorted(range(len(runs)), key=lambda i: abs(i - len(runs) // 2))
    rows = next((runs[i] for i in order if len(runs[i]) == best), runs[len(runs) // 2][:best])
    n = max(2, int(length))
    agree = tally.get(best, 0) / len(counts)
    return best, [r / (n - 1) for r in rows], agree


def _merge_regions(boxes: list[list[float]], W: int, H: int) -> list[list[float]]:
    """Merge overlapping detector boxes into stack regions."""
    regions: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        for r in regions:
            if b[0] < r[2] and b[2] > r[0] and b[1] < r[3] and b[3] > r[1]:
                r[0], r[1] = min(r[0], b[0]), min(r[1], b[1])
                r[2], r[3] = max(r[2], b[2]), max(r[3], b[3])
                break
        else:
            regions.append(list(b))
    # Ignore slivers; keep at most 4 stacks.
    regions = [r for r in regions if (r[2] - r[0]) > 0.08 * W and (r[3] - r[1]) > 0.15 * H]
    return regions[:4]


class StackProvider(CountProvider):
    name = "stack"

    def __init__(self, locator=None):
        # A DetectorProvider used to find where the stacks are. Optional.
        self.locator = locator

    def warm_up(self) -> None:
        if self.locator is not None:
            self.locator.warm_up()

    async def count(self, image: bytes, media_type: str, item: str | None,
                    line: tuple[float, float, float, float] | None = None) -> CountResult:
        import asyncio

        img = Image.open(io.BytesIO(image)).convert("RGB")
        W, H = img.size
        name = (item or "items").strip()

        if line is not None:
            return self._count_line(img, name, line)

        regions: list[list[float]] = []
        if self.locator is not None:
            try:
                boxes, scores = await asyncio.to_thread(self.locator.detect, img, item)
                kept = [b for b, s in zip(boxes, scores) if s >= 0.2]
                regions = _merge_regions(kept, W, H)
            except Exception:  # fall back to the middle of the photo
                regions = []
        if not regions:
            regions = [[W * 0.15, H * 0.05, W * 0.85, H * 0.95]]

        gray_all = np.asarray(img.convert("L"), dtype=float)
        total, points, agreements = 0, [], []
        for r in regions:
            x0, y0, x1, y1 = (int(max(0, r[0])), int(max(0, r[1])), int(min(W, r[2])), int(min(H, r[3])))
            n, rows, agree, col = count_layers(gray_all[y0:y1, x0:x1])
            total += n
            agreements.append(agree)
            points += [[round((x0 + col) / W, 4), round((y0 + y) / H, 4)] for y in rows]

        if total == 0:
            return CountResult(item=name, count=None, confidence="low",
                               notes="Couldn't find repeated edges. Photograph the stack from the side, straight on.",
                               retake_tip="Stand to the side of the stack so every seat edge or rim faces you.")

        agree = min(agreements)
        confidence = "medium" if agree >= 0.6 else "low"
        result = CountResult(
            item=name,
            count=total,
            confidence=confidence,
            notes=("Counted as a stack: one dot per visible edge. "
                   + ("" if len(regions) == 1 else f"{len(regions)} stacks found. ")
                   + "Items at the very top or bottom may be missed."),
            retake_tip="" if confidence == "medium" else
                       "Photograph the stack from the side with the edges facing you and nothing in front.",
        )
        result.model = "Stacked items"
        result.points = points
        return result

    def _count_line(self, img: Image.Image, name: str, line) -> CountResult:
        W, H = img.size
        x0, y0, x1, y1 = line
        p0, p1 = (x0 * W, y0 * H), (x1 * W, y1 * H)
        gray = np.asarray(img.convert("L"), dtype=float)
        n, positions, agree = count_along_line(gray, p0, p1)
        if n == 0:
            return CountResult(item=name, count=None, confidence="low",
                               notes="No repeated edges along that line. Draw it across every seat edge or rim, from the top item to the bottom one.",
                               retake_tip="Draw the line along one side of the stack, crossing each edge once.")
        points = [[round(x0 + t * (x1 - x0), 4), round(y0 + t * (y1 - y0), 4)] for t in positions]
        result = CountResult(
            item=name,
            count=n,
            confidence="medium" if agree >= 0.6 else "low",
            notes="Counted the edges along your line, one per item. Check the numbers.",
            retake_tip="" if agree >= 0.6 else "Try the line a little to one side, where the edges are clearest.",
        )
        result.model = "Stacked items"
        result.points = points
        return result
