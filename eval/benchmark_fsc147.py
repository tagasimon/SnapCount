"""Benchmark the free models on FSC-147 photos with known counts.

Runs every model once per photo, caches the raw detections, then:
  1. picks each model's score threshold on the EVEN-numbered photos (tuning half),
  2. reports accuracy on the ODD-numbered photos (held-out half), per model and for
     the Auto consensus.

Usage (from the repo root, backend venv active):
  python eval/benchmark_fsc147.py --images /path/to/fsc/images --labels /path/to/picks.csv
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from PIL import Image  # noqa: E402

from app.consensus import combine  # noqa: E402
from app.detectors import SPECS, DetectorProvider  # noqa: E402


def pct_err(pred, true):
    return abs(pred - true) / true * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--cache", default=Path(__file__).with_name("fsc_raw.json"), type=Path)
    ap.add_argument("--models", default=",".join(SPECS))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.labels.open()))
    models = {m: DetectorProvider(SPECS[m]) for m in args.models.split(",")}
    raw = json.loads(args.cache.read_text()) if args.cache.exists() else {}

    # 1. raw detections (cached)
    for mid, prov in models.items():
        times = []
        for r in rows:
            key = f"{mid}|{r['filename']}"
            if key in raw:
                continue
            img = Image.open(args.images / r["filename"]).convert("RGB")
            t = time.perf_counter()
            boxes, scores = prov.detect(img, r["item"])
            times.append(time.perf_counter() - t)
            raw[key] = {"boxes": boxes, "scores": scores, "w": img.width, "h": img.height}
        if times:
            print(f"{mid}: {len(times)} photos, median {statistics.median(times):.1f}s per photo")
        args.cache.write_text(json.dumps(raw))

    def counts(mid, thr):
        prov = models[mid]
        out = []
        for r in rows:
            d = raw[f"{mid}|{r['filename']}"]
            out.append(len(prov.filter(d["boxes"], d["scores"], d["w"], d["h"], threshold=thr, item=r["item"])))
        return out

    tune = [i for i in range(len(rows)) if i % 2 == 0]
    test = [i for i in range(len(rows)) if i % 2 == 1]
    truth = [int(r["true_count"]) for r in rows]

    # 2. tune thresholds on the even half (minimise median % error)
    grid = [round(0.05 * k, 2) for k in range(2, 9)]  # 0.10 .. 0.40
    best = {}
    for mid in models:
        scored = []
        for thr in grid:
            c = counts(mid, thr)
            scored.append((statistics.median(pct_err(c[i], truth[i]) for i in tune), thr))
        best[mid] = min(scored)[1]
    print("\nThresholds picked on the tuning half:", best)

    # 3. report on the held-out half
    per = {mid: counts(mid, best[mid]) for mid in models}
    print(f"\nHeld-out photos: {len(test)}")
    print(f"{'photo':<10}{'item':<13}{'true':>5}  " + "".join(f"{m[:12]:>13}" for m in models) + f"{'AUTO':>8}")
    auto_errs, errs = [], {m: [] for m in models}
    for i in test:
        r = rows[i]
        vals = {m: per[m][i] for m in models}
        combo = combine({m: (v if v > 0 else None) for m, v in vals.items()})
        auto = combo["count"]
        for m in models:
            errs[m].append(pct_err(vals[m], truth[i]))
        auto_errs.append(pct_err(auto or 0, truth[i]))
        print(f"{r['filename']:<10}{r['item'][:12]:<13}{truth[i]:>5}  "
              + "".join(f"{vals[m]:>13}" for m in models) + f"{auto or 0:>8}  ({combo['confidence']})")

    def summary(e):
        within = sum(1 for x in e if x <= 10)
        return f"median {statistics.median(e):5.1f}%  mean {statistics.mean(e):5.1f}%  within 10%: {within}/{len(e)}"

    print()
    for m in models:
        print(f"{SPECS[m].name:<15} {summary(errs[m])}")
    print(f"{'Auto (median)':<15} {summary(auto_errs)}")


if __name__ == "__main__":
    main()
