"""Measure SnapCount against real photos with known counts.

1. Put photos in eval/photos/
2. Add a row per photo to eval/labels.csv (count the items yourself, carefully)
3. Run:  python eval/run_eval.py --server http://localhost:8000 --model auto
   (repeat with --model owlv2 / grounding_dino / omdet_turbo to compare)
Results are written to eval/results.csv and summarised in the terminal.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--labels", default=HERE / "labels.csv", type=Path)
    p.add_argument("--photos", default=HERE / "photos", type=Path)
    p.add_argument("--out", default=HERE / "results.csv", type=Path)
    p.add_argument("--token", default=os.getenv("APP_TOKEN", ""))
    p.add_argument("--model", default="auto", help="auto, owlv2, grounding_dino, omdet_turbo or claude")
    args = p.parse_args()

    rows = list(csv.DictReader(args.labels.open()))
    if not rows:
        print("labels.csv has no rows yet. Add your photos and their true counts first.")
        return 1

    headers = {"X-App-Token": args.token} if args.token else {}
    results = []
    with httpx.Client(timeout=300) as client:
        for row in rows:
            path = args.photos / row["filename"]
            if not path.exists():
                print(f"skip {row['filename']}: file not found")
                continue
            true_count = int(row["true_count"])
            should_warn = row.get("should_warn", "").strip().lower() in ("yes", "y", "true", "1")

            started = time.perf_counter()
            with path.open("rb") as f:
                r = client.post(
                    f"{args.server.rstrip('/')}/count",
                    files={"image": (path.name, f, "image/jpeg")},
                    data={"item": row.get("item", ""), "model": args.model},
                    headers=headers,
                )
            latency = time.perf_counter() - started

            if r.status_code != 200:
                print(f"{row['filename']}: server error {r.status_code} {r.text[:120]}")
                results.append({**row, "ai_count": "", "abs_error": "", "pct_error": "",
                                "confidence": "", "warned": "", "warning_correct": "",
                                "latency_s": f"{latency:.1f}", "notes": f"HTTP {r.status_code}"})
                continue

            body = r.json()
            ai = body["count"]
            warned = body["retake_suggested"] or ai is None
            abs_err = "" if ai is None else abs(ai - true_count)
            pct_err = "" if ai is None or true_count == 0 else f"{100 * abs(ai - true_count) / true_count:.1f}"
            results.append({
                **row,
                "ai_count": "" if ai is None else ai,
                "abs_error": abs_err,
                "pct_error": pct_err,
                "confidence": body["confidence"],
                "warned": "yes" if warned else "no",
                "warning_correct": "yes" if warned == should_warn else "no",
                "latency_s": f"{latency:.1f}",
                "notes": body["notes"],
            })
            shown = "can't count" if ai is None else ai
            print(f"{row['filename']:<28} true={true_count:<5} ai={shown!s:<12} "
                  f"err={pct_err or '-':>6}%  conf={body['confidence']:<6} warned={'yes' if warned else 'no'}")

    if not results:
        return 1
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    scored = [r for r in results if r["pct_error"] != ""]
    print()
    print(f"Photos: {len(results)}   counted: {len(scored)}   refused: "
          f"{sum(1 for r in results if r['ai_count'] == '' and r['confidence'])}")
    if scored:
        errors = sorted(float(r["pct_error"]) for r in scored)
        mean = sum(errors) / len(errors)
        median = errors[len(errors) // 2]
        within5 = sum(1 for e in errors if e <= 5)
        print(f"Mean error: {mean:.1f}%   median: {median:.1f}%   within 5%: {within5}/{len(errors)}")
        for conf in ("high", "medium", "low"):
            sub = [float(r["pct_error"]) for r in scored if r["confidence"] == conf]
            if sub:
                print(f"  {conf:<6} confidence: {len(sub)} photos, mean error {sum(sub) / len(sub):.1f}%")
    judged = [r for r in results if r["warning_correct"]]
    if judged:
        ok = sum(1 for r in judged if r["warning_correct"] == "yes")
        print(f"Warning behaved as expected on {ok}/{len(judged)} photos")
    print(f"\nFull results: {args.out}")
    print("A handful of photos is not proof of accuracy. Aim for 20+ per item type before trusting the numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
