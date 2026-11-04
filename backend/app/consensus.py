"""Combine several models' counts into one answer for Auto mode.

There's no way to know which model is right on a new photo, so Auto uses agreement:
- the combined count is the median of the models that found something
  (with three models, that's the middle one, which ignores one outlier);
- confidence comes from how closely the models agree;
- the model whose count is closest to the median is the one whose dots are shown.
"""

from __future__ import annotations

import statistics


def combine(counts: dict[str, int | None]) -> dict:
    valid = {m: c for m, c in counts.items() if c is not None and c > 0}
    if not valid:
        return {"count": None, "confidence": "low", "best_model": None, "spread": None,
                "agreeing": [], "notes": "None of the models found the item in this photo."}

    values = sorted(valid.values())
    median = statistics.median(values)
    best = min(valid, key=lambda m: (abs(valid[m] - median), m))
    # Report the count of the model closest to the median, so the number matches the dots
    # shown on the photo. With three models this is exactly the median.
    count = valid[best]

    if len(valid) == 1:
        only = next(iter(valid))
        return {"count": count, "confidence": "low", "best_model": only, "spread": None,
                "agreeing": [only],
                "notes": "Only one model found the item, so there was nothing to compare it with."}

    spread = (values[-1] - values[0]) / max(median, 1)
    agreeing = [m for m, c in valid.items() if abs(c - median) <= max(1, 0.1 * median)]
    if spread <= 0.10:
        confidence = "high"
        notes = f"The models agree closely ({values[0]}–{values[-1]})."
    elif spread <= 0.30:
        confidence = "medium"
        notes = f"The models roughly agree ({values[0]}–{values[-1]}). Check the dots."
    elif (len(agreeing) >= 2 and count >= 4
          and all(valid[m] < count for m in valid if m not in agreeing)):
        # Two models agree closely and one is far off: likely that one failed.
        confidence = "medium"
        others = [str(valid[m]) for m in valid if m not in agreeing]
        notes = (f"Two models agree on about {count}; one gave {', '.join(others)}. Check the dots.")
    else:
        confidence = "low"
        notes = f"The models disagree ({values[0]}–{values[-1]}). Check the dots or count a section by hand."

    # If any model saw nothing at all, the models disagree about what's in the photo, so
    # never call the result reliable.
    # One to three items is easy to get "right" by boxing the whole pile; don't call it sure.
    if count <= 3 and confidence == "high":
        confidence = "medium"
        notes += " Few items were found; if this is a pile or a stack, check the dots."

    missed = [m for m, c in counts.items() if not c]
    if missed and confidence != "low":
        confidence = "medium" if confidence == "high" else "low"
        notes += " One model found none, so treat this with care."
    return {"count": count, "confidence": confidence, "best_model": best,
            "spread": round(spread, 3), "agreeing": sorted(agreeing), "notes": notes}
