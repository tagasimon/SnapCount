"""The counting instructions and the structured-output schema sent to the vision model."""

SYSTEM_PROMPT = """You count physical items in photographs for stocktaking. Accuracy and honesty matter more than giving a number.

How to count:
1. Identify the target item. If the user named one, count ONLY that item and ignore everything else. If no item was named, count the most numerous repeated item and say what it is.
2. Look at how the items are arranged (rows, stacks, piles, scattered) and split the image into sections: rows, columns, stacks or regions.
3. Count each section separately and methodically. Do not estimate a whole pile at a glance.
   - Nested or stacked items (e.g. plastic chairs): count the visible repeated edges, rims or seat lips in each stack.
   - Logs, pipes, poles: count the visible ends.
   - Boxes, bags, bricks: count visible faces row by row; only add hidden items if the arrangement makes them certain (e.g. a full regular block) and say so in the notes.
4. Add the section counts to get the total.

Confidence:
- "high": every counted item is clearly visible and separable.
- "medium": some overlap, small or distant items, or a few uncertain items; total could be off by a little.
- "low": heavy overlap, hidden items, blur, darkness, items cut off at the edges, or nested stacks where edges can't be separated.

If the photo does not contain enough information to count the items (target not present, far too blurry, or almost everything hidden), set countable to false, set count to null and explain why.

Never inflate confidence. Never count items you cannot see unless the arrangement makes them certain, and say when you did.
Always report your result by calling the report_count tool."""

REPORT_TOOL = {
    "name": "report_count",
    "description": "Report the item count for the photo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "item": {
                "type": "string",
                "description": "What was counted, e.g. 'plastic chairs'.",
            },
            "sections": {
                "type": "array",
                "description": "Per-section counts, in the order you counted them.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "e.g. 'left stack', 'row 2'"},
                        "count": {"type": "integer", "minimum": 0},
                    },
                    "required": ["label", "count"],
                },
            },
            "countable": {
                "type": "boolean",
                "description": "False if the photo does not contain enough information to count.",
            },
            "count": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Total count, or null if not countable.",
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {
                "type": "string",
                "description": "One or two short sentences on what limits accuracy, in plain language for a worker.",
            },
            "retake_tip": {
                "type": "string",
                "description": "If confidence is not high: one short, specific tip for a better photo. Otherwise empty.",
            },
        },
        "required": ["item", "sections", "countable", "count", "confidence", "notes", "retake_tip"],
    },
}


def user_text(item: str | None) -> str:
    if item:
        return f"Count the {item} in this photo."
    return "Count the main repeated item in this photo."
