# Copyright © 2025 Apple Inc.

import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def _next_json_start(text, start):
    """Return the index of the next ``{`` or ``[`` at or after ``start``.

    Used to skip unparseable spans (malformed objects, stray ``<tool_call>``
    markers, trailing prose) and try again at the next JSON value.
    """
    positions = [i for i in (text.find("{", start), text.find("[", start)) if i != -1]
    return min(positions) if positions else None


def _iter_json_values(text):
    """Yield each top-level JSON value in ``text`` using raw_decode.

    Tolerates back-to-back JSON objects (e.g. ``{...}\n{...}``) that a single
    ``json.loads`` would reject with "Extra data", as well as trailing prose,
    stray tool markers, and malformed leading objects.
    """
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = _next_json_start(text, i + 1)
            if i is None:
                break
            continue
        yield obj
        i = end


def parse_tool_call(text, tools=None):
    text = text.strip()
    if not text:
        raise ValueError("Empty tool call")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    values = [
        value
        for value in _iter_json_values(text)
        if isinstance(value, dict) and isinstance(value.get("name"), str)
    ]
    if not values:
        raise ValueError(f"Could not parse tool call: {text!r}")
    if len(values) == 1:
        return values[0]
    return values
