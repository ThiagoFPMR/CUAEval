#!/usr/bin/env python3
"""Verify Holo3's coordinate convention against a served endpoint -- no VM needed.

Fires ONE localization call ("Localize <element> ... Click(x, y)") at your SGLang
server on a static screenshot, then shows where each coordinate-space hypothesis
would put the click. This is how you set HOLO_COORD_DIVISOR before running the
full OSWorld benchmark, where a wrong divisor silently misses every target.

The localization request is built with the SAME machinery the agent uses
(mm_agents.holo3_agent / holo3_format), so what you verify here is exactly what
the agent will do.

Read the RAW model coords it prints:
  * values ~0-1000        -> model emits [0,1000]  -> HOLO_COORD_DIVISOR=1000 (default)
  * values ~0.0-1.0       -> normalized [0,1]       -> HOLO_COORD_DIVISOR=1
  * values ~ image pixels -> raw pixel space        -> HOLO_COORD_DIVISOR=<none>, coords are px
The saved --draw image places a labeled dot per hypothesis; whichever lands on
the element you named is your convention.

Usage:
  export OPENAI_BASE_URL=http://localhost:8090/v1 OPENAI_API_KEY=EMPTY
  python scripts/python/holo3_grounding_smoketest.py \
      --image shot.png --element "the File menu in the top-left" \
      --model holo3-gimp-pruned --draw grounded.png

Tip: any GIMP screenshot works, e.g. one from your downloaded AgentNet subset
(/data/agentnet/ubuntu_images/<uuid>.png).
"""
import argparse
import base64
import os
import sys
from io import BytesIO

# Allow `import mm_agents...` when run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image, ImageDraw  # noqa: E402

from mm_agents.holo3_agent import Holo3Agent, json_schema_response_format  # noqa: E402
from mm_agents.holo3_format import LocalizerOutput, build_localization_messages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="screenshot to localize on")
    ap.add_argument("--element", required=True, help="element description / instruction")
    ap.add_argument("--model", default=os.environ.get("HOLO_MODEL", "holo3-gimp-pruned"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8090/v1"))
    ap.add_argument("--draw", default=None, help="save a copy with a marker per hypothesis")
    args = ap.parse_args()

    os.environ["OPENAI_BASE_URL"] = args.base_url  # honored by Holo3Agent.call_llm

    # Normalize to PNG bytes so the data-URL mime is correct regardless of input.
    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    buf = BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Grounding is single-shot with thinking off (H's element-localization surface).
    agent = Holo3Agent(model=args.model, enable_thinking=False)
    agent.reset()

    messages = agent._inline_images(
        build_localization_messages(instruction=args.element, image=b64))

    print(f"[cfg] endpoint={args.base_url}  model={args.model}  image={W}x{H}")
    print(f"[cfg] element={args.element!r}\n")
    # Grounding runs with thinking off; force clean JSON so a bare endpoint that
    # rambles still yields a Click(x, y) to read the coordinate convention from.
    raw = agent.call_llm(messages, 256, temperature=0.0,
                         response_format=json_schema_response_format(LocalizerOutput, "localize"))
    print("=== RAW model response ===")
    print(raw or "(empty)")
    print("==========================\n")

    obj = agent._parse_json(raw)
    if not obj or "x" not in obj or "y" not in obj:
        print("[fail] could not parse a Click(x, y) from the response. If the model "
              "rambled instead of emitting JSON, it may need guided/structured decoding.")
        return 1
    x, y = float(obj["x"]), float(obj["y"])
    print(f"[raw coords] x={x}  y={y}\n")

    # Resolve under each hypothesis and clamp into the image.
    def clamp(px, py):
        return max(0, min(W - 1, round(px))), max(0, min(H - 1, round(py)))

    hypotheses = {
        "[0,1000] (default)": clamp(x / 1000 * W, y / 1000 * H),
        "normalized [0,1]":   clamp(x * W, y * H),
        "raw pixels":         clamp(x, y),
    }
    print("Pixel each convention resolves to (image is {}x{}):".format(W, H))
    for name, (px, py) in hypotheses.items():
        print(f"  {name:22s} -> ({px}, {py})")

    if args.draw:
        canvas = img.copy()
        d = ImageDraw.Draw(canvas)
        colors = {"[0,1000] (default)": (255, 0, 0), "normalized [0,1]": (0, 160, 255),
                  "raw pixels": (0, 200, 0)}
        r = max(6, W // 120)
        for name, (px, py) in hypotheses.items():
            c = colors[name]
            d.ellipse([px - r, py - r, px + r, py + r], outline=c, width=3)
            d.line([px - r * 2, py, px + r * 2, py], fill=c, width=2)
            d.line([px, py - r * 2, px, py + r * 2], fill=c, width=2)
            d.text((px + r + 2, py - r), name.split()[0], fill=c)
        canvas.save(args.draw)
        print(f"\n[draw] saved {args.draw} — red=[0,1000], blue=normalized, green=raw px.")
        print("Whichever marker sits on the element you named is your HOLO_COORD_DIVISOR.")

    print("\nNext: set HOLO_COORD_DIVISOR accordingly (1000 / 1 / — for raw px) and run the benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
