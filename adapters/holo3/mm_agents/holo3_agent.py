"""OSWorld agent for a self-hosted Holo3 (e.g. the GIMP-pruned Holo3-35B-A3B).

Drives a stock OpenAI-compatible endpoint (SGLang / vLLM) that serves Holo3, in
the model's native two-pass loop -- unlike mm_agents/surferH, which delegates the
whole task to H Company's hosted Agent Platform. Points at OPENAI_BASE_URL, so it
plugs into the same tunnel-to-vast flow run_osworld_here.sh already uses.

Per step (mirrors H's hai-cookbook navigation_step.py):
  1. navigation call  -> {note, thought, action}. The action names a UI element
     by description and carries rough coordinates.
  2. localization call -> {action: click, x, y} for coordinate actions
     (click/drag). The localizer's coords are authoritative; nav coords are the
     fallback if it fails.

The prompts/schemas come from holo3_format.py -- a byte-for-byte copy of the
module used to build the EASY-EP calibration set -- so the served (pruned) model
sees exactly the distribution it was calibrated/pruned for.

Coordinates: Holo3 emits model-space coords (H's hub convention is [0, 1000]);
we rescale to OSWorld screen pixels via the live screenshot size. The divisor is
the one thing to sanity-check with a grounding smoke test -- override with
HOLO_COORD_DIVISOR if clicks land off (e.g. set it to 1 if the model turns out to
emit already-normalized [0,1] coords).

Contract matches mm_agents/omnibrowse_agent.py: reset(...) then
predict(instruction, obs) -> (response_str, [pyautogui_code | "WAIT"|"DONE"|"FAIL"]).
"""
import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import openai
from PIL import Image

from mm_agents.holo3_format import (
    ClickAction,
    NavigationStep,
    build_localization_messages,
    build_navigation_messages,
)

logger = logging.getLogger("desktopenv.holo3_agent")

MAX_RETRY_TIMES = 3


def json_schema_response_format(model_cls, name: str) -> dict:
    """OpenAI/SGLang guided-decoding format that forces the model to emit this schema.

    Without this the served Holo3 reasons in prose instead of returning the
    structured {note,thought,action} / {action,x,y} JSON, so the parse fails.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model_cls.model_json_schema(), "strict": True},
    }

# Holo click_type -> pyautogui pointer function.
_CLICK_FN = {
    "left": "click", "double": "doubleClick", "right": "rightClick",
    "middle": "middleClick", "triple": "tripleClick", "move": "moveTo",
}


class Holo3Agent:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "holo3-gimp-pruned",
        max_tokens: int = 1024,
        top_p: float = 0.95,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coord_divisor: Optional[float] = None,
        scroll_clicks: int = 3,
        invert_scroll: bool = False,
        history_length: int = 6,
        max_parse_retries: int = 3,
        retry_temperature: float = 0.0,
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        # Model coordinate space. H's hub documents [0, 1000]; env overrides so a
        # grounding smoke test can retune without a rebuild.
        self.coord_divisor = float(
            coord_divisor if coord_divisor is not None
            else os.environ.get("HOLO_COORD_DIVISOR", "1000"))
        self.scroll_clicks = scroll_clicks
        # pyautogui.scroll is "positive = up"; flip if a trial scrolls the wrong way.
        self.invert_scroll = invert_scroll
        self.history_length = history_length
        self.max_parse_retries = max(1, max_parse_retries)
        self.retry_temperature = retry_temperature

        assert action_space == "pyautogui", "Holo3Agent only supports pyautogui"
        assert observation_type == "screenshot", "Holo3Agent only supports screenshot"

        self._reset_memory()

    def _reset_memory(self) -> None:
        self.screenshots: List[str] = []       # base64 PNG per step
        self.notes: List[str] = []             # model note per step
        self.thoughts: List[str] = []          # model thought per step
        self.action_log: List[str] = []        # short human-readable action per step
        self.responses: List[str] = []         # raw nav+loc JSON per step

    # ------------------------------------------------------------------ predict
    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        png = obs["screenshot"]
        width, height = Image.open(BytesIO(png)).size
        b64 = base64.b64encode(png).decode("utf-8")
        self.screenshots.append(b64)

        nav = self._navigate(instruction, b64)
        if not nav or not isinstance(nav.get("action"), dict):
            logger.error("Holo3 navigation produced no usable action.")
            self.notes.append(""); self.thoughts.append("")
            self.action_log.append(""); self.responses.append(json.dumps(nav or {}))
            return json.dumps(nav or {}), []

        self.notes.append(str(nav.get("note", "")))
        self.thoughts.append(str(nav.get("thought", "")))

        pyautogui_code, log, loc = self._translate(nav["action"], b64, width, height)
        self.action_log.append(log)
        self.responses.append(json.dumps({"navigation": nav, "localization": loc}))
        logger.info("Holo3 step: %s -> %s", log, pyautogui_code)
        return self.responses[-1], pyautogui_code

    # --------------------------------------------------------------- two passes
    def _navigate(self, instruction: str, b64: str) -> Optional[dict]:
        messages = build_navigation_messages(
            task=instruction, image_path="__CURRENT__", step=len(self.screenshots))
        messages = self._inline_image(messages, b64)
        self._inject_history(messages)
        resp = self.call_llm(messages, self.max_tokens,
                             json_schema_response_format(NavigationStep, "navigation_step"))
        return self._parse_json(resp)

    def _localize(self, element: str, b64: str) -> Optional[Tuple[float, float]]:
        messages = build_localization_messages(instruction=element, image_path="__CURRENT__")
        messages = self._inline_image(messages, b64)
        obj = self._parse_json(self.call_llm(
            messages, 256, json_schema_response_format(ClickAction, "click")))
        if obj and "x" in obj and "y" in obj:
            try:
                return float(obj["x"]), float(obj["y"])
            except (TypeError, ValueError):
                return None
        return None

    # ---------------------------------------------------------------- translate
    def _translate(self, action: dict, b64: str, width: int,
                   height: int) -> Tuple[List[str], str, Optional[dict]]:
        """Holo action dict -> (pyautogui code list, log string, localization dict)."""
        kind = action.get("action")
        loc: Optional[dict] = None

        def to_px(xm, ym) -> Tuple[int, int]:
            x = round(float(xm) / self.coord_divisor * width)
            y = round(float(ym) / self.coord_divisor * height)
            return max(0, min(width - 1, x)), max(0, min(height - 1, y))

        def grounded_xy(element: str) -> Optional[Tuple[int, int]]:
            nonlocal loc
            g = self._localize(element, b64)                # localizer is authoritative
            if g is None:
                if "x" in action and "y" in action:         # fall back to nav coords
                    return to_px(action["x"], action["y"])
                return None
            loc = {"x": g[0], "y": g[1]}
            return to_px(g[0], g[1])

        if kind == "click_element":
            xy = grounded_xy(action.get("element", ""))
            if xy is None:
                return [], "click_element (localization failed)", loc
            fn = _CLICK_FN.get(action.get("click_type", "left"), "click")
            return [f"pyautogui.{fn}({xy[0]}, {xy[1]})"], f"{fn} @ {xy}", loc

        if kind == "drag":
            xy = grounded_xy(action.get("element", ""))
            if xy is None:
                return [], "drag (localization failed)", loc
            button = action.get("button", "left")
            return [f"pyautogui.dragTo({xy[0]}, {xy[1]}, button={button!r}, duration=0.5)"], \
                f"drag @ {xy}", loc

        if kind == "write":
            content = action.get("content", "")
            return [f"pyautogui.write({content!r}, interval=0.02)"], f"write {content!r}", loc

        if kind == "key":
            keys = action.get("keys", [])
            keys = [keys] if isinstance(keys, str) else list(keys)
            if action.get("hold"):
                inner = ", ".join(repr(k) for k in keys)
                return [f"pyautogui.hotkey({inner})"], f"hotkey {keys}", loc
            if len(keys) == 1:
                presses = int(action.get("presses", 1) or 1)
                code = f"pyautogui.press({keys[0]!r}" + (f", presses={presses})" if presses > 1 else ")")
                return [code], f"press {keys[0]} x{presses}", loc
            return [f"pyautogui.press({keys!r})"], f"press {keys}", loc

        if kind == "scroll":
            direction = action.get("direction", "down")
            mag = self.scroll_clicks
            if direction in ("up", "down"):
                amt = mag if direction == "up" else -mag
                if self.invert_scroll:
                    amt = -amt
                return [f"pyautogui.scroll({amt})"], f"scroll {direction}", loc
            amt = mag if direction == "right" else -mag
            return [f"pyautogui.hscroll({amt})"], f"hscroll {direction}", loc

        if kind == "wait":
            return ["WAIT"], "wait", loc

        if kind == "answer":
            status = action.get("status", "success")
            return (["DONE"] if status == "success" else ["FAIL"]), f"answer {status}", loc

        logger.error("Unknown Holo3 action: %r", action)
        return [], f"unknown {kind}", loc

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _inline_image(messages: List[dict], b64: str) -> List[dict]:
        """Swap holo3_format's path-based image items for OpenAI data-URL parts."""
        data_url = f"data:image/png;base64,{b64}"
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                new = []
                for item in content:
                    if isinstance(item, dict) and (item.get("type") == "image" or "image" in item):
                        new.append({"type": "image_url", "image_url": {"url": data_url}})
                    else:
                        new.append(item)
                out.append({"role": m["role"], "content": new})
            else:
                out.append(m)
        return out

    def _inject_history(self, messages: List[dict]) -> None:
        """Prepend a compact previous-actions log to the navigation user turn.

        Holo's navigation loop conditions on a running action memory; we keep the
        current screenshot as the only image and summarize prior steps as text.
        """
        if not self.action_log or self.history_length <= 0:
            return
        recent = self.action_log[-self.history_length:]
        past_notes = [n for n in self.notes[-self.history_length:] if n]
        log_text = "<previous_actions>\n" + "\n".join(
            f"{i + 1}. {a}" for i, a in enumerate(recent)) + "\n</previous_actions>\n"
        if past_notes:
            log_text += "<notes>\n" + "\n".join(past_notes) + "\n</notes>\n"
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                m["content"].insert(0, {"type": "text", "text": log_text})
                break

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Lenient JSON extraction: whole string, fenced block, or first {...}."""
        if not text or not text.strip():
            return None
        for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)):
            try:
                return json.loads(candidate.strip())
            except Exception:  # noqa: BLE001
                continue
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
        return None

    def call_llm(self, messages: List[dict], max_tokens: int,
                 response_format: Optional[dict] = None) -> str:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                      temperature=self.temperature, top_p=self.top_p)
        if response_format is not None:
            kwargs["response_format"] = response_format
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                logger.error("[Holo3] call failed (attempt %d/%d): %s", attempt, MAX_RETRY_TIMES, e)
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(5)
        return ""

    # ------------------------------------------------------------------- reset
    def reset(self, _logger=None, vm_ip=None, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.holo3_agent")
        self.vm_ip = vm_ip
        self._reset_memory()
