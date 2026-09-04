"""Holo3 agent message format for the GIMP calibration demonstrations.

Mirrors H Company's official hai-cookbook (utils/navigation.py +
utils/localization.py) so the EASY-EP calibration inputs look exactly like what
Holo3 sees at inference -- the whole point of few-shot expert localization is
that the demonstrations match the served distribution.

Holo3 is driven as TWO passes over the same weights, with different prompts:

  * navigation -- system prompt (guidelines + an embedded <output_json_format>
    JSON schema) then <task>/<observation><screenshot>. The model returns
    {note, thought, action}; the action names a UI element by *description*.
  * localization -- a GUI-general "Localize an element ... output Click(x, y)"
    prompt. The model grounds an element description to pixel coordinates.

The cookbook's navigation prompt/action space is web-only (cookies, captchas,
go_back/goto/refresh). AgentNet GIMP is desktop, and Holo3's real *desktop*
action space is generated server-side and is not public, so the navigation
ActionSpace below is a desktop adaptation (click/write/key/scroll/drag/wait/
answer) that keeps Holo's exact {note, thought, action} structure and schema
embedding. The localization prompt is used verbatim -- it is already GUI-general.
"""
import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# --- Navigation: desktop-adapted from hai-cookbook utils/navigation.py ----------

NAV_SYSTEM_PROMPT: str = """Imagine you are a robot operating a desktop computer, just like a human. Now you need to complete a task.
In each iteration, you will receive an Observation that includes a screenshot of the current screen.
Carefully analyze the visual information to identify what to do, then follow the guidelines to choose the next action.
You should detail your thought (i.e. reasoning steps) before taking the action.
Also detail in the note field any information extracted from the screen that is relevant to solving the task.

Guidelines:
- Use the task and the current screenshot to decide the single next action.
- Refer to a target by its visible name, label, icon, or distinguishing features (shape, colour, position) in the element field, never by coordinates.
- To type into a field, make sure it is focused first (click it), then use the write action.
- Consolidate repeated key presses using the presses field rather than emitting many key actions.
- For window controls, identify them correctly: minimize is the '—', maximize the '□', close the 'X'.
- Only answer with status 'success' when the screenshot confirms the task is complete; use 'failure' when it cannot be completed.

# <output_json_format>
# ```json
# {output_format}
# ```
# </output_json_format>
"""


class ClickElementAction(BaseModel):
    """Click a desktop UI element identified by its description."""

    action: Literal["click_element"] = "click_element"
    click_type: Literal["left", "double", "right", "middle", "triple", "move"] = "left"
    element: str = Field(description="text description of the element")
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")


class WriteElementAction(BaseModel):
    """Type text into the currently focused field."""

    action: Literal["write"] = "write"
    content: str = Field(description="Text to type")


class KeyAction(BaseModel):
    """Press a key, a chord of keys, or a key repeated several times."""

    action: Literal["key"] = "key"
    keys: list[str] = Field(description="Key names, e.g. ['enter'] or ['ctrl', 'c']")
    hold: bool = Field(default=False, description="True = press all keys together (hotkey chord)")
    presses: int = Field(default=1, description="Number of times to press (for a single key)")


class ScrollAction(BaseModel):
    """Scroll in a direction."""

    action: Literal["scroll"] = "scroll"
    direction: Literal["up", "down", "left", "right"] = "down"


class DragAction(BaseModel):
    """Drag from the current cursor position to a desktop element."""

    action: Literal["drag"] = "drag"
    element: str = Field(description="text description of the drop target")
    x: int
    y: int
    button: Literal["left", "right", "middle"] = "left"


class WaitAction(BaseModel):
    """Wait briefly for the UI to update."""

    action: Literal["wait"] = "wait"


class AnswerAction(BaseModel):
    """Terminal action: the task is complete or cannot be completed."""

    action: Literal["answer"] = "answer"
    status: Literal["success", "failure"] = "success"
    content: str = Field(default="", description="Optional final message")


DesktopAction = Union[
    ClickElementAction, WriteElementAction, KeyAction, ScrollAction,
    DragAction, WaitAction, AnswerAction,
]


class NavigationStep(BaseModel):
    """Output of the desktop navigation agent."""

    note: str = Field(default="", description="Task-relevant info extracted from the screen; empty if none.")
    thought: str = Field(description="Reasoning about the next step (<4 lines).")
    action: DesktopAction = Field(discriminator="action")


def _image_item(image_path: str) -> dict[str, Any]:
    # Path-based item (not base64): collect_expert_stats.py opens these and
    # passes them to the Holo3 processor. The chat template renders any content
    # item carrying an 'image' key as a vision placeholder.
    return {"type": "image", "image": image_path}


def build_navigation_messages(task: str, image_path: str, step: int = 1) -> list[dict]:
    system_prompt = NAV_SYSTEM_PROMPT.format(
        output_format=json.dumps(NavigationStep.model_json_schema()))
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"<task>\n{task}\n</task>\n"},
            {"type": "text", "text": f"<observation step={step}>\n<screenshot>\n"},
            _image_item(image_path),
            {"type": "text", "text": "\n</screenshot>\n</observation>\n"},
        ]},
    ]


# --- Localization: verbatim from hai-cookbook utils/localization.py --------------

LOCALIZATION_TASK_PROMPT = """\
Localize an element on the GUI image according to my instructions
and output a click position as Click(x, y) with x num pixels from
the left edge and y num pixels from the top edge.
"""


class ClickAction(BaseModel):
    """Click at specific coordinates on the screen."""

    action: Literal["click"] = "click"
    x: int
    y: int


def build_localization_messages(instruction: str, image_path: str) -> list[dict]:
    return [
        {"role": "system", "content": [
            {"type": "text", "text": json.dumps([ClickAction.model_json_schema()])}]},
        {"role": "user", "content": [
            _image_item(image_path),
            {"type": "text", "text": f"{LOCALIZATION_TASK_PROMPT}\n{instruction}"},
        ]},
    ]
