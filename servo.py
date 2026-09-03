"""Continuous Visual Servoing and Cursor Execution Engine.

Implements reflexive micro-actor trajectory tracing using PyAutoGUI sub-step
interpolations, discrete interaction primitives, and comprehensive keyboard
and mouse hardware safety protocols.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Sequence, Tuple

import pyautogui  # type: ignore

# Configure module-level logger
logger = logging.getLogger("ServoController")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# PyAutoGUI Safety Configuration
pyautogui.PAUSE = 0.001   # Minimize default inter-command pause latency

# Known modifier keys for recovery safeguards
MODIFIER_KEYS = {"shift", "ctrl", "alt", "win", "command", "option"}


def configure_failsafe(enable: bool = True) -> None:
    """Configure PyAutoGUI fail-safe with automatic background/headless session detection.

    If the desktop cursor is pinned to (0, 0) by the OS (common in background, service,
    or headless sessions), PyAutoGUI's FailSafe will falsely trigger on every call.
    This function automatically detects that state and avoids false aborts.

    Args:
        enable: Desired failsafe status when on an active user desktop.
    """
    try:
        cur_pos = pyautogui.position()
        if cur_pos[0] == 0 and cur_pos[1] == 0:
            logger.debug(
                "Cursor detected at (0, 0) (background/headless session). Setting FAILSAFE=False."
            )
            pyautogui.FAILSAFE = False
        else:
            pyautogui.FAILSAFE = enable
    except Exception:
        pyautogui.FAILSAFE = False


# Initial configuration
configure_failsafe(enable=True)


def get_screen_bounds() -> Tuple[int, int]:
    """Get current primary screen resolution (width, height).

    Returns:
        Tuple of (screen_width, screen_height).
    """
    try:
        size = pyautogui.size()
        return int(size[0]), int(size[1])
    except Exception as exc:
        logger.debug("Failed to query screen size from pyautogui: %s; using default 1920x1080", exc)
        return 1920, 1080


def clamp_coordinates(x: int, y: int) -> Tuple[int, int]:
    """Clamp target coordinates within physical display bounds.

    Args:
        x: Target horizontal coordinate.
        y: Target vertical coordinate.

    Returns:
        Clamped (x, y) tuple safely inside active screen space.
    """
    w, h = get_screen_bounds()
    clamped_x = max(1, min(int(x), w - 2))
    clamped_y = max(1, min(int(y), h - 2))
    return clamped_x, clamped_y


def generate_substep_trajectory(
    start: Tuple[int, int], end: Tuple[int, int], steps: int = 20
) -> List[Tuple[int, int]]:
    """Generate smooth sub-step coordinate trajectory between two points.

    Uses smooth S-curve easing (smoothstep interpolation) to model natural
    human/robotic cursor acceleration and deceleration.

    Args:
        start: Starting (x, y) coordinate.
        end: Ending (x, y) coordinate.
        steps: Total discrete sub-steps along the trajectory.

    Returns:
        Ordered list of coordinate tuples tracing the interpolated path.
    """
    steps = max(2, steps)
    path: List[Tuple[int, int]] = []
    x0, y0 = start
    x1, y1 = end

    for i in range(steps):
        t = i / float(steps - 1)
        # Smoothstep curve: 3*t^2 - 2*t^3
        smooth_t = t * t * (3.0 - 2.0 * t)
        curr_x = int(round(x0 + (x1 - x0) * smooth_t))
        curr_y = int(round(y0 + (y1 - y0) * smooth_t))
        path.append(clamp_coordinates(curr_x, curr_y))

    return path


def execute_continuous_trajectory(coords_list: List[Tuple[int, int]], drag: bool = False) -> None:
    """Execute continuous mouse trajectory with sub-step micro-movements.

    Smoothly positions the cursor to the initial anchor and traces each trajectory
    point using sub-step movements. If drag=True, holds the mouse button down.

    Args:
        coords_list: Non-empty ordered sequence of (x, y) coordinates.
        drag: If True, performs drag with mouse button down. Otherwise moves smoothly.

    Raises:
        ValueError: If coords_list is empty.
    """
    if not coords_list:
        raise ValueError("coords_list cannot be empty in execute_continuous_trajectory.")

    configure_failsafe(enable=True)
    sanitized_coords = [clamp_coordinates(pt[0], pt[1]) for pt in coords_list]
    start_x, start_y = sanitized_coords[0]

    logger.info(
        "Beginning continuous trajectory execution across %d points (Start: %s, End: %s, Drag: %s).",
        len(sanitized_coords),
        (start_x, start_y),
        sanitized_coords[-1],
        drag,
    )

    try:
        # Move smoothly to start anchor
        pyautogui.moveTo(start_x, start_y, duration=0.1)
        if drag:
            pyautogui.mouseDown(button="left")

        # Trace trajectory sub-steps
        for pt_x, pt_y in sanitized_coords[1:]:
            pyautogui.moveTo(pt_x, pt_y, duration=0.01, _pause=False)

    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafe triggered! Aborting trajectory execution.")
        raise
    except Exception as exc:
        logger.error("Exception during trajectory execution: %s", exc)
        raise
    finally:
        if drag:
            try:
                pyautogui.mouseUp(button="left")
                logger.debug("Mouse button released successfully.")
            except Exception as release_exc:
                logger.error("Failed to release mouse button in finally block: %s", release_exc)


def click_at(x: int, y: int, button: str = "left", smooth: bool = True) -> None:
    """Perform a discrete mouse click at specified coordinates.

    Args:
        x: Horizontal coordinate.
        y: Vertical coordinate.
        button: Mouse button ('left', 'right', 'middle').
        smooth: Whether to smoothly glide cursor prior to clicking.
    """
    configure_failsafe(enable=True)
    safe_x, safe_y = clamp_coordinates(x, y)
    logger.debug("Executing click_at (%d, %d) [button=%s]", safe_x, safe_y, button)

    try:
        if smooth:
            pyautogui.moveTo(safe_x, safe_y, duration=0.08)
        pyautogui.click(x=safe_x, y=safe_y, button=button)
    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafe triggered during click_at.")
        raise


def double_click_at(x: int, y: int) -> None:
    """Perform a double click at specified coordinates.

    Args:
        x: Horizontal coordinate.
        y: Vertical coordinate.
    """
    configure_failsafe(enable=True)
    safe_x, safe_y = clamp_coordinates(x, y)
    logger.debug("Executing double_click_at (%d, %d)", safe_x, safe_y)
    try:
        pyautogui.moveTo(safe_x, safe_y, duration=0.08)
        pyautogui.doubleClick(x=safe_x, y=safe_y)
    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafe triggered during double_click_at.")
        raise


def type_text(text: str, interval: float = 0.02) -> None:
    """Safely type a sequence of characters with timing control.

    Args:
        text: String content to type.
        interval: Delay in seconds between individual keystrokes.
    """
    if not text:
        return
    logger.debug("Typing %d characters (interval=%.3fs)", len(text), interval)
    try:
        pyautogui.write(text, interval=interval)
    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafe triggered during type_text.")
        raise


def hotkey(*keys: str) -> None:
    """Execute a keyboard shortcut sequence with modifier-release safety guarantees.

    Args:
        keys: Key names to press simultaneously (e.g., 'ctrl', 's' or 'alt', 'tab').
    """
    if not keys:
        return

    clean_keys = [k.strip().lower() for k in keys if k.strip()]
    logger.info("Triggering hotkey sequence: %s", "+".join(clean_keys))

    try:
        pyautogui.hotkey(*clean_keys)
    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafe triggered during hotkey.")
        raise
    except Exception as exc:
        logger.error("Exception during hotkey execution: %s", exc)
        raise
    finally:
        # Emergency safeguard: ensure modifier keys are never left held down
        for k in clean_keys:
            if k in MODIFIER_KEYS:
                try:
                    pyautogui.keyUp(k)
                except Exception:
                    pass
