"""Spatial Refinement and Multi-Turn Coordination Engine.

Orchestrates System 1 (reflexive micro-actor) and System 2 (cognitive macro-planner)
with multi-turn closed-loop visual error correction, dynamic 5% frame crosshair
overlays, and robust coordinate parsing.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

# Configure module-level logger
logger = logging.getLogger("CoordinationHarness")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Regex targeting '(x, y)' coordinate pair
COORD_REGEX = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


class VLMClientProtocol(Protocol):
    """Protocol for cognitive macro-planner VLM clients."""

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Query VLM with text prompt and visual snapshot."""
        ...


def extract_coordinates(vlm_output: str) -> Optional[Tuple[int, int]]:
    """Extract the final coordinate pair written on the last line of VLM output.

    Specifically inspects the last non-empty line of the VLM response for
    a coordinate pattern in the format '(x, y)'. If the last line contains
    multiple coordinates, the final match is extracted. If no coordinates
    appear on the last line, searches backwards through preceding lines.

    Args:
        vlm_output: Raw textual response returned by the Planner VLM.

    Returns:
        Tuple of (x, y) integers if matched, None otherwise.
    """
    if not vlm_output or not vlm_output.strip():
        logger.warning("extract_coordinates received empty or whitespace VLM output.")
        return None

    lines = [line.strip() for line in vlm_output.strip().splitlines() if line.strip()]
    if not lines:
        return None

    # Priority 1: Check the final line explicitly
    last_line = lines[-1]
    matches = COORD_REGEX.findall(last_line)
    if matches:
        last_match = matches[-1]
        coords = (int(last_match[0]), int(last_match[1]))
        logger.debug("Extracted coordinates %s from last line: '%s'", coords, last_line)
        return coords

    # Priority 2: Fallback scan backwards through preceding lines
    for line in reversed(lines[:-1]):
        matches = COORD_REGEX.findall(line)
        if matches:
            last_match = matches[-1]
            coords = (int(last_match[0]), int(last_match[1]))
            logger.warning(
                "Coordinates %s found on fallback line: '%s' (not on final line)",
                coords,
                line,
            )
            return coords

    logger.warning("No valid coordinate pattern '(x, y)' found in VLM output.")
    return None


def render_visual_feedback(
    image_path: str,
    prev_coords: Tuple[int, int],
    output_path: Optional[str] = None,
    alpha: float = 0.65,
) -> str:
    """Overlay a dynamic semi-transparent red crosshair spanning 5% of frame dimensions.

    Visual feedback mechanism invoked when an action fails to trigger an expected
    interface state transition. Overlays a semi-transparent red crosshair (#FF0000)
    centered directly at failed coordinates.

    Args:
        image_path: Path to clean source screenshot.
        prev_coords: Coordinate tuple (x, y) of the failed interaction attempt.
        output_path: Destination path for annotated image. If None, appends
                     '_feedback.png' to source filename.
        alpha: Transparency weighting for the red crosshair overlay.

    Returns:
        Path to the saved annotated image.

    Raises:
        FileNotFoundError: If image_path cannot be found or read.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Source image not found for visual feedback: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to decode image at path: {image_path}")

    h, w = img.shape[:2]
    cx, cy = int(prev_coords[0]), int(prev_coords[1])

    # Clamp center coordinates within image bounds
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    # Calculate dynamic cross-hair spanning exactly 5% of the frame dimensions
    span_x = max(4, int(round(w * 0.05)))
    span_y = max(4, int(round(h * 0.05)))

    half_span_x = span_x // 2
    half_span_y = span_y // 2

    # Crosshair endpoints
    x_start = max(0, cx - half_span_x)
    x_end = min(w - 1, cx + half_span_x)
    y_start = max(0, cy - half_span_y)
    y_end = min(h - 1, cy + half_span_y)

    # Prepare semi-transparent overlay layer
    overlay = img.copy()

    # Red color in BGR: #FF0000 -> (0, 0, 255)
    red_bgr = (0, 0, 255)
    line_thickness = max(2, int(round(min(w, h) * 0.003)))

    # Draw horizontal line
    cv2.line(overlay, (x_start, cy), (x_end, cy), red_bgr, line_thickness, cv2.LINE_AA)
    # Draw vertical line
    cv2.line(overlay, (cx, y_start), (cx, y_end), red_bgr, line_thickness, cv2.LINE_AA)
    # Draw subtle center circle for targeting precision
    circle_radius = max(3, line_thickness * 2)
    cv2.circle(overlay, (cx, cy), circle_radius, red_bgr, line_thickness, cv2.LINE_AA)

    # Alpha blend overlay with base frame
    beta = 1.0 - alpha
    blended = cv2.addWeighted(overlay, alpha, img, beta, 0)

    # Determine destination filepath
    if output_path is None:
        root, ext = os.path.splitext(image_path)
        output_path = f"{root}_feedback{ext if ext else '.png'}"

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(output_path, blended)
    if not success:
        raise IOError(f"Failed to write visual feedback image to {output_path}")

    logger.info(
        "Visual feedback crosshair rendered at (%d, %d) [Span: %dx%d px] -> %s",
        cx,
        cy,
        span_x,
        span_y,
        output_path,
    )
    return output_path


class CoordinationHarness:
    """Orchestrates System 1 micro-actor execution and System 2 cognitive planning."""

    def __init__(
        self,
        vlm_client: Any,
        work_dir: str = "./agent_workspace",
    ) -> None:
        """Initialize coordination harness.

        Args:
            vlm_client: Client capable of querying VLM model.
            work_dir: Storage directory for intermediate visual feedback snapshots.
        """
        self.vlm_client = vlm_client
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.turn_history: List[Dict[str, Any]] = []

    def format_planning_prompt(
        self, base_prompt: str, prev_coords: Optional[Tuple[int, int]] = None
    ) -> str:
        """Compose planner prompt with spatial feedback instructions if previous attempt failed.

        Args:
            base_prompt: High-level task instruction.
            prev_coords: Previous attempted coordinates, if any.

        Returns:
            Formatted prompt string.
        """
        prompt = base_prompt.strip()
        if prev_coords is not None:
            spatial_suffix = (
                f"\n\nLast attempt: ({prev_coords[0]}, {prev_coords[1]}). "
                "If incorrect, use the red cross to adjust."
            )
            prompt += spatial_suffix

        # Append strict output format instruction
        prompt += "\nRespond with your reasoning, and end with the exact target coordinates on the final line in format: (x, y)"
        return prompt

    def run_interaction_cycle(
        self,
        goal: str,
        initial_screenshot_path: str,
        execute_action_fn: Any,
        is_state_transition_fn: Any,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Execute multi-turn visual servoing cycle with automatic error refinement.

        1. Queries Planner VLM with current interface snapshot.
        2. Parses (x, y) target coordinates from final line.
        3. Executes micro-actor cursor trajectory.
        4. Verifies state transition using dirty-frame detection.
        5. If transition fails, renders 5% red crosshair feedback and loops.

        Args:
            goal: Task goal description.
            initial_screenshot_path: Path to baseline screenshot.
            execute_action_fn: Callback taking (x, y) to perform action.
            is_state_transition_fn: Callback checking if UI transitioned (returns bool).
            max_attempts: Max error-correction attempts.

        Returns:
            Summary dictionary with status, attempts count, and coordinate history.
        """
        current_image_path = initial_screenshot_path
        prev_coords: Optional[Tuple[int, int]] = None
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            logger.info("--- Starting Coordination Turn %d/%d ---", attempt, max_attempts)

            # 1. Format prompt with spatial error feedback if applicable
            prompt = self.format_planning_prompt(goal, prev_coords)
            logger.debug("Prompting VLM:\n%s", prompt)

            # 2. Query VLM macro-planner
            vlm_response = self.vlm_client.query(
                prompt=prompt, image_path=current_image_path
            )
            logger.info("VLM Response (Turn %d):\n%s", attempt, vlm_response.strip())

            # 3. Parse coordinates
            coords = extract_coordinates(vlm_response)
            if coords is None:
                logger.error("Failed to parse coordinates from VLM response on attempt %d.", attempt)
                return {
                    "success": False,
                    "reason": "Coordinate parsing failure",
                    "attempts": attempt,
                    "history": self.turn_history,
                }

            # 4. Execute micro-actor cursor action
            logger.info("System 1 executing action at coordinates: %s", coords)
            execute_action_fn(coords[0], coords[1])

            # Small settling pause for UI render
            time.sleep(0.1)

            # 5. Check state transition (reflexive sensory verification)
            transition_succeeded = is_state_transition_fn()

            self.turn_history.append(
                {
                    "turn": attempt,
                    "coords": coords,
                    "transition_succeeded": transition_succeeded,
                    "input_image": current_image_path,
                }
            )

            if transition_succeeded:
                logger.info("State transition verified! Goal successfully completed at %s.", coords)
                return {
                    "success": True,
                    "final_coords": coords,
                    "attempts": attempt,
                    "history": self.turn_history,
                }

            logger.warning(
                "Action at %s did NOT trigger expected state transition. Generating visual feedback...",
                coords,
            )

            # 6. Render visual feedback crosshair for next turn
            feedback_path = os.path.join(
                self.work_dir, f"visual_feedback_turn_{attempt}.png"
            )
            render_visual_feedback(
                image_path=initial_screenshot_path,
                prev_coords=coords,
                output_path=feedback_path,
            )

            # Update loop state for next turn
            current_image_path = feedback_path
            prev_coords = coords

        logger.error("Exceeded maximum coordination attempts (%d).", max_attempts)
        return {
            "success": False,
            "reason": "Max attempts exceeded without state transition",
            "attempts": attempt,
            "history": self.turn_history,
        }
