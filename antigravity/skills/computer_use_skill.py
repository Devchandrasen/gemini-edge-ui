"""Custom Skill Module: ComputerUseSkill.

Integrates local low-latency components (DXcam, PyAutoGUI, and Local VLM Client)
to enable decoupled GUI interaction and OS automation.
"""

from __future__ import annotations

import base64
import ctypes
import os
import re
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui  # type: ignore
import requests

import ctypes

# Attach to interactive user desktop station on Windows before DXcam import
try:
    user32 = ctypes.windll.user32
    h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
    if h_desk:
        user32.SetThreadDesktop(h_desk)
except Exception:
    pass

try:
    import dxcam  # type: ignore
except ImportError:
    dxcam = None

# Safety Safeguard: Mouse pointer ko screen ke top-left (0,0) par le jaane se agent turant ruk jayega.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.2


class ComputerUseSkill:
    """Antigravity Custom Skill: Ek complete decoupled GUI interaction pipeline.

    Jo physical PC controls (mouse/keyboard) ko local VLM reasoning se jodta hai.
    """

    def __init__(
        self,
        local_vlm_url: str = "http://localhost:8000/v1",
        target_fps: int = 30,
    ) -> None:
        self.vlm_url = local_vlm_url
        self.target_fps = target_fps
        self.prev_frame: Optional[np.ndarray] = None
        self._attach_desktop()

        # Windows-native super-fast screen capture setup
        try:
            if dxcam is not None:
                self.camera = dxcam.create(backend="dxgi", processor_backend="numpy")
                print("[ComputerUse] DXcam backend successfully initialized.")
            else:
                print("[ComputerUse] DXcam not installed, using fallback.")
                self.camera = None
        except Exception as e:
            print(f"[ComputerUse] DXcam init failed, fallback is required: {e}")
            self.camera = None

    def _attach_desktop(self) -> None:
        """Attach thread to active user desktop station on Windows."""
        try:
            user32 = ctypes.windll.user32
            h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
            if h_desk:
                user32.SetThreadDesktop(h_desk)
        except Exception:
            pass

    def capture_screenshot(
        self, output_path: str = "temp_screen.png"
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Ultra-low-latency screenshot capture."""
        frame: Optional[np.ndarray] = None
        if self.camera:
            try:
                frame = self.camera.grab()
            except Exception:
                frame = None

            if frame is None and self.prev_frame is not None:
                frame = self.prev_frame
            elif frame is not None:
                self.prev_frame = frame

        if frame is None:
            # Multi-platform fallback (MSS or PIL ImageGrab)
            try:
                import mss
                with mss.mss() as sct:
                    shot = sct.grab(sct.monitors[1])
                    frame = np.array(shot)[:, :, :3]
            except Exception:
                try:
                    from PIL import ImageGrab
                    screenshot = ImageGrab.grab()
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    frame = None
            if frame is not None:
                self.prev_frame = frame

        if frame is not None:
            cv2.imwrite(output_path, frame)
            return output_path, frame
        return None, None

    def is_screen_dirty(
        self, current_frame: Optional[np.ndarray], threshold: float = 0.01
    ) -> bool:
        """Temporal filter: Screen par actual visual change hone par hi VLM ko wake-up karein."""
        if self.prev_frame is None or current_frame is None:
            return True
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_prev, gray_curr)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_ratio = float(np.sum(thresh == 255)) / float(thresh.size)
        return changed_ratio > threshold

    def draw_failed_crosshair(
        self, image_path: str, failed_coords: Tuple[int, int]
    ) -> str:
        """Visual feedback rendering (See, Point, Refine model)."""
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        x, y = failed_coords

        # 5% marker lines draw karein (red color #FF0000 -> BGR (0, 0, 255))
        span_x, span_y = int(w * 0.05), int(h * 0.05)
        cv2.line(img, (x - span_x // 2, y), (x + span_x // 2, y), (0, 0, 255), 2)
        cv2.line(img, (x, y - span_y // 2), (x, y + span_y // 2), (0, 0, 255), 2)

        feedback_path = "feedback_screen.png"
        cv2.imwrite(feedback_path, img)
        return feedback_path

    def query_local_vlm(self, prompt: str, image_path: str) -> Optional[str]:
        """Local OpenAI-compatible API endpoint (vLLM/Ollama) se coordinate prediction fetch karna."""
        # Note: Local endpoint calls ke liye real visual inputs ko base64 format mein pack kiya jata hai
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen2.5-vl:7b",  # local serve kiya gaya model tag
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                f"{self.vlm_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"[VLM HTTP {response.status_code}]: {response.text}")
        except Exception as e:
            print(f"[VLM Connection Error] local API is offline: {e}")
        return None

    def parse_coordinate_response(
        self, text_output: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        """Regular Expression se final line par predicted (x,y) coordinates extract karna."""
        if not text_output:
            return None
        last_line = text_output.strip().split("\n")[-1]
        match = re.search(r"\((\d+),\s*(\d+)\)", last_line)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def execute_skill(self, objective: str) -> None:
        """Skill Execution loop jo objective complete hone tak physical interactions handle karega."""
        print(f"\n[Antigravity GUI-Agent] Task Initialized: '{objective}'")
        print("ALERT: Apne physical mouse ko TOP-LEFT corner par le jaane se execution turant cancel ho jayegi.\n")

        step = 1
        previous_attempt: Optional[Tuple[int, int]] = None

        while step <= 10:  # Safety boundary of max 10 execution cycles per task
            print(f"--- [Step {step}] ---")
            screenshot_file = f"step_{step}_clean.png"
            path, raw_frame = self.capture_screenshot(screenshot_file)

            if path is None:
                print("Error: Screen ingestion pipe crashed.")
                break

            # Step-by-Step Prompt structure
            prompt = (
                f"You are a local OS Controller executing: '{objective}'.\n"
                "Analyze the screenshot carefully and determine the next logical action (e.g., click a button, open a menu).\n"
                "End your response with precise coordinate pairs on the last line inside parenthesis, e.g.: (450,120)"
            )

            if previous_attempt:
                # Prior attempt target mark karke corrected image send karein
                path = self.draw_failed_crosshair(screenshot_file, previous_attempt)
                prompt += (
                    f"\nNote: Your previous action at {previous_attempt} missed. "
                    "Identify the red crosshair, calculate spatial correction, and output updated coordinates."
                )

            # VLM se raw reasoning output fetch karein
            raw_response = self.query_local_vlm(prompt, path)
            print(f"[System 2 Reasoning]:\n{raw_response}")

            coords = self.parse_coordinate_response(raw_response)

            if coords:
                x, y = coords
                # System 1 Visual Servoing: Cursor move aur physical click execute karein
                pyautogui.moveTo(x, y, duration=0.4)
                pyautogui.click()
                print(f"[System 1 Execution]: Clicked coordinates ({x}, {y})")

                # State feedback validation wait
                time.sleep(1.5)

                # Check target verification loop
                # Agar state change nahi hui, to next loop mein previous coordinate update trigger hoga
                previous_attempt = (x, y)
            else:
                print("[Parsing Error] Could not determine targets. Re-evaluating context...")
                time.sleep(1.0)

            step += 1

        print("[Antigravity GUI-Agent] Task evaluation lifecycle finished.")
