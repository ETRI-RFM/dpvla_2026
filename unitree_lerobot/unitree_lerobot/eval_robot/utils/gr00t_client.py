"""Minimal ZMQ client for the GR00T policy server.

No dependency on the gr00t package. Only requires pyzmq + msgpack-numpy.
Mirrors the wire protocol used by gr00t.policy.server_client.PolicyClient.
"""
from __future__ import annotations

from typing import Any

import msgpack_numpy as mnp
import numpy as np
import zmq


class Gr00tClient:
    """ZMQ REQ client for the GR00T inference server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 60000,
    ):
        self.ctx = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self) -> None:
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        try:
            self.sock.send(mnp.packb(req))
            resp = mnp.unpackb(self.sock.recv(), raw=False)
        except zmq.error.Again:
            self.sock.close()
            self._init_socket()
            raise
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> dict:
        return self._call("ping")

    def reset(self) -> Any:
        return self._call("reset", {"options": None})

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        resp = self._call("get_action", {"observation": observation, "options": None})
        return resp[0], resp[1]

    def close(self) -> None:
        try:
            self.sock.close()
            self.ctx.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Observation / action helpers
#
# The trained checkpoint expects (see conf.yaml of the checkpoint):
#   state    : dict[left_arm(7), right_arm(7), left_hand(6), right_hand(6)]
#              each as float32 (B=1, T=1, D)
#   video    : dict[cam_right_high, cam_left_high, cam_left_wrist, cam_right_wrist]
#              each as uint8 (B=1, T=1, H, W, 3)
#   language : {"annotation.human.task_description": [[task_text]]}
#
# State layout assumed by callers:
#   [arm_q(14) | left_hand(6) | right_hand(6)]  -> total 26 dims
# ---------------------------------------------------------------------------


def _pack_video(img, bgr_to_rgb: bool = False) -> np.ndarray:
    """Pack an HWC image into (B=1, T=1, H, W, 3) uint8.

    image_client decodes JPEGs via cv2.imdecode, which yields BGR.
    Whether the model expects RGB or BGR depends on how the training data
    was prepared (see /tmp/groot_color_check for a visual reference).
    Set `bgr_to_rgb=True` to swap channels before sending to the server.
    """
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if bgr_to_rgb:
        arr = arr[..., ::-1]
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return arr[None, None, ...]


def build_gr00t_observation(
    lerobot_obs: dict,
    state_26d: np.ndarray,
    task_text: str,
    bgr_to_rgb: bool = False,
) -> dict:
    """Build a gr00t-format observation from the unitree pipeline outputs."""
    s = state_26d.astype(np.float32, copy=False)
    return {
        "video": {
            "cam_right_high":  _pack_video(lerobot_obs["observation.images.cam_right_high"], bgr_to_rgb),
            "cam_left_high":   _pack_video(lerobot_obs["observation.images.cam_left_high"],  bgr_to_rgb),
            "cam_left_wrist":  _pack_video(lerobot_obs["observation.images.cam_left_wrist"], bgr_to_rgb),
            "cam_right_wrist": _pack_video(lerobot_obs["observation.images.cam_right_wrist"], bgr_to_rgb),
        },
        "state": {
            "left_arm":   s[None, None, 0:7],
            "right_arm":  s[None, None, 7:14],
            "left_hand":  s[None, None, 14:20],
            "right_hand": s[None, None, 20:26],
        },
        "language": {
            "annotation.human.task_description": [[task_text]],
        },
    }


def parse_gr00t_action_chunk(action_dict: dict) -> dict:
    """Unbatch the (B=1, T, D) action chunks coming back from the server."""
    return {
        "left_arm":   np.asarray(action_dict["left_arm"])[0],
        "right_arm":  np.asarray(action_dict["right_arm"])[0],
        "left_hand":  np.asarray(action_dict["left_hand"])[0],
        "right_hand": np.asarray(action_dict["right_hand"])[0],
    }
