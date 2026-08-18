"""Local V4L2 camera streamer + recorder.

Replaces guvcview for the in-browser recording flow:

  - reads /dev/video4 (Intel RealSense color stream, YUYV) via OpenCV,
  - serves the latest frame as JPEG at /snapshot.jpg (polled by <img>),
  - records to MP4 when /record/start is POST-ed, stops on /record/stop.

Runs in the `tv` conda env (cv2 + numpy).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False


# ---- Shared state ----
_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_latest_shape: tuple[int, int] | None = None        # (height, width)
_running = True

_rec_lock = threading.Lock()
_rec: dict | None = None                            # {writer, file, frames, started_at}


def _publish_frame(frame, jpeg_quality: int) -> None:
    """Encode a BGR frame as JPEG + push to preview state + write to recorder
    if active. Shared by both backends (pyrealsense2 + cv2 fallback)."""
    global _latest_jpeg, _latest_shape

    eok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if eok:
        with _lock:
            _latest_jpeg = buf.tobytes()
            _latest_shape = (frame.shape[0], frame.shape[1])

    with _rec_lock:
        r = _rec
        if r is not None and r.get("writer") is not None:
            try:
                if r["writer_kind"] == "ffmpeg":
                    r["writer"].stdin.write(frame.tobytes())
                else:
                    r["writer"].write(frame)
                r["frames"] += 1
            except (cv2.error, BrokenPipeError, OSError) as e:
                sys.stderr.write(f"[rs] writer error: {e}\n")


def _capture_loop_rs(width: int, height: int, fps: float,
                     jpeg_quality: int) -> None:
    """Capture via pyrealsense2 — librealsense2 handles auto WB/exposure
    correctly on Intel RealSense cameras (D435 etc.), so colors come out
    right with no manual tweaking."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        sys.stderr.write(f"[rs] pyrealsense2 start failed: {e}\n")
        sys.exit(1)
    dev = profile.get_device()
    sys.stderr.write(
        f"[rs] opened via pyrealsense2: "
        f"{dev.get_info(rs.camera_info.name)} "
        f"({dev.get_info(rs.camera_info.serial_number)})  "
        f"{width}x{height} @ {fps:.0f}fps  format=BGR8\n"
    )
    sys.stderr.flush()

    frames_logged = 0
    last_log = time.monotonic()
    try:
        while _running:
            try:
                frames = pipeline.wait_for_frames(2000)
            except RuntimeError as e:
                sys.stderr.write(f"[rs] wait_for_frames timeout: {e}\n")
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())  # already BGR
            _publish_frame(frame, jpeg_quality)

            frames_logged += 1
            now = time.monotonic()
            if now - last_log >= 5.0:
                sys.stderr.write(
                    f"[rs] {frames_logged / (now - last_log):.1f} fps "
                    f"({frames_logged} frames in {now - last_log:.1f}s)\n"
                )
                sys.stderr.flush()
                frames_logged = 0
                last_log = now
    finally:
        pipeline.stop()


def _capture_loop_cv2(device: str, jpeg_quality: int, width: int,
                      height: int, fps: float) -> None:
    """Fallback capture via cv2.VideoCapture for non-RealSense devices."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.stderr.write(f"[rs] ERR: cannot open {device}\n")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    sys.stderr.write(
        f"[rs] opened via cv2: {device} {actual_w}x{actual_h} "
        f"@ {actual_fps:.1f}fps (cv2 fallback)\n"
    )
    sys.stderr.flush()

    frames_logged = 0
    last_log = time.monotonic()
    while _running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        _publish_frame(frame, jpeg_quality)
        frames_logged += 1
        now = time.monotonic()
        if now - last_log >= 5.0:
            sys.stderr.write(
                f"[rs] {frames_logged / (now - last_log):.1f} fps "
                f"({frames_logged} frames in {now - last_log:.1f}s)\n"
            )
            sys.stderr.flush()
            frames_logged = 0
            last_log = now
    cap.release()


def capture_loop(device: str, jpeg_quality: int, width: int, height: int,
                 fps: float, prefer_rs: bool = True) -> None:
    """Choose the best backend: pyrealsense2 first (correct colors on Intel
    RealSense), cv2.VideoCapture fallback otherwise."""
    if prefer_rs and _HAS_RS:
        try:
            _capture_loop_rs(width, height, fps, jpeg_quality)
            return
        except SystemExit:
            raise
        except Exception as e:
            sys.stderr.write(
                f"[rs] pyrealsense2 failed ({e}); falling back to cv2\n"
            )
    _capture_loop_cv2(device, jpeg_quality, width, height, fps)


def start_recording(out_dir: Path, filename: str | None, fps: float) -> dict:
    """Open a new mp4 VideoWriter; uses the latest frame to pick a size."""
    global _rec
    with _rec_lock:
        if _rec is not None:
            return {"ok": False, "error": "already recording",
                    "file": _rec["file"]}
    with _lock:
        shape = _latest_shape
    if shape is None:
        return {"ok": False, "error": "no frame yet — preview not ready"}
    h, w = shape

    out_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"g1_{time.strftime('%y%m%d_%H%M%S')}.mp4"
    if not name.lower().endswith(".mp4"):
        name = name + ".mp4"
    out = (out_dir / name).resolve()

    # Prefer ffmpeg-via-pipe for H.264/AAC mp4 (plays natively in browsers).
    # cv2's bundled ffmpeg often lacks the libx264 encoder, so we shell out
    # to the system ffmpeg binary and feed raw BGR frames over stdin.
    ffmpeg_bin = shutil.which("ffmpeg")
    writer_kind = None
    writer_obj = None
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(int(round(fps))),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(out),
        ]
        try:
            writer_obj = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            writer_kind = "ffmpeg"
        except OSError as e:
            sys.stderr.write(f"[rs] ffmpeg spawn failed: {e}\n")
            writer_obj = None

    if writer_obj is None:
        # Fallback: cv2 mp4v (works for any local player but NOT browser).
        w_try = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        if not w_try.isOpened():
            return {"ok": False, "error": f"cannot open VideoWriter: {out}"}
        writer_obj = w_try
        writer_kind = "cv2"
        sys.stderr.write(
            "[rs] WARN: using cv2 mp4v fallback — browser <video> may not "
            "play this file directly.\n"
        )
        sys.stderr.flush()

    with _rec_lock:
        _rec = {
            "writer": writer_obj,
            "writer_kind": writer_kind,
            "file": str(out),
            "frames": 0,
            "started_at": time.time(),
            "fps": fps,
            "size": [w, h],
        }
    sys.stderr.write(
        f"[rs] ● recording started ({writer_kind}) → {out}\n"
    )
    sys.stderr.flush()
    return {"ok": True, "file": str(out), "size": [w, h], "fps": fps,
            "codec": writer_kind}


def stop_recording() -> dict:
    """Close the writer (if any) and return summary."""
    global _rec
    with _rec_lock:
        r = _rec
        _rec = None
    if r is None:
        return {"ok": False, "error": "not recording"}
    try:
        if r["writer_kind"] == "ffmpeg":
            try: r["writer"].stdin.close()
            except Exception: pass
            try: r["writer"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                r["writer"].terminate()
        else:
            r["writer"].release()
    except Exception:
        pass
    duration = time.time() - r["started_at"]
    sys.stderr.write(
        f"[rs] ■ recording stopped: {r['file']} "
        f"({r['frames']} frames, {duration:.1f}s)\n"
    )
    sys.stderr.flush()
    return {
        "ok": True,
        "file": r["file"],
        "frames": r["frames"],
        "duration_s": duration,
        "size": r["size"],
        "fps": r["fps"],
    }


def record_status() -> dict:
    with _rec_lock:
        r = _rec
    if r is None:
        return {"recording": False}
    return {
        "recording": True,
        "file": r["file"],
        "frames": r["frames"],
        "duration_s": time.time() - r["started_at"],
        "size": r["size"],
        "fps": r["fps"],
    }


class Handler(BaseHTTPRequestHandler):
    out_dir: Path = Path.home() / "Videos"
    rec_fps: float = 30.0

    def log_message(self, fmt, *args):  # quiet
        return

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        raw = self.path.split("?", 1)[0]
        if raw == "/snapshot.jpg":
            with _lock:
                frame = _latest_jpeg
            if frame is None:
                self._json({"ok": False, "error": "no frame yet"}, status=503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return
        if raw == "/record/status":
            return self._json(record_status())
        if raw == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(404)

    def do_POST(self):
        raw = self.path.split("?", 1)[0]
        body = self._read_json()
        if raw == "/record/start":
            # Frontend may override output directory + filename per recording.
            raw_dir = (body.get("out_dir") or "").strip()
            out_dir = (
                Path(raw_dir).expanduser().resolve()
                if raw_dir else self.out_dir
            )
            filename = (body.get("filename") or "").strip() or None
            r = start_recording(out_dir, filename, self.rec_fps)
            return self._json(r)
        if raw == "/record/stop":
            return self._json(stop_recording())
        self.send_error(404)

    def do_OPTIONS(self):
        # Permit cross-origin preflight (frontend at :8765 calls us at :8767).
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/dev/video4",
                   help="V4L2 device path (default: Intel RealSense color)")
    p.add_argument("--http-port", type=int, default=8767)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--out-dir", default=str(Path.home() / "Videos"))
    args = p.parse_args()

    Handler.out_dir = Path(args.out_dir).expanduser()
    Handler.rec_fps = args.fps

    threading.Thread(
        target=capture_loop,
        args=(args.device, args.jpeg_quality, args.width, args.height, args.fps),
        daemon=True,
    ).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler)
    server.daemon_threads = True
    sys.stderr.write(
        f"[rs] HTTP ready: http://127.0.0.1:{args.http_port}/  "
        f"(snapshot, record/start, record/stop, record/status)\n"
    )
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[rs] interrupted\n")
    finally:
        # On clean shutdown, close any active recording so file is playable.
        if _rec is not None:
            stop_recording()


if __name__ == "__main__":
    main()
