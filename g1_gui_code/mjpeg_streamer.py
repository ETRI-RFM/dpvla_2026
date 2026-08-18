"""MJPEG re-broadcaster for the G1 image_server stream.

Subscribes to ZeroMQ image_server on tcp://<host>:5555 (same protocol as
xr_teleoperate/teleop/image_server/image_client.py), strips the small
header, optionally selects a head view, then serves the latest frame as
multipart/x-mixed-replace MJPEG over HTTP so a browser <img> tag can
display the live preview.

Runs in the `tv` conda env (zmq + cv2 + numpy required).
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import zmq


_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_running = True


def _probe_tcp(host: str, port: int, timeout: float = 2.0) -> str | None:
    """Quick TCP reachability probe. Returns None on success, or a reason str."""
    import socket as _socket
    try:
        s = _socket.create_connection((host, port), timeout=timeout)
        s.close()
        return None
    except ConnectionRefusedError:
        return "connection refused (image_server is NOT listening on this port)"
    except _socket.timeout:
        return "timeout (host unreachable or firewalled)"
    except OSError as e:
        return f"OSError: {e}"


def receiver_loop(host: str, port: int, view_index: int | None,
                  jpeg_quality: int) -> None:
    """Receive ZMQ messages, decode, post-process, re-encode, publish."""
    global _latest_jpeg

    # Quick reachability check so the user gets a clear message instead of
    # silent waiting if image_server isn't actually running.
    reason = _probe_tcp(host, port)
    if reason is None:
        sys.stderr.write(f"[mjpeg] TCP {host}:{port} reachable ✓\n")
    else:
        sys.stderr.write(
            f"[mjpeg] WARN: TCP {host}:{port} {reason}\n"
            f"[mjpeg] WARN: make sure G1 Camera is running "
            f"(① Connect + Run image_server.py) — frames may not arrive.\n"
        )
    sys.stderr.flush()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.RCVTIMEO = 1500
    sys.stderr.write(
        f"[mjpeg] subscribed to tcp://{host}:{port}; waiting for frames...\n"
    )
    sys.stderr.flush()

    frames = 0
    total_frames = 0
    last_log = time.monotonic()
    last_warn = time.monotonic()
    started = time.monotonic()
    while _running:
        try:
            msg = sock.recv()
        except zmq.error.Again:
            now = time.monotonic()
            if total_frames == 0 and now - last_warn >= 5.0:
                sys.stderr.write(
                    f"[mjpeg] WARN: still no frames after "
                    f"{now - started:.0f}s — image_server may not be "
                    f"running on {host}:{port}\n"
                )
                sys.stderr.flush()
                last_warn = now
            continue
        except zmq.error.ZMQError as e:
            sys.stderr.write(f"[mjpeg] zmq error: {e}\n")
            time.sleep(0.2)
            continue

        # image_client.py uses two possible header formats:
        #   normal:    struct.pack('III', h, w, pad_b)               12 bytes
        #   unit_test: struct.pack('dIIII', ts, fid, h, w, pad_b)    24 bytes
        # Detect by looking for the JPEG SOI marker FF D8 FF.
        jpg_bytes = None
        head_h = head_w = pad_b = 0
        for hdr_size, fmt in ((12, "III"), (24, "dIIII")):
            if len(msg) <= hdr_size:
                continue
            if msg[hdr_size:hdr_size + 3] == b"\xff\xd8\xff":
                jpg_bytes = msg[hdr_size:]
                if fmt == "III":
                    head_h, head_w, pad_b = struct.unpack(fmt, msg[:hdr_size])
                else:
                    _ts, _fid, head_h, head_w, pad_b = struct.unpack(
                        fmt, msg[:hdr_size]
                    )
                break
        if jpg_bytes is None:
            continue

        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Split head / wrist (head is leftmost head_w columns; wrist is the
        # rest if present). Strip bottom padding from head.
        head_w = head_w or img.shape[1]
        head_h = head_h or img.shape[0]
        head = img[:, :head_w]
        if pad_b > 0:
            head = head[:head_h, :]
        wrist = img[:, head_w:] if img.shape[1] > head_w else None

        # If multiple head views are concatenated horizontally (binocular),
        # pick the requested one.
        if view_index is not None and head.shape[1] >= 1280:
            num_views = max(1, head.shape[1] // 640)
            idx = min(max(view_index, 0), num_views - 1)
            seg_w = head.shape[1] // num_views
            head = head[:, idx * seg_w:(idx + 1) * seg_w]

        # Compose: head + wrist side-by-side when wrist exists.
        if wrist is not None and wrist.size > 0:
            wh = head.shape[0]
            ww = max(1, int(wrist.shape[1] * wh / max(1, wrist.shape[0])))
            display = cv2.hconcat([head, cv2.resize(wrist, (ww, wh))])
        else:
            display = head

        ok, buf = cv2.imencode(
            ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not ok:
            continue
        with _lock:
            _latest_jpeg = buf.tobytes()

        frames += 1
        total_frames += 1
        if total_frames == 1:
            sys.stderr.write(
                f"[mjpeg] ✓ first frame received "
                f"({display.shape[1]}x{display.shape[0]}, "
                f"{len(_latest_jpeg)/1024:.1f} KB JPEG)\n"
            )
            sys.stderr.flush()
        now = time.monotonic()
        if now - last_log >= 5.0:
            sys.stderr.write(
                f"[mjpeg] {frames / (now - last_log):.1f} fps "
                f"({frames} frames in last {now - last_log:.1f}s)\n"
            )
            sys.stderr.flush()
            frames = 0
            last_log = now


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        return

    def do_GET(self):
        # Drop any ?query — our HTML polls /snapshot.jpg?t=<ts> for cache
        # busting, and exact-match path comparison would 404 on those.
        raw = self.path.split("?", 1)[0]
        if raw.startswith("/stream.mjpg"):
            return self._mjpeg()
        if raw == "/snapshot.jpg":
            return self._snapshot()
        if raw == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(404)

    def _snapshot(self):
        with _lock:
            frame = _latest_jpeg
        if frame is None:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def _mjpeg(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache, no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            last_sent_id = 0
            while _running:
                with _lock:
                    frame = _latest_jpeg
                    fid = id(frame)
                if frame is None or fid == last_sent_id:
                    time.sleep(0.02)
                    continue
                last_sent_id = fid
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                # Cap at ~30 fps to keep CPU/bandwidth in check.
                time.sleep(1.0 / 30.0)
        except Exception:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zmq-host", default="192.168.123.164")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--http-port", type=int, default=8766)
    parser.add_argument("--view-index", type=int, default=None,
                        help="If head image contains N concatenated views, "
                             "select which one to display (0-based).")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()

    threading.Thread(
        target=receiver_loop,
        args=(args.zmq_host, args.zmq_port, args.view_index, args.jpeg_quality),
        daemon=True,
    ).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), StreamHandler)
    server.daemon_threads = True
    sys.stderr.write(
        f"[mjpeg] HTTP server ready: "
        f"http://127.0.0.1:{args.http_port}/stream.mjpg\n"
    )
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[mjpeg] interrupted\n")


if __name__ == "__main__":
    main()
