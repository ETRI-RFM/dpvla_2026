"""G1 Inference web server (stdlib HTTP + psutil for orphan cleanup)."""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================
# Configuration
# ============================================================
ROBOT_HOST = "unitree@192.168.123.164"
ROBOT_PASSWORD = "123"
ROBOT_MENU = "1"
PEXPECT_PYTHON = "/usr/bin/python3"

EVAL_PROTOCOL_JSON = "/home/goodman/g1_eval_protocol/evaluation_protocol_config.json"
LANG_INSTRUCTION_TXT = "/home/goodman/g1_eval_protocol/language_instruction.txt"
LANGUAGE_LIST_HTML = "/home/goodman/Downloads/actual_short_long_language_list_by_task.html"

GR00T_VENV_PY = "/mnt/ssd/Isaac-GR00T/.venv/bin/python"
GR00T_DEFAULT_MODEL = (
    "/mnt/ssd/GROOT/outputs/train_g1_groot/"
    "20260513_groot_finetune_g1_n1.7_longInst_3epoch/checkpoint-180000"
)
GR00T_HOST = "0.0.0.0"
GR00T_PORT = 5555

DP_VLA_DEFAULT_MODEL = (
    "/home/goodman/unitree_v030/act_validation/250616/"
    "dpact_dinov3_variance_config_final_260518/checkpoints/0160000/pretrained_model"
)
DP_VLA_FREQUENCY = 30

CLIENT_CONDA_ENV = "unitree_lerobot_v03"
CLIENT_WORKDIR = "/home/goodman/unitree_v030/unitree_IL_lerobot"
TELEOP_IMAGE_CLIENT_DIR = "~/projects/xr_teleoperate/teleop/image_server"
TELEOP_CONDA_ENV = "tv"

DEFAULT_ARM = "G1_29"
DEFAULT_EE = "brainco"
DEFAULT_ACTION_STEPS = 16

INIT_POSE_MIN = 0
INIT_POSE_MAX = 33422

REC_DEFAULT_DEVICE = "/dev/video4"

HTTP_PORT = 8765
MJPEG_PORT = 8766
RECORDER_PORT = 8767
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
MJPEG_STREAMER = HERE / "mjpeg_streamer.py"
REALSENSE_STREAMER = HERE / "realsense_streamer.py"
TV_PYTHON = "/home/goodman/miniforge3/envs/tv/bin/python"
RECORDER_OUT_DIR = str(Path.home() / "Videos")

# Version stamp injected into index.html as ?v=… on static asset URLs so
# browsers always pick up the latest css/js after a server restart, even if
# they aggressively cache.
SERVER_START_EPOCH = int(__import__("time").time())

# Per-launch PID journal — used to find/kill orphans on restart.
PID_STATE = Path.home() / ".cache" / "g1_inference_ui" / "pids.txt"

CONDA_INIT = (
    "{ source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true; }"
)


# ============================================================
# pexpect helper scripts
# ============================================================
def pexpect_ssh_script(initial_cmd: str = "") -> str:
    """Return a python script that: ssh + password + menu + (optional initial
    remote command), then relays stdin lines into the ssh pty.

    Special stdin tokens:
      "__INTR__"  → send Ctrl+C (0x03) through the pty
      "__EXIT__"  → close ssh and exit
    All other stdin lines are forwarded verbatim (with a trailing newline).
    Remote stdout/stderr is teed to local stdout."""
    return (
        "import os, sys, time, threading, pexpect\n"
        f"host = {ROBOT_HOST!r}\n"
        f"password = {ROBOT_PASSWORD!r}\n"
        f"menu = {ROBOT_MENU!r}\n"
        f"initial_cmd = {initial_cmd!r}\n"
        "spawn_cmd = ('ssh -tt -o StrictHostKeyChecking=no '\n"
        "             '-o UserKnownHostsFile=/dev/null '\n"
        "             '-o LogLevel=ERROR ' + host)\n"
        "c = pexpect.spawn(spawn_cmd, encoding='utf-8', timeout=30)\n"
        "c.logfile_read = sys.stdout\n"
        "try:\n"
        "    c.expect(['password:', 'Password:'])\n"
        "    c.sendline(password)\n"
        "    time.sleep(2)\n"
        "    c.sendline(menu)\n"
        "    time.sleep(2)\n"
        "    # Quiet the remote: disable color codes and OSC 8 hyperlinks\n"
        "    # from rich-based loggers so the web console isn't littered\n"
        "    # with ANSI escapes.\n"
        "    c.sendline('export NO_COLOR=1 TERM=dumb COLUMNS=200')\n"
        "    time.sleep(0.3)\n"
        "except Exception as e:\n"
        "    sys.stderr.write('login failed: ' + str(e) + chr(10))\n"
        "    sys.exit(1)\n"
        "\n"
        "if initial_cmd:\n"
        "    c.sendline(initial_cmd)\n"
        "\n"
        "def relay():\n"
        "    while True:\n"
        "        try:\n"
        "            line = sys.stdin.readline()\n"
        "        except Exception:\n"
        "            return\n"
        "        if not line:\n"
        "            return\n"
        "        stripped = line.rstrip(chr(10))\n"
        "        if stripped == '__EXIT__':\n"
        "            try: c.close(force=True)\n"
        "            except Exception: pass\n"
        "            return\n"
        "        if stripped == '__INTR__':\n"
        "            c.send(chr(3))\n"
        "            continue\n"
        "        c.send(line)\n"
        "\n"
        "threading.Thread(target=relay, daemon=True).start()\n"
        "try:\n"
        "    c.expect(pexpect.EOF, timeout=None)\n"
        "except pexpect.EOF:\n"
        "    pass\n"
    )


# ============================================================
# Command builders
# ============================================================
def cmd_image_server_relay() -> str:
    # Connect = ssh + login + cd into the working directory (absolute path
    # so re-clicking Connect or Run later is idempotent regardless of where
    # the previous run was killed). Run button then just sends the binary.
    py = pexpect_ssh_script(initial_cmd="cd ~/image_server")
    return f"{PEXPECT_PYTHON} -c {shlex.quote(py)}"


def cmd_brainco_hand_relay() -> str:
    py = pexpect_ssh_script(initial_cmd="cd ~/brainco_hand_service/bin")
    return f"{PEXPECT_PYTHON} -c {shlex.quote(py)}"


def cmd_image_client_local() -> str:
    """MJPEG re-broadcaster — replaces the cv2-window image_client.py so the
    camera preview can be embedded into the web UI via <img>."""
    # Strip the user@ from ROBOT_HOST → bare IP for the ZMQ connect.
    zmq_host = ROBOT_HOST.split("@", 1)[-1]
    return (
        f"{shlex.quote(TV_PYTHON)} -u "
        f"{shlex.quote(str(MJPEG_STREAMER))} "
        f"--zmq-host {zmq_host} --zmq-port 5555 "
        f"--http-port {MJPEG_PORT}"
    )


def cmd_gr00t_server(model_path: str) -> str:
    return (
        f"{GR00T_VENV_PY} -m gr00t.eval.run_gr00t_server "
        f"--model-path {shlex.quote(model_path)} "
        "--embodiment-tag new_embodiment "
        f"--device cuda --host {GR00T_HOST} --port {GR00T_PORT}"
    )


def cmd_inference_gr00t(episode_idx: int) -> str:
    return (
        f"{CONDA_INIT} && "
        f"cd {shlex.quote(CLIENT_WORKDIR)} && "
        f"conda activate {CLIENT_CONDA_ENV} && "
        "python -m unitree_lerobot.eval_robot.eval_g1_groot "
        f"  --arm {DEFAULT_ARM} --ee {DEFAULT_EE} "
        f"  --server-host localhost --server-port {GR00T_PORT} "
        f"  --episode-idx {episode_idx} "
        f"  --action-steps-per-chunk {DEFAULT_ACTION_STEPS}"
    )


def cmd_recorder(device: str) -> str:
    """Local V4L2 streamer + recorder (replaces guvcview for in-browser flow)."""
    return (
        f"{shlex.quote(TV_PYTHON)} -u "
        f"{shlex.quote(str(REALSENSE_STREAMER))} "
        f"--device {shlex.quote(device)} "
        f"--http-port {RECORDER_PORT} "
        f"--out-dir {shlex.quote(RECORDER_OUT_DIR)}"
    )


def cmd_inference_dp(model_path: str, episode_idx: int) -> str:
    return (
        f"{CONDA_INIT} && "
        f"cd {shlex.quote(CLIENT_WORKDIR)} && "
        f"conda activate {CLIENT_CONDA_ENV} && "
        "python -m unitree_lerobot.eval_robot.eval_g1_dp_new "
        f"  --frequency {DP_VLA_FREQUENCY} --arm {DEFAULT_ARM} --ee {DEFAULT_EE} "
        f"  --system1_cfg_path {shlex.quote(model_path)} "
        f"  --episode_idx {episode_idx} --use_dataset_hand_pose"
    )


# ============================================================
# Process manager
# ============================================================
class ProcessManager:
    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.histories: dict[str, list[str]] = {}
        self.subscribers: dict[str, list[queue.Queue]] = {}
        # Multiplexed subscribers receive (label, line) for every service —
        # avoids burning Chrome's per-origin HTTP/1.1 6-connection budget
        # with one SSE per service.
        self.all_subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()

    # ----- log fan-out -----
    def _push(self, label: str, line: str) -> None:
        with self.lock:
            buf = self.histories.setdefault(label, [])
            buf.append(line)
            if len(buf) > 5000:
                del buf[: len(buf) - 2500]
            subs = list(self.subscribers.get(label, []))
            all_subs = list(self.all_subscribers)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass
        for q in all_subs:
            try:
                q.put_nowait((label, line))
            except queue.Full:
                pass

    def subscribe(self, label: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=5000)
        with self.lock:
            for line in self.histories.get(label, []):
                try:
                    q.put_nowait(line)
                except queue.Full:
                    break
            self.subscribers.setdefault(label, []).append(q)
        return q

    def unsubscribe(self, label: str, q: queue.Queue) -> None:
        with self.lock:
            try:
                self.subscribers.get(label, []).remove(q)
            except ValueError:
                pass

    def subscribe_all(self) -> queue.Queue:
        """One subscriber receives every (label, line) pair across all
        services. Keeps the connection count to 1 instead of N."""
        q: queue.Queue = queue.Queue(maxsize=20000)
        with self.lock:
            # Stream existing history first (chronological per-label is
            # close enough — they're already interleaved by arrival).
            for label, history in self.histories.items():
                for line in history:
                    try:
                        q.put_nowait((label, line))
                    except queue.Full:
                        break
            self.all_subscribers.append(q)
        return q

    def unsubscribe_all(self, q: queue.Queue) -> None:
        with self.lock:
            try:
                self.all_subscribers.remove(q)
            except ValueError:
                pass

    # ----- lifecycle -----
    def start(self, label: str, cmd: str, with_stdin: bool = True) -> dict:
        with self.lock:
            existing = self.procs.get(label)
            if existing and existing.poll() is None:
                return {"ok": False, "error": "already running",
                        "pid": existing.pid}
            self.histories[label] = []
        self._push(label, f"$ {cmd}\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdin=subprocess.PIPE if with_stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1, text=True,
                start_new_session=True,
                env=env,
            )
        except OSError as e:
            self._push(label, f"[launch failed: {e}]\n")
            return {"ok": False, "error": str(e)}
        with self.lock:
            self.procs[label] = proc
        _record_pid(proc.pid, label)
        threading.Thread(
            target=self._reader, args=(label, proc), daemon=True,
        ).start()
        return {"ok": True, "pid": proc.pid}

    def _reader(self, label: str, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._push(label, line)
        except (OSError, ValueError):
            pass
        rc = proc.wait()
        self._push(label, f"[exited with code {rc}]\n")

    def stop(self, label: str, mode: str = "kill") -> dict:
        with self.lock:
            proc = self.procs.get(label)
        if not proc or proc.poll() is not None:
            return {"ok": False, "error": "not running"}
        if mode == "intr":
            if proc.stdin is None:
                return {"ok": False, "error": "no stdin (use mode=kill)"}
            try:
                proc.stdin.write("__INTR__\n")
                proc.stdin.flush()
            except OSError as e:
                return {"ok": False, "error": str(e)}
            self._push(label, "[sent __INTR__ → Ctrl+C through pty]\n")
            return {"ok": True}
        # default: SIGINT to local process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except OSError:
            try:
                proc.send_signal(signal.SIGINT)
            except OSError:
                return {"ok": False, "error": "send failed"}
        self._push(label, "[sent SIGINT to local process group]\n")
        return {"ok": True}

    def send(self, label: str, text: str) -> dict:
        with self.lock:
            proc = self.procs.get(label)
        if not proc or proc.poll() is not None:
            return {"ok": False, "error": "not running"}
        if proc.stdin is None:
            return {"ok": False, "error": "no stdin"}
        if not text.endswith("\n"):
            text = text + "\n"
        try:
            proc.stdin.write(text)
            proc.stdin.flush()
        except OSError as e:
            return {"ok": False, "error": str(e)}
        # Echo what we sent
        self._push(label, f"[sent stdin: {text.rstrip()}]\n")
        return {"ok": True}

    def status(self) -> dict:
        with self.lock:
            out = {}
            for label, p in self.procs.items():
                out[label] = {
                    "alive": p.poll() is None,
                    "pid": p.pid,
                    "rc": p.returncode,
                }
            return out

    # Services that talk to the robot via pexpect-managed ssh — we want a
    # graceful remote Ctrl+C before yanking the local process.
    _SSH_RELAY_LABELS = ("image_server", "brainco_hand")

    def kill_all(self) -> None:
        with self.lock:
            procs = list(self.procs.items())

        # Phase 1: graceful remote shutdown for ssh-relay services.
        sent_intr = False
        for label, proc in procs:
            if proc.poll() is not None or proc.stdin is None:
                continue
            if label in self._SSH_RELAY_LABELS:
                try:
                    proc.stdin.write("__INTR__\n")
                    proc.stdin.flush()
                    sent_intr = True
                except OSError:
                    pass
        if sent_intr:
            time.sleep(0.8)

        # Phase 2: SIGTERM (or SIGINT) to every local process group.
        for label, proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except OSError:
                    try:
                        proc.terminate()
                    except OSError:
                        pass


def _record_pid(pid: int, label: str) -> None:
    try:
        PID_STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_STATE, "a") as f:
            f.write(f"{pid},{label}\n")
    except OSError:
        pass


def cleanup_previous_runs() -> list[tuple[int, str]]:
    """Read the PID journal and kill any orphans from a previous run."""
    if not PID_STATE.exists():
        return []
    killed: list[tuple[int, str]] = []
    try:
        lines = PID_STATE.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        parts = line.split(",", 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        label = parts[1]
        try:
            alive = psutil.pid_exists(pid) if psutil else True
        except Exception:
            alive = False
        if not alive:
            continue
        try:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            killed.append((pid, label))
        except (ProcessLookupError, OSError):
            pass
    try:
        PID_STATE.unlink()
    except OSError:
        pass
    return killed


# ============================================================
# Config parsing (eval protocol + language list HTML)
# ============================================================
class _LangListParser(HTMLParser):
    _TASK_RE = re.compile(r"Task\s+(\d+)")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.result: dict[tuple[str, int], list[str]] = {}
        self._section: str | None = None
        self._task: int | None = None
        self._in_h3 = False
        self._h3 = ""
        self._in_li = False
        self._li = ""

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag == "section" and "lang-block" in cls:
            self._section = "short" if "short" in cls else (
                "long" if "long" in cls else None)
        elif tag == "h3":
            self._in_h3, self._h3 = True, ""
        elif tag == "li":
            self._in_li, self._li = True, ""

    def handle_endtag(self, tag):
        if tag == "section":
            self._section = None
            self._task = None
        elif tag == "h3":
            self._in_h3 = False
            m = self._TASK_RE.search(self._h3)
            if m and self._section:
                self._task = int(m.group(1))
                self.result.setdefault((self._section, self._task), [])
        elif tag == "li":
            self._in_li = False
            if self._section and self._task:
                txt = self._li.strip()
                if txt:
                    self.result.setdefault(
                        (self._section, self._task), []).append(txt)

    def handle_data(self, data):
        if self._in_h3:
            self._h3 += data
        elif self._in_li:
            self._li += data


def parse_language_list(path: str | Path) -> dict[tuple[str, int], list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}
    parser = _LangListParser()
    parser.feed(p.read_text(encoding="utf-8"))
    return parser.result


# ============================================================
# Local-filesystem directory listing (for the model-path picker)
# ============================================================
def _fs_list(target: str | None) -> dict:
    """Return a JSON-serialisable view of `target` for the directory picker.

    Always returns sorted dirs (then files, capped at 500) plus the resolved
    absolute path and parent. Falls back to the user's home if `target` is
    empty / invalid / not a directory.
    """
    raw = (target or "").strip()
    if not raw:
        p = Path.home()
    else:
        try:
            p = Path(raw).expanduser()
        except (OSError, RuntimeError, ValueError):
            return {"ok": False, "error": f"invalid path: {raw}"}
    try:
        p = p.resolve()
    except (OSError, RuntimeError):
        return {"ok": False, "error": f"resolve failed: {raw}"}
    if not p.is_dir():
        return {"ok": False, "error": f"not a directory: {p}"}

    try:
        entries = list(p.iterdir())
    except PermissionError:
        return {"ok": False, "error": f"permission denied: {p}"}
    except OSError as e:
        return {"ok": False, "error": str(e)}

    dirs: list[dict] = []
    files: list[dict] = []
    for e in entries:
        try:
            is_dir = e.is_dir()
        except OSError:
            continue
        if e.name.startswith("."):
            continue  # skip dotfiles by default
        item = {"name": e.name, "path": str(e)}
        if is_dir:
            dirs.append(item)
        else:
            files.append(item)
    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda d: d["name"].lower())

    parent = p.parent
    return {
        "ok": True,
        "path": str(p),
        "parent": str(parent) if parent != p else None,
        "home": str(Path.home()),
        "dirs": dirs,
        "files": files[:500],
    }


# ============================================================
# App
# ============================================================
class App:
    def __init__(self) -> None:
        self.pm = ProcessManager()
        try:
            with open(EVAL_PROTOCOL_JSON) as f:
                self.eval_protocol = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.eval_protocol = {}
        self.lang_lists = parse_language_list(LANGUAGE_LIST_HTML)

    def config_dict(self) -> dict:
        ll: dict[str, dict[str, list[str]]] = {"short": {}, "long": {}}
        for (sec, t), lines in self.lang_lists.items():
            ll[sec][str(t)] = lines
        return {
            "robot_host": ROBOT_HOST,
            "gr00t_default_model": GR00T_DEFAULT_MODEL,
            "dp_vla_default_model": DP_VLA_DEFAULT_MODEL,
            "rec_default_device": REC_DEFAULT_DEVICE,
            "init_pose_min": INIT_POSE_MIN,
            "init_pose_max": INIT_POSE_MAX,
            "eval_protocol": self.eval_protocol,
            "lang_lists": ll,
            "lang_instruction_txt": LANG_INSTRUCTION_TXT,
            "mjpeg_port": MJPEG_PORT,
            "recorder_port": RECORDER_PORT,
            "recorder_out_dir": RECORDER_OUT_DIR,
        }

    def start_service(self, label: str, params: dict | None = None) -> dict:
        params = params or {}
        if label == "image_server":
            return self.pm.start(label, cmd_image_server_relay(), with_stdin=True)
        if label == "image_client":
            return self.pm.start(label, cmd_image_client_local(), with_stdin=False)
        if label == "brainco_hand":
            return self.pm.start(label, cmd_brainco_hand_relay(), with_stdin=True)
        if label == "gr00t_server":
            model_path = (params.get("model_path") or GR00T_DEFAULT_MODEL).strip()
            return self.pm.start(label, cmd_gr00t_server(model_path),
                                 with_stdin=False)
        if label == "recorder":
            device = (params.get("device") or REC_DEFAULT_DEVICE).strip()
            return self.pm.start(label, cmd_recorder(device),
                                 with_stdin=False)
        if label == "inference":
            model_type = params.get("model_type", "gr00t")
            try:
                episode_idx = int(params.get("episode_idx"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "episode_idx must be integer"}
            instruction = (params.get("instruction") or "").strip()
            # Write the instruction file the scripts read by default.
            try:
                Path(LANG_INSTRUCTION_TXT).parent.mkdir(parents=True, exist_ok=True)
                Path(LANG_INSTRUCTION_TXT).write_text(instruction + "\n")
            except OSError as e:
                return {"ok": False, "error": f"write instruction failed: {e}"}
            if model_type == "gr00t":
                cmd = cmd_inference_gr00t(episode_idx)
            else:
                model_path = (params.get("model_path") or DP_VLA_DEFAULT_MODEL).strip()
                if not model_path:
                    return {"ok": False, "error": "model_path required for DP-VLA"}
                cmd = cmd_inference_dp(model_path, episode_idx)
            return self.pm.start(label, cmd, with_stdin=True)
        return {"ok": False, "error": f"unknown service: {label}"}

    def list_recordings(self) -> dict:
        """List .mp4 files in the recorder output dir (newest first)."""
        out = Path(RECORDER_OUT_DIR).expanduser()
        if not out.is_dir():
            return {"ok": True, "dir": str(out), "items": []}
        items = []
        for p in out.glob("*.mp4"):
            try:
                st = p.stat()
            except OSError:
                continue
            items.append({
                "path": str(p.resolve()),
                "name": p.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"ok": True, "dir": str(out), "items": items}

    def launch_guvcview(self, device: str) -> dict:
        if shutil.which("guvcview") is None:
            return {"ok": False, "error": "guvcview not installed"}
        try:
            subprocess.Popen(
                ["guvcview", f"--device={device}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}


# ============================================================
# HTTP handler
# ============================================================
class Handler(BaseHTTPRequestHandler):
    app: App | None = None

    def log_message(self, format, *args):
        return  # quieter

    def handle_one_request(self):
        # Silence BrokenPipeError/ConnectionResetError tracebacks — they
        # happen normally when the browser aborts a request (e.g. fast
        # polling cancels previous in-flight fetch).
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def _static(self, name: str, content_type: str | None = None) -> None:
        p = (STATIC_DIR / name).resolve()
        try:
            p.relative_to(STATIC_DIR)
        except ValueError:
            self.send_error(404)
            return
        if not p.is_file():
            self.send_error(404)
            return
        ct = content_type or {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(p.suffix, "application/octet-stream")
        # Inject the CURRENT time into index.html static asset URLs so the
        # browser is guaranteed to see a fresh ?v=… on every page load. Pages
        # are tiny; this is cheap and makes cache-busting bulletproof for
        # development.
        if name.endswith("index.html"):
            text = p.read_text(encoding="utf-8")
            text = text.replace("{{V}}", str(int(time.time())))
            data = text.encode("utf-8")
        else:
            data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_get(self, upstream: str) -> None:
        """GET proxy: forwards response status, content-type, body."""
        try:
            try:
                with urllib.request.urlopen(upstream, timeout=3) as resp:
                    body = resp.read()
                    ct = resp.headers.get("Content-Type", "application/octet-stream")
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ct)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read() or b""
                except Exception:
                    body = b""
                self.send_response(e.code)
                self.send_header("Content-Type",
                                 e.headers.get("Content-Type", "text/plain"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except urllib.error.URLError as e:
                return self._json(
                    {"ok": False, "error": f"upstream unreachable: {e.reason}"},
                    status=503,
                )
        except (BrokenPipeError, ConnectionResetError):
            # Client gave up before we finished writing — that's fine.
            pass

    def _proxy_post(self, upstream: str, body: dict) -> None:
        """POST proxy: JSON body forwarded, JSON response forwarded."""
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            upstream, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                reply = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type",
                                 resp.headers.get("Content-Type",
                                                  "application/json"))
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
        except urllib.error.HTTPError as e:
            try:
                reply = e.read() or b""
            except Exception:
                reply = b""
            self.send_response(e.code)
            self.send_header("Content-Type",
                             e.headers.get("Content-Type",
                                           "application/json"))
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
        except urllib.error.URLError as e:
            return self._json(
                {"ok": False, "error": f"upstream unreachable: {e.reason}"},
                status=503,
            )

    # Video MIME types accepted by /api/video.
    _VIDEO_TYPES = {
        ".mp4":  "video/mp4",
        ".m4v":  "video/mp4",
        ".mkv":  "video/x-matroska",
        ".mov":  "video/quicktime",
        ".webm": "video/webm",
        ".avi":  "video/x-msvideo",
    }

    def _serve_video(self, raw_path: str | None) -> None:
        """Serve a video file with HTTP byte-range support so <video> seeking
        works. Only files with known video extensions are served — naive
        guard against drive-by reads of arbitrary files."""
        if not raw_path:
            self.send_error(400, "missing path query")
            return
        try:
            p = Path(raw_path).expanduser().resolve()
        except (OSError, ValueError):
            self.send_error(400, "invalid path")
            return
        ct = self._VIDEO_TYPES.get(p.suffix.lower())
        if ct is None:
            self.send_error(403, "not a recognized video file")
            return
        if not p.is_file():
            self.send_error(404)
            return
        try:
            file_size = p.stat().st_size
        except OSError as e:
            self.send_error(500, str(e))
            return

        range_header = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", range_header) if range_header else None
        try:
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                if start >= file_size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ct)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(p, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client aborted (seek, pause, etc.) — fine.
            return

    def _sse_stream(self, label: str) -> None:
        # Special case: `/events/_all` multiplexes every service's log over
        # a single SSE connection so we don't burn N slots out of the
        # browser's per-origin HTTP/1.1 connection budget.
        if label == "_all":
            return self._sse_stream_all()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.app.pm.subscribe(label)
        try:
            while True:
                try:
                    line = q.get(timeout=15)
                    payload = json.dumps({"line": line})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.pm.unsubscribe(label, q)

    def _sse_stream_all(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.app.pm.subscribe_all()
        try:
            while True:
                try:
                    label, line = q.get(timeout=15)
                    payload = json.dumps({"label": label, "line": line})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.pm.unsubscribe_all(q)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._static("index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/config":
            return self._json(self.app.config_dict())
        if path == "/api/status":
            return self._json(self.app.pm.status())
        if path == "/api/recordings":
            return self._json(self.app.list_recordings())
        if path == "/api/video":
            qs = parse_qs(parsed.query)
            return self._serve_video(qs.get("path", [None])[0])
        if path.startswith("/events/"):
            return self._sse_stream(path[len("/events/"):])
        # Same-origin proxy to the local recorder streamer so the browser
        # doesn't reject cross-port localhost requests.
        if path == "/api/recorder/snapshot.jpg":
            return self._proxy_get(
                f"http://127.0.0.1:{RECORDER_PORT}/snapshot.jpg")
        if path == "/api/recorder/status":
            return self._proxy_get(
                f"http://127.0.0.1:{RECORDER_PORT}/record/status")
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/api/service/start":
            return self._json(self.app.start_service(
                body.get("label", ""), body.get("params", {})))
        if path == "/api/service/stop":
            return self._json(self.app.pm.stop(
                body.get("label", ""), body.get("mode", "kill")))
        if path == "/api/service/send":
            return self._json(self.app.pm.send(
                body.get("label", ""), body.get("text", "")))
        if path == "/api/fs/list":
            return self._json(_fs_list(body.get("path")))
        if path == "/api/instruction":
            text = (body.get("text") or "").strip()
            try:
                Path(LANG_INSTRUCTION_TXT).parent.mkdir(
                    parents=True, exist_ok=True)
                Path(LANG_INSTRUCTION_TXT).write_text(text + "\n")
            except OSError as e:
                return self._json({"ok": False, "error": str(e)})
            return self._json({"ok": True, "path": LANG_INSTRUCTION_TXT,
                               "length": len(text)})
        if path == "/api/guvcview/launch":
            return self._json(self.app.launch_guvcview(
                body.get("device", REC_DEFAULT_DEVICE)))
        if path == "/api/recorder/start":
            return self._proxy_post(
                f"http://127.0.0.1:{RECORDER_PORT}/record/start", body)
        if path == "/api/recorder/stop":
            return self._proxy_post(
                f"http://127.0.0.1:{RECORDER_PORT}/record/stop", body)
        if path == "/api/shutdown":
            self.app.pm.kill_all()
            self._json({"ok": True})
            threading.Thread(
                target=lambda: (time.sleep(0.3), os._exit(0)),
                daemon=True,
            ).start()
            return
        self.send_error(404)


# ============================================================
# Entry point
# ============================================================
def main() -> None:
    killed = cleanup_previous_runs()
    if killed:
        sys.stderr.write(
            f"[startup] killed {len(killed)} orphaned process(es) from prior run: "
            f"{killed}\n"
        )

    Handler.app = App()

    # Auto-start the local v4l2 recorder/streamer so the camera preview is
    # live from page load without any UI interaction. Best-effort: if the
    # device is busy or missing, the service simply isn't running and the
    # frontend shows "recorder offline".
    auto = Handler.app.start_service(
        "recorder", {"device": REC_DEFAULT_DEVICE}
    )
    if auto.get("ok"):
        sys.stderr.write(
            f"[server] recorder auto-started (pid {auto['pid']}, "
            f"device {REC_DEFAULT_DEVICE})\n"
        )
    else:
        sys.stderr.write(
            f"[server] recorder auto-start failed: {auto.get('error')}\n"
        )

    httpd = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{HTTP_PORT}/"
    sys.stderr.write(f"[server] G1 Inference UI on {url}\n")
    sys.stderr.write("[server] Ctrl+C to shut down and kill all subprocesses.\n")

    threading.Thread(
        target=lambda: (time.sleep(0.5), webbrowser.open(url)),
        daemon=True,
    ).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[server] shutting down\n")
    finally:
        Handler.app.pm.kill_all()


if __name__ == "__main__":
    main()
