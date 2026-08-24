"""G1 Inference Interface (draft).

A modern tkinter UI that orchestrates the G1 evaluation pipeline:
  Setup    : opens external terminals (re-launchable per service) for
             image_server / image_client / brainco_hand_server / GR00T.
  Mode     : Evaluation (long|short x task x episode resolved from
             evaluation_protocol_config.json) or Inference (free
             init_pose_idx + instruction).
  Record   : launches guvcview on /dev/videoX for live preview;
             recording itself is toggled inside guvcview.
  Run      : writes the chosen instruction to language_instruction.txt
             and launches eval_g1_groot in an external terminal.

Edit the constants at the top of the file before wiring this into a
stable run flow.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import threading
import tkinter as tk
from html.parser import HTMLParser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ============================================================
# Constants
# ============================================================
ROBOT_HOST = "unitree@192.168.123.164"
ROBOT_PASSWORD = "123"
ROBOT_MENU_CHOICE = "1"

# Python interpreter that has `pexpect` available (used to automate
# password+menu over ssh for the brainco hand server).
PEXPECT_PYTHON = "/usr/bin/python3"

EVAL_PROTOCOL_JSON = "/home/goodman/g1_eval_protocol/evaluation_protocol_config.json"
LANG_INSTRUCTION_TXT = "/home/goodman/g1_eval_protocol/language_instruction.txt"

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

# RealSense color stream on this machine is /dev/video4 (YUYV).
REC_DEFAULT_DEVICE = "/dev/video4"

# Inference-mode init_pose_idx allowed range.
INIT_POSE_MIN = 0
INIT_POSE_MAX = 33422

# Language list HTML (parsed at startup to populate the inference dropdowns).
LANGUAGE_LIST_HTML = (
    "/home/goodman/Downloads/actual_short_long_language_list_by_task.html"
)


# ============================================================
# Theme
# ============================================================
COLORS = {
    "bg":          "#F1F3F6",
    "card":        "#FFFFFF",
    "card_border": "#E3E6EB",
    "text":        "#1F2937",
    "subtle":      "#6B7280",
    "muted":       "#9CA3AF",
    "accent":      "#2563EB",
    "accent_hov":  "#1D4ED8",
    "accent_soft": "#EBF1FE",
    "success":     "#16A34A",
    "success_hov": "#15803D",
    "danger":      "#DC2626",
    "warn":        "#F59E0B",
    "field":       "#F7F8FA",
    "field_focus": "#FFFFFF",
    "chip":        "#EEF1F4",
}

FONT_FAMILY = "Ubuntu"
F_BODY        = (FONT_FAMILY, 10)
F_SMALL       = (FONT_FAMILY, 9)
F_SECTION     = (FONT_FAMILY, 12, "bold")
F_TITLE       = (FONT_FAMILY, 18, "bold")
F_NUMBER      = (FONT_FAMILY, 12, "bold")
F_BTN         = (FONT_FAMILY, 10)
F_BTN_BOLD    = (FONT_FAMILY, 10, "bold")
F_RUN         = (FONT_FAMILY, 11, "bold")
F_VALUE_LG    = (FONT_FAMILY, 16, "bold")


def setup_styles(root: tk.Tk) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")

    s.configure(".", background=COLORS["bg"], foreground=COLORS["text"], font=F_BODY)
    s.configure("TFrame", background=COLORS["bg"])
    s.configure("Card.TFrame", background=COLORS["card"])
    s.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=F_BODY)

    s.configure("TButton",
                background=COLORS["chip"], foreground=COLORS["text"],
                padding=(14, 8), font=F_BTN,
                borderwidth=0, relief="flat")
    s.map("TButton",
          background=[("active", "#E2E6EB"), ("pressed", "#D6DBE2")])

    s.configure("Accent.TButton",
                background=COLORS["accent"], foreground="#FFFFFF",
                padding=(14, 10), font=F_BTN_BOLD,
                borderwidth=0, relief="flat", focusthickness=0)
    s.map("Accent.TButton",
          background=[("active", COLORS["accent_hov"]), ("pressed", COLORS["accent_hov"])])

    s.configure("Primary.TButton",
                background=COLORS["accent"], foreground="#FFFFFF",
                padding=(24, 12), font=F_RUN,
                borderwidth=0, relief="flat", focusthickness=0)
    s.map("Primary.TButton",
          background=[("active", COLORS["accent_hov"]), ("pressed", COLORS["accent_hov"]),
                      ("disabled", "#B7CCF7")])

    s.configure("Success.TButton",
                background=COLORS["success"], foreground="#FFFFFF",
                padding=(14, 10), font=F_BTN_BOLD,
                borderwidth=0, relief="flat", focusthickness=0)
    s.map("Success.TButton",
          background=[("active", COLORS["success_hov"]), ("pressed", COLORS["success_hov"]),
                      ("disabled", "#B8DCC4")])

    s.configure("Danger.TButton",
                background=COLORS["danger"], foreground="#FFFFFF",
                padding=(14, 10), font=F_BTN_BOLD,
                borderwidth=0, relief="flat", focusthickness=0)
    s.map("Danger.TButton",
          background=[("active", "#B91C1C"), ("pressed", "#B91C1C"),
                      ("disabled", "#F0BABA")])

    s.configure("Ghost.TButton",
                background=COLORS["card"], foreground=COLORS["text"],
                padding=(12, 8), font=F_BTN,
                borderwidth=1, relief="solid",
                bordercolor=COLORS["card_border"],
                lightcolor=COLORS["card_border"],
                darkcolor=COLORS["card_border"])
    s.map("Ghost.TButton",
          background=[("active", COLORS["field"])],
          bordercolor=[("active", COLORS["accent"])])

    s.configure("TEntry",
                fieldbackground=COLORS["field"], foreground=COLORS["text"],
                bordercolor=COLORS["card_border"],
                lightcolor=COLORS["card_border"],
                darkcolor=COLORS["card_border"],
                padding=8, font=F_BODY)
    s.map("TEntry",
          fieldbackground=[("focus", COLORS["field_focus"])],
          bordercolor=[("focus", COLORS["accent"])],
          lightcolor=[("focus", COLORS["accent"])],
          darkcolor=[("focus", COLORS["accent"])])

    s.configure("TCombobox",
                fieldbackground=COLORS["field"],
                background=COLORS["card"],
                foreground=COLORS["text"],
                selectbackground=COLORS["field"],
                selectforeground=COLORS["text"],
                bordercolor=COLORS["card_border"],
                lightcolor=COLORS["card_border"],
                darkcolor=COLORS["card_border"],
                arrowcolor=COLORS["subtle"],
                padding=6, font=F_BODY)
    s.map("TCombobox",
          fieldbackground=[("readonly", COLORS["field"]), ("focus", COLORS["field_focus"])],
          foreground=[("readonly", COLORS["text"])],
          bordercolor=[("focus", COLORS["accent"])])
    root.option_add("*TCombobox*Listbox.background", COLORS["card"])
    root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    root.option_add("*TCombobox*Listbox.font", F_BODY)

    s.configure("TRadiobutton",
                background=COLORS["card"], foreground=COLORS["text"],
                font=F_BODY, focuscolor=COLORS["card"],
                indicatorcolor=COLORS["field"])
    s.map("TRadiobutton",
          background=[("active", COLORS["card"])],
          indicatorcolor=[("selected", COLORS["accent"])])

    s.configure("Vertical.TScrollbar",
                background=COLORS["bg"], troughcolor=COLORS["bg"],
                bordercolor=COLORS["bg"], arrowcolor=COLORS["subtle"],
                lightcolor=COLORS["bg"], darkcolor=COLORS["bg"])


# ============================================================
# Card widget (1px border using outer/inner frames)
# ============================================================
class Card(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=COLORS["card_border"],
                         highlightthickness=0, bd=0, **kw)
        self.body = tk.Frame(self, bg=COLORS["card"], padx=24, pady=20)
        self.body.pack(fill="both", expand=True, padx=1, pady=1)

    def title(self, text: str) -> tk.Label:
        return tk.Label(self.body, text=text, bg=COLORS["card"],
                        fg=COLORS["text"], font=F_SECTION,
                        anchor="w")

    def add_title(self, text: str) -> None:
        self.title(text).pack(anchor="w")

    def add_subtitle(self, text: str) -> None:
        tk.Label(self.body, text=text, bg=COLORS["card"],
                 fg=COLORS["subtle"], font=F_SMALL,
                 anchor="w").pack(anchor="w", pady=(2, 14))


# ============================================================
# External terminal helper
# ============================================================
def open_external_terminal(title: str, bash_cmd: str) -> bool:
    full = f"{bash_cmd}\nexec bash"
    candidates = [
        ["gnome-terminal", "--title", title, "--", "bash", "-lc", full],
        ["konsole", "-p", f"tabtitle={title}", "-e", "bash", "-lc", full],
        ["xterm", "-T", title, "-e", "bash", "-lc", full],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.Popen(cmd)
            return True
        except OSError:
            continue
    messagebox.showerror(
        "Terminal not found",
        "No supported terminal (gnome-terminal / konsole / xterm) found on PATH.",
    )
    return False


# ============================================================
# Setup-step commands
# ============================================================
# `bash -lc` does not reliably pick up conda init across machines, so every
# `conda activate` invocation we generate sources the first conda.sh we find.
CONDA_INIT_SNIPPET = (
    "{ "
    "source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null "
    "|| source /opt/conda/etc/profile.d/conda.sh 2>/dev/null "
    "|| true; "
    "}"
)


def cmd_image_server() -> str:
    return (
        "echo '=== Step 1: G1 image_server ==='\n"
        "echo 'password: 123, menu choice: 1, then run:'\n"
        "echo '  cd image_server/ && python image_server.py'\n"
        f"ssh {ROBOT_HOST}"
    )


def cmd_image_client() -> str:
    return (
        "echo '=== Step 2: camera client (preview) ==='\n"
        f"{CONDA_INIT_SNIPPET} && "
        f"cd {TELEOP_IMAGE_CLIENT_DIR} && "
        f"conda activate {TELEOP_CONDA_ENV} && "
        "python image_client.py"
    )


def cmd_brainco_hand() -> str:
    return (
        "echo '=== Step 3: brainco_hand_server ==='\n"
        "echo 'password: 123, menu choice: 1, then run:'\n"
        "echo '  cd brainco_hand_service/bin/ && ./brainco_hand_server'\n"
        f"ssh {ROBOT_HOST}"
    )


def _ssh_remote_pexpect_script(
    host: str, password: str, menu_choice: str, remote_cmd: str
) -> str:
    """Return a Python (pexpect) script that ssh's to `host`, auto-enters the
    password and the numeric menu choice, then runs `remote_cmd` and streams
    everything to stdout."""
    return (
        "import sys, time, pexpect\n"
        f'host = {host!r}\n'
        f'password = {password!r}\n'
        f'menu = {menu_choice!r}\n'
        f'remote = {remote_cmd!r}\n'
        'spawn_cmd = (\n'
        '    "ssh -tt "\n'
        '    "-o StrictHostKeyChecking=no "\n'
        '    "-o UserKnownHostsFile=/dev/null "\n'
        '    "-o LogLevel=ERROR " + host\n'
        ')\n'
        'c = pexpect.spawn(spawn_cmd, encoding="utf-8", timeout=30)\n'
        'c.logfile_read = sys.stdout\n'
        'try:\n'
        '    c.expect(["password:", "Password:"])\n'
        '    c.sendline(password)\n'
        '    time.sleep(2)\n'
        '    c.sendline(menu)\n'
        '    time.sleep(2)\n'
        '    c.sendline(remote)\n'
        '    c.expect(pexpect.EOF, timeout=None)\n'
        'except pexpect.EOF:\n'
        '    pass\n'
        'except Exception as e:\n'
        '    sys.stderr.write("pexpect error: " + str(e) + chr(10))\n'
        '    sys.exit(1)\n'
    )


def cmd_brainco_hand_auto() -> str:
    """Auto-runs ssh + password + menu + brainco_hand_server."""
    py = _ssh_remote_pexpect_script(
        host=ROBOT_HOST,
        password=ROBOT_PASSWORD,
        menu_choice=ROBOT_MENU_CHOICE,
        remote_cmd="cd brainco_hand_service/bin/ && ./brainco_hand_server",
    )
    return (
        "echo '=== Step 3: brainco_hand_server (auto) ==='\n"
        f"{PEXPECT_PYTHON} -c {shlex.quote(py)}"
    )


def cmd_gr00t_server(model_path: str) -> str:
    return (
        "echo '=== Step 4: GR00T server ==='\n"
        f"{GR00T_VENV_PY} -m gr00t.eval.run_gr00t_server "
        f"--model-path {model_path} "
        f"--embodiment-tag new_embodiment "
        f"--device cuda --host {GR00T_HOST} --port {GR00T_PORT}"
    )


def cmd_eval_client(episode_idx: int) -> str:
    return (
        "echo '=== Step 5: G1 inference client (eval_g1_groot) ==='\n"
        f"{CONDA_INIT_SNIPPET} && "
        f"cd {CLIENT_WORKDIR} && "
        f"conda activate {CLIENT_CONDA_ENV} && "
        "python -m unitree_lerobot.eval_robot.eval_g1_groot "
        f"  --arm {DEFAULT_ARM} --ee {DEFAULT_EE} "
        f"  --server-host localhost --server-port {GR00T_PORT} "
        f"  --episode-idx {episode_idx} "
        f"  --action-steps-per-chunk {DEFAULT_ACTION_STEPS}"
    )


def cmd_dp_client(system1_cfg_path: str, episode_idx: int) -> str:
    return (
        "echo '=== G1 inference client (eval_g1_dp_new / DP-VLA) ==='\n"
        f"{CONDA_INIT_SNIPPET} && "
        f"cd {CLIENT_WORKDIR} && "
        f"conda activate {CLIENT_CONDA_ENV} && "
        "python -m unitree_lerobot.eval_robot.eval_g1_dp_new "
        f"  --frequency {DP_VLA_FREQUENCY} --arm {DEFAULT_ARM} --ee {DEFAULT_EE} "
        f"  --system1_cfg_path {system1_cfg_path} "
        f"  --episode_idx {episode_idx} "
        f"  --use_dataset_hand_pose"
    )


# ============================================================
# Language list parsing (short_long_language_list_by_task.html)
# ============================================================
class _LangListParser(HTMLParser):
    """Extract instructions per (section, task#) from the language-list HTML."""

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

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = dict(attrs).get("class") or ""
        if tag == "section" and "lang-block" in cls:
            if "short" in cls:
                self._section = "short"
            elif "long" in cls:
                self._section = "long"
        elif tag == "h3":
            self._in_h3 = True
            self._h3 = ""
        elif tag == "li":
            self._in_li = True
            self._li = ""

    def handle_endtag(self, tag: str) -> None:
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
                        (self._section, self._task), []
                    ).append(txt)

    def handle_data(self, data: str) -> None:
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
# App
# ============================================================
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("G1 Inference")
        self.geometry("1180x900")
        self.minsize(960, 720)
        self.configure(bg=COLORS["bg"])

        setup_styles(self)

        with open(EVAL_PROTOCOL_JSON) as f:
            self.protocol: dict = json.load(f)

        self.lang_lists: dict[tuple[str, int], list[str]] = parse_language_list(
            LANGUAGE_LIST_HTML
        )

        self.status = tk.StringVar(value="ready")
        self.status_kind = tk.StringVar(value="ok")  # ok | info | warn | err

        # Model selection state
        self.model_type = tk.StringVar(value="gr00t")  # gr00t | dp_vla
        self.gr00t_path = tk.StringVar(value=GR00T_DEFAULT_MODEL)
        self.dp_path = tk.StringVar(value=DP_VLA_DEFAULT_MODEL)

        # Inference subprocess + log plumbing
        self.infer_proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()
        # Background services (camera_client, brainco_hand) — killed on close.
        self.bg_procs: list[subprocess.Popen] = []

        self._build()
        self.wm_protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._drain_log)

    # ----------------------------------------------------- layout
    def _build(self) -> None:
        self._build_header()
        self._build_inference_console()  # bottom-anchored first so scroll fills middle
        self._build_scroll_area()
        self._build_setup_card()
        self._build_model_card()
        self._build_mode_card()
        self._build_panel_holder()
        self._build_eval_panel()
        self._build_infer_panel()
        self._build_record_card()
        self._switch_mode()
        self._update_inference_buttons()

    def _build_header(self) -> None:
        hdr = tk.Frame(self, bg=COLORS["bg"])
        hdr.pack(side="top", fill="x", padx=28, pady=(22, 6))

        left = tk.Frame(hdr, bg=COLORS["bg"])
        left.pack(side="left")
        tk.Label(left, text="G1 Inference", bg=COLORS["bg"],
                 fg=COLORS["text"], font=F_TITLE).pack(anchor="w")
        tk.Label(left, text="Unitree G1 · GR00T N1.7 · evaluation orchestrator",
                 bg=COLORS["bg"], fg=COLORS["subtle"], font=F_SMALL).pack(anchor="w")

        # Status chip
        pill_outer = tk.Frame(hdr, bg=COLORS["card_border"])
        pill_outer.pack(side="right", pady=(8, 0))
        pill = tk.Frame(pill_outer, bg=COLORS["card"], padx=10, pady=6)
        pill.pack(padx=1, pady=1)
        self.status_dot = tk.Label(pill, text="●", bg=COLORS["card"],
                                   fg=COLORS["success"], font=F_BODY)
        self.status_dot.pack(side="left")
        tk.Label(pill, textvariable=self.status, bg=COLORS["card"],
                 fg=COLORS["subtle"], font=F_BODY).pack(side="left", padx=(6, 0))

        # Divider under header
        tk.Frame(self, bg=COLORS["card_border"], height=1).pack(
            side="top", fill="x", padx=28, pady=(14, 0))

    def _build_inference_console(self) -> None:
        cons = tk.Frame(self, bg=COLORS["bg"])
        cons.pack(side="bottom", fill="x", padx=28, pady=(10, 18))
        tk.Frame(cons, bg=COLORS["card_border"], height=1).pack(
            fill="x", pady=(0, 12))

        # Button row
        bar = tk.Frame(cons, bg=COLORS["bg"])
        bar.pack(fill="x", pady=(0, 10))
        self.go_btn = ttk.Button(
            bar, text="▶  Go to init pose",
            style="Primary.TButton", command=self._go_to_init_pose)
        self.go_btn.pack(side="left")
        self.run_s_btn = ttk.Button(
            bar, text="s  Run inference",
            style="Success.TButton", command=self._send_run_inference)
        self.run_s_btn.pack(side="left", padx=(8, 0))
        self.finish_btn = ttk.Button(
            bar, text="■  Finish (Ctrl+C)",
            style="Danger.TButton", command=self._finish_inference)
        self.finish_btn.pack(side="left", padx=(8, 0))

        self.summary = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.summary, bg=COLORS["bg"],
                 fg=COLORS["subtle"], font=F_SMALL).pack(
            side="right", padx=(0, 0))

        # Log (terminal-style)
        log_outer = tk.Frame(cons, bg=COLORS["card_border"])
        log_outer.pack(fill="x")
        log_wrap = tk.Frame(log_outer, bg="#0F1419")
        log_wrap.pack(fill="x", padx=1, pady=1)

        self.log = tk.Text(
            log_wrap, height=12, bg="#0F1419", fg="#D7DAE0",
            insertbackground="#D7DAE0",
            font=("Ubuntu Mono", 9), wrap="word",
            relief="flat", borderwidth=0, padx=10, pady=8,
            state="disabled",
        )
        self.log.pack(side="left", fill="both", expand=True)
        log_sb = ttk.Scrollbar(log_wrap, orient="vertical",
                               command=self.log.yview)
        self.log.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")

        self.log.tag_configure("cmd",  foreground="#8DAEFF")
        self.log.tag_configure("ok",   foreground="#16A34A")
        self.log.tag_configure("warn", foreground="#F59E0B")
        self.log.tag_configure("err",  foreground="#FF6B6B")
        self.log.tag_configure("hint", foreground="#9CA3AF")

        self._log_write(
            "Inference console — output of eval_g1_groot / eval_g1_dp_new "
            "will appear here.\n"
            "Workflow:  ① Go to init pose  →  watch for the “Enter 's' to …” "
            "prompt  →  ② Run inference  →  ③ Finish (Ctrl+C) when done.\n\n",
            "hint",
        )

    def _build_scroll_area(self) -> None:
        outer = tk.Frame(self, bg=COLORS["bg"])
        outer.pack(side="top", fill="both", expand=True, padx=28, pady=(14, 0))

        canvas = tk.Canvas(outer, bg=COLORS["bg"], highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.content = tk.Frame(canvas, bg=COLORS["bg"])
        win = canvas.create_window((0, 0), window=self.content, anchor="nw")

        def on_inner(_e: object = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.content.bind("<Configure>", on_inner)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))

        def on_wheel(e: tk.Event) -> None:
            if getattr(e, "delta", 0):
                canvas.yview_scroll(-int(e.delta / 120), "units")
            else:
                canvas.yview_scroll(-1 if e.num == 4 else 1, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(seq, on_wheel)

    # ----------------------------------------------------- cards
    def _new_card(self) -> Card:
        card = Card(self.content)
        card.pack(fill="x", pady=(0, 14))
        return card

    def _build_setup_card(self) -> None:
        card = self._new_card()
        card.add_title("Setup")
        card.add_subtitle("Open each service in its own terminal — restart any that drops")

        row = tk.Frame(card.body, bg=COLORS["card"])
        row.pack(fill="x")
        for i in range(3):
            row.columnconfigure(i, weight=1, uniform="step")

        self._step_button(row, 0, "1", "G1 Camera",
                          "ssh → image_server.py  (external terminal)",
                          lambda: open_external_terminal("G1 image_server",
                                                         cmd_image_server()))
        self._step_button(row, 1, "2", "Camera Client",
                          "tv env → image_client.py  (auto, background)",
                          lambda: self._launch_bg("camera_client",
                                                  cmd_image_client()))
        self._step_button(row, 2, "3", "Brainco Hand",
                          "ssh (auto pw+menu) → brainco_hand_server",
                          lambda: self._launch_bg("brainco_hand",
                                                  cmd_brainco_hand_auto()))

    def _step_button(self, parent: tk.Widget, col: int, num: str,
                     title: str, sub: str, cmd) -> None:
        outer = tk.Frame(parent, bg=COLORS["card_border"])
        outer.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
        inner = tk.Frame(outer, bg=COLORS["card"], padx=14, pady=14)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        head = tk.Frame(inner, bg=COLORS["card"])
        head.pack(fill="x")
        badge_outer = tk.Frame(head, bg=COLORS["accent"])
        badge_outer.pack(side="left")
        tk.Label(badge_outer, text=num, bg=COLORS["accent"], fg="#FFFFFF",
                 font=F_NUMBER, padx=10, pady=2).pack()
        tk.Label(head, text=title, bg=COLORS["card"],
                 fg=COLORS["text"], font=F_BTN_BOLD).pack(side="left", padx=(10, 0))

        tk.Label(inner, text=sub, bg=COLORS["card"],
                 fg=COLORS["subtle"], font=F_SMALL,
                 anchor="w").pack(anchor="w", pady=(6, 12))
        ttk.Button(inner, text="Launch", style="Accent.TButton",
                   command=cmd).pack(fill="x")

    def _build_model_card(self) -> None:
        card = self._new_card()
        card.add_title("Model")
        card.add_subtitle("Pick the policy backend; the path field follows the choice")

        # Type segmented buttons
        seg = tk.Frame(card.body, bg=COLORS["card"])
        seg.pack(fill="x")
        self.btn_gr00t = self._segment(
            seg, "GR00T", lambda: self._set_model_type("gr00t"))
        self.btn_gr00t.pack(side="left", padx=(0, 8))
        self.btn_dp = self._segment(
            seg, "DP-VLA", lambda: self._set_model_type("dp_vla"))
        self.btn_dp.pack(side="left")

        # Path row (label + entry + browse + launch[GR00T only])
        self.model_row = tk.Frame(card.body, bg=COLORS["card"])
        self.model_row.pack(fill="x", pady=(14, 0))
        self.model_row.columnconfigure(1, weight=1)

        self.model_path_label = tk.Label(
            self.model_row, text="", bg=COLORS["card"],
            fg=COLORS["subtle"], font=F_SMALL, anchor="w")
        self.model_path_label.grid(row=0, column=0, sticky="w",
                                   padx=(0, 16), pady=8)

        # Active entry is rebuilt on type switch (so textvariable stays linked
        # to the right StringVar without juggling traces).
        self.model_path_entry = ttk.Entry(self.model_row, textvariable=self.gr00t_path)
        self.model_path_entry.grid(row=0, column=1, sticky="we", padx=(0, 8))

        ttk.Button(self.model_row, text="Browse", style="Ghost.TButton",
                   command=self._browse_model).grid(row=0, column=2, padx=(0, 8))

        self.launch_gr00t_btn = ttk.Button(
            self.model_row, text="④ Launch GR00T",
            style="Success.TButton", command=self._launch_gr00t)
        self.launch_gr00t_btn.grid(row=0, column=3)

        self._refresh_model_buttons()
        self._apply_model_type()

    def _set_model_type(self, t: str) -> None:
        self.model_type.set(t)
        self._refresh_model_buttons()
        self._apply_model_type()

    def _refresh_model_buttons(self) -> None:
        t = self.model_type.get()
        for btn, val in [(self.btn_gr00t, "gr00t"), (self.btn_dp, "dp_vla")]:
            if val == t:
                btn.config(bg=COLORS["accent"], fg="#FFFFFF",
                           activebackground=COLORS["accent_hov"],
                           activeforeground="#FFFFFF")
            else:
                btn.config(bg=COLORS["chip"], fg=COLORS["text"],
                           activebackground="#E2E6EB",
                           activeforeground=COLORS["text"])

    def _apply_model_type(self) -> None:
        if self.model_type.get() == "gr00t":
            self.model_path_label.config(text="GR00T checkpoint")
            self.model_path_entry.config(textvariable=self.gr00t_path)
            self.launch_gr00t_btn.grid()
        else:
            self.model_path_label.config(text="system1_cfg_path")
            self.model_path_entry.config(textvariable=self.dp_path)
            self.launch_gr00t_btn.grid_remove()

    def _build_mode_card(self) -> None:
        card = self._new_card()
        card.add_title("Mode")

        row = tk.Frame(card.body, bg=COLORS["card"])
        row.pack(fill="x")

        self.mode = tk.StringVar(value="eval")
        self.btn_eval = self._segment(
            row, "평가  Evaluation", lambda: self._set_mode("eval"))
        self.btn_eval.pack(side="left", padx=(0, 8))
        self.btn_infer = self._segment(
            row, "추론  Inference", lambda: self._set_mode("infer"))
        self.btn_infer.pack(side="left")
        self._refresh_mode_buttons()

    def _segment(self, parent: tk.Widget, text: str, cmd) -> tk.Button:
        return tk.Button(parent, text=text, command=cmd,
                         bd=0, relief="flat", cursor="hand2",
                         padx=22, pady=10, font=F_BTN_BOLD,
                         highlightthickness=0)

    def _set_mode(self, m: str) -> None:
        self.mode.set(m)
        self._refresh_mode_buttons()
        self._switch_mode()

    def _refresh_mode_buttons(self) -> None:
        m = self.mode.get()
        for btn, val in [(self.btn_eval, "eval"), (self.btn_infer, "infer")]:
            if val == m:
                btn.config(bg=COLORS["accent"], fg="#FFFFFF",
                           activebackground=COLORS["accent_hov"],
                           activeforeground="#FFFFFF")
            else:
                btn.config(bg=COLORS["chip"], fg=COLORS["text"],
                           activebackground="#E2E6EB",
                           activeforeground=COLORS["text"])

    # ----- panel holder so mode cards stay between mode and record -----
    def _build_panel_holder(self) -> None:
        self.panel_holder = tk.Frame(self.content, bg=COLORS["bg"])
        self.panel_holder.pack(fill="x")

    def _build_eval_panel(self) -> None:
        self.eval_card = Card(self.panel_holder)
        c = self.eval_card
        c.add_title("Evaluation parameters")
        c.add_subtitle("Resolved from evaluation_protocol_config.json")

        grid = tk.Frame(c.body, bg=COLORS["card"])
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        self._field_label(grid, "Instruction type", 0)
        self.eval_inst_type = tk.StringVar(value="long")
        rb = tk.Frame(grid, bg=COLORS["card"])
        rb.grid(row=0, column=1, sticky="w", pady=8)
        ttk.Radiobutton(rb, text="Long", variable=self.eval_inst_type,
                        value="long", command=self._refresh_eval_preview).pack(
            side="left", padx=(0, 18))
        ttk.Radiobutton(rb, text="Short", variable=self.eval_inst_type,
                        value="short", command=self._refresh_eval_preview).pack(
            side="left")

        self._field_label(grid, "Task", 1)
        self.eval_task = tk.StringVar()
        task_cb = ttk.Combobox(grid, textvariable=self.eval_task,
                               values=sorted(self.protocol.keys()),
                               state="readonly", width=16)
        task_cb.grid(row=1, column=1, sticky="w", pady=8)
        task_cb.bind("<<ComboboxSelected>>",
                     lambda _: self._refresh_eval_episode())

        self._field_label(grid, "Episode", 2)
        self.eval_episode = tk.StringVar()
        self.eval_episode_cb = ttk.Combobox(grid, textvariable=self.eval_episode,
                                            state="readonly", width=16)
        self.eval_episode_cb.grid(row=2, column=1, sticky="w", pady=8)
        self.eval_episode_cb.bind("<<ComboboxSelected>>",
                                  lambda _: self._refresh_eval_preview())

        # Resolved view (highlighted block)
        resolved_outer = tk.Frame(c.body, bg=COLORS["card_border"])
        resolved_outer.pack(fill="x", pady=(14, 0))
        resolved = tk.Frame(resolved_outer, bg=COLORS["accent_soft"],
                            padx=16, pady=14)
        resolved.pack(fill="x", padx=1, pady=1)

        top = tk.Frame(resolved, bg=COLORS["accent_soft"])
        top.pack(fill="x")
        tk.Label(top, text="init_pose_idx",
                 bg=COLORS["accent_soft"], fg=COLORS["subtle"],
                 font=F_SMALL).pack(side="left")
        self.eval_init_pose_lbl = tk.Label(
            top, text="—", bg=COLORS["accent_soft"],
            fg=COLORS["accent"], font=F_VALUE_LG)
        self.eval_init_pose_lbl.pack(side="left", padx=(10, 0))

        tk.Label(resolved, text="instruction",
                 bg=COLORS["accent_soft"], fg=COLORS["subtle"],
                 font=F_SMALL).pack(anchor="w", pady=(10, 4))
        self.eval_inst_view = tk.Text(
            resolved, height=3, wrap="word",
            bg=COLORS["accent_soft"], fg=COLORS["text"],
            font=F_BODY, relief="flat", borderwidth=0,
            highlightthickness=0, padx=0, pady=0,
        )
        self.eval_inst_view.pack(fill="x")

    def _build_infer_panel(self) -> None:
        self.infer_card = Card(self.panel_holder)
        c = self.infer_card
        c.add_title("Inference parameters")
        c.add_subtitle(
            "Free init_pose + per-task language picker; or write a custom instruction"
        )

        grid = tk.Frame(c.body, bg=COLORS["card"])
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        # --- init_pose_idx with range hint + validation
        self._field_label(grid, "init_pose_idx", 0)
        ipose_row = tk.Frame(grid, bg=COLORS["card"])
        ipose_row.grid(row=0, column=1, sticky="w", pady=8)
        vcmd = (self.register(self._validate_init_pose_keystroke), "%P")
        self.infer_init_pose = tk.StringVar(value="3000")
        ttk.Entry(ipose_row, textvariable=self.infer_init_pose, width=12,
                  validate="key", validatecommand=vcmd).pack(side="left")
        tk.Label(ipose_row, text=f"  range: {INIT_POSE_MIN} – {INIT_POSE_MAX}",
                 bg=COLORS["card"], fg=COLORS["muted"],
                 font=F_SMALL).pack(side="left", padx=(6, 0))

        # --- Category (Short/Long × Task 1..4)
        self._field_label(grid, "Category", 1)
        self.infer_category = tk.StringVar()
        self.infer_category_cb = ttk.Combobox(
            grid, textvariable=self.infer_category,
            values=self._category_options(),
            state="readonly", width=22,
        )
        self.infer_category_cb.grid(row=1, column=1, sticky="w", pady=8)
        self.infer_category_cb.bind(
            "<<ComboboxSelected>>", lambda _: self._refresh_infer_combo())

        # --- Instruction (depends on category)
        self._field_label(grid, "Instruction", 2)
        self.infer_instr = tk.StringVar()
        self.infer_instr_cb = ttk.Combobox(
            grid, textvariable=self.infer_instr, state="readonly")
        self.infer_instr_cb.grid(row=2, column=1, sticky="we", pady=8)
        self.infer_instr_cb.bind(
            "<<ComboboxSelected>>", lambda _: self._on_infer_instr_pick())

        # --- Custom instruction with Send
        self._field_label(grid, "Custom", 3)
        custom_row = tk.Frame(grid, bg=COLORS["card"])
        custom_row.grid(row=3, column=1, sticky="we", pady=8)
        custom_row.columnconfigure(0, weight=1)
        self.infer_custom = tk.StringVar()
        ttk.Entry(custom_row, textvariable=self.infer_custom).grid(
            row=0, column=0, sticky="we", padx=(0, 8))
        ttk.Button(custom_row, text="Send", style="Accent.TButton",
                   command=self._send_custom_instruction).grid(row=0, column=1)

        # --- Active instruction preview (highlighted block)
        active_outer = tk.Frame(c.body, bg=COLORS["card_border"])
        active_outer.pack(fill="x", pady=(14, 0))
        active = tk.Frame(active_outer, bg=COLORS["accent_soft"],
                          padx=16, pady=14)
        active.pack(fill="x", padx=1, pady=1)
        head = tk.Frame(active, bg=COLORS["accent_soft"])
        head.pack(fill="x")
        tk.Label(head, text="Active instruction",
                 bg=COLORS["accent_soft"], fg=COLORS["subtle"],
                 font=F_SMALL).pack(side="left")
        self.infer_active_source = tk.StringVar(value="—")
        tk.Label(head, textvariable=self.infer_active_source,
                 bg=COLORS["accent_soft"], fg=COLORS["accent"],
                 font=F_BTN_BOLD).pack(side="left", padx=(10, 0))

        self.infer_active = tk.StringVar(value="")
        self.infer_active_view = tk.Text(
            active, height=3, wrap="word",
            bg=COLORS["accent_soft"], fg=COLORS["text"],
            font=F_BODY, relief="flat", borderwidth=0,
            highlightthickness=0, padx=0, pady=0,
        )
        self.infer_active_view.pack(fill="x", pady=(6, 0))

    # ---- inference panel helpers ----
    def _category_options(self) -> list[str]:
        opts = []
        for section in ("Short", "Long"):
            for t in (1, 2, 3, 4):
                if self.lang_lists.get((section.lower(), t)):
                    opts.append(f"{section} · Task {t}")
        return opts

    def _refresh_infer_combo(self) -> None:
        cat = self.infer_category.get()
        m = re.match(r"(Short|Long)\s*·\s*Task\s*(\d+)", cat)
        if not m:
            self.infer_instr_cb["values"] = []
            self.infer_instr.set("")
            return
        section, t = m.group(1).lower(), int(m.group(2))
        items = self.lang_lists.get((section, t), [])
        self.infer_instr_cb["values"] = items
        self.infer_instr.set("")

    def _on_infer_instr_pick(self) -> None:
        instr = self.infer_instr.get()
        if instr:
            self._set_active_instruction(instr, source=self.infer_category.get())

    def _send_custom_instruction(self) -> None:
        text = self.infer_custom.get().strip()
        if not text:
            self._set_status("Custom instruction is empty.", "warn")
            return
        self._set_active_instruction(text, source="Custom")

    def _set_active_instruction(self, text: str, source: str) -> None:
        self.infer_active.set(text)
        self.infer_active_source.set(source)
        self.infer_active_view.delete("1.0", "end")
        self.infer_active_view.insert("1.0", text)

    # ---- init_pose_idx keystroke validation ----
    def _validate_init_pose_keystroke(self, proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        try:
            v = int(proposed)
        except ValueError:
            return False
        return v <= INIT_POSE_MAX

    def _field_label(self, parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(parent, text=text, bg=COLORS["card"],
                 fg=COLORS["subtle"], font=F_SMALL,
                 anchor="w").grid(row=row, column=0, sticky="w",
                                  padx=(0, 20), pady=8)

    def _build_record_card(self) -> None:
        card = self._new_card()
        card.add_title("Video recording")
        card.add_subtitle("Launch guvcview on the Intel RealSense color stream "
                          "— start/stop recording inside guvcview")

        row = tk.Frame(card.body, bg=COLORS["card"])
        row.pack(fill="x")
        row.columnconfigure(3, weight=1)

        self._field_label(row, "Device", 0)
        self.rec_device = tk.StringVar(value=REC_DEFAULT_DEVICE)
        ttk.Combobox(row, textvariable=self.rec_device,
                     values=sorted(glob.glob("/dev/video*")),
                     width=18).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Button(row, text="▶ Launch guvcview", style="Accent.TButton",
                   command=self._launch_guvcview).grid(
            row=0, column=2, padx=(20, 0))
        self.rec_status = tk.StringVar(value="idle")
        tk.Label(row, textvariable=self.rec_status, bg=COLORS["card"],
                 fg=COLORS["subtle"], font=F_SMALL).grid(
            row=0, column=3, sticky="w", padx=(16, 0))

    # ----------------------------------------------------- logic
    def _collect_unique(self, key: str) -> list[str]:
        s: set[str] = set()
        for task in self.protocol.values():
            for entry in task.values():
                v = entry.get(key)
                if v:
                    s.add(v)
        return sorted(s)

    def _switch_mode(self) -> None:
        for w in self.panel_holder.winfo_children():
            w.pack_forget()
        if self.mode.get() == "eval":
            self.eval_card.pack(fill="x", pady=(0, 14))
        else:
            self.infer_card.pack(fill="x", pady=(0, 14))

    def _refresh_eval_episode(self) -> None:
        task = self.eval_task.get()
        if task in self.protocol:
            eps = sorted(self.protocol[task].keys(), key=int)
            self.eval_episode_cb["values"] = eps
            if eps:
                self.eval_episode.set(eps[0])
        self._refresh_eval_preview()

    def _refresh_eval_preview(self) -> None:
        task = self.eval_task.get()
        ep = self.eval_episode.get()
        inst_type = self.eval_inst_type.get()
        if task in self.protocol and ep in self.protocol[task]:
            entry = self.protocol[task][ep]
            self.eval_init_pose_lbl.config(text=str(entry["init_pose_idx"]))
            self.eval_inst_view.delete("1.0", "end")
            self.eval_inst_view.insert("1.0", entry.get(inst_type, ""))
        else:
            self.eval_init_pose_lbl.config(text="—")
            self.eval_inst_view.delete("1.0", "end")

    def _active_model_var(self) -> tk.StringVar:
        return self.gr00t_path if self.model_type.get() == "gr00t" else self.dp_path

    def _browse_model(self) -> None:
        var = self._active_model_var()
        init = str(Path(var.get()).parent) if var.get() else "/"
        title = ("Choose GR00T checkpoint directory"
                 if self.model_type.get() == "gr00t"
                 else "Choose DP-VLA system1_cfg_path (pretrained_model dir)")
        p = filedialog.askdirectory(title=title, initialdir=init)
        if p:
            var.set(p)

    def _launch_gr00t(self) -> None:
        path = self.gr00t_path.get().strip()
        if not path:
            self._set_status("GR00T model path is empty.", "err")
            return
        if not Path(path).exists():
            if not messagebox.askyesno(
                "Path not found",
                f"'{path}' does not exist on this machine.\nLaunch anyway?",
            ):
                return
        open_external_terminal("GR00T server", cmd_gr00t_server(path))
        self._set_status(f"GR00T server launched · {Path(path).name}", "info")

    def _launch_guvcview(self) -> None:
        if shutil.which("guvcview") is None:
            self._set_status("guvcview not installed.", "err")
            return
        device = self.rec_device.get().strip()
        try:
            subprocess.Popen(["guvcview", f"--device={device}"])
            self.rec_status.set(f"launched on {device}")
            self._set_status(f"guvcview running · {device}", "info")
        except OSError as e:
            self._set_status(f"guvcview failed: {e}", "err")

    def _resolve_run_args(self) -> tuple[int, str] | None:
        if self.mode.get() == "eval":
            task = self.eval_task.get()
            ep = self.eval_episode.get()
            if not task or not ep:
                self._set_status("Pick a task and an episode.", "warn")
                return None
            entry = self.protocol[task][ep]
            return int(entry["init_pose_idx"]), entry[self.eval_inst_type.get()]
        # ---- inference mode ----
        raw = self.infer_init_pose.get().strip()
        if raw == "":
            self._set_status("init_pose_idx is empty.", "warn")
            return None
        try:
            init_pose = int(raw)
        except ValueError:
            self._set_status("init_pose_idx must be an integer.", "err")
            return None
        if not (INIT_POSE_MIN <= init_pose <= INIT_POSE_MAX):
            self._set_status(
                f"init_pose_idx out of range "
                f"({INIT_POSE_MIN}..{INIT_POSE_MAX}).", "err"
            )
            return None
        instr = self.infer_active.get().strip()
        if not instr:
            self._set_status(
                "No active instruction. Pick one or write a custom and Send.",
                "warn",
            )
            return None
        return init_pose, instr

    # ----------------------------------------------------- inference subprocess
    def _go_to_init_pose(self) -> None:
        if self.infer_proc and self.infer_proc.poll() is None:
            self._set_status("Inference already running.", "warn")
            return
        resolved = self._resolve_run_args()
        if resolved is None:
            return
        episode_idx, instruction = resolved

        Path(LANG_INSTRUCTION_TXT).parent.mkdir(parents=True, exist_ok=True)
        Path(LANG_INSTRUCTION_TXT).write_text(instruction.strip() + "\n")

        if self.model_type.get() == "gr00t":
            cmd_str = cmd_eval_client(episode_idx)
            backend = "eval_g1_groot"
        else:
            dp_path = self.dp_path.get().strip()
            if not dp_path:
                self._set_status("DP-VLA system1_cfg_path is empty.", "err")
                return
            cmd_str = cmd_dp_client(dp_path, episode_idx)
            backend = "eval_g1_dp_new"

        self._log_clear()
        self._log_write(f"$ {cmd_str}\n\n", "cmd")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            self.infer_proc = subprocess.Popen(
                ["bash", "-lc", cmd_str],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1, text=True,
                start_new_session=True,
                env=env,
            )
        except OSError as e:
            self._set_status(f"Failed to launch: {e}", "err")
            self._log_write(f"\n[launch failed: {e}]\n", "err")
            return

        threading.Thread(
            target=self._stdout_reader, args=(self.infer_proc,),
            daemon=True,
        ).start()
        self._update_inference_buttons()
        short = instruction if len(instruction) <= 70 else instruction[:67] + "…"
        self._set_status(f"{backend} · waiting for 's'", "info")
        self.summary.set(f"instruction: {short}")

    def _send_run_inference(self) -> None:
        if not (self.infer_proc and self.infer_proc.poll() is None):
            self._set_status("No inference process running.", "warn")
            return
        try:
            assert self.infer_proc.stdin is not None
            self.infer_proc.stdin.write("s\n")
            self.infer_proc.stdin.flush()
        except OSError as e:
            self._set_status(f"send 's' failed: {e}", "err")
            self._log_write(f"\n[send 's' failed: {e}]\n", "err")
            return
        self._log_write("\n[sent 's' to stdin]\n", "ok")
        self._set_status("inference running", "ok")

    def _finish_inference(self) -> None:
        if not (self.infer_proc and self.infer_proc.poll() is None):
            self._set_status("No inference process running.", "warn")
            return
        try:
            os.killpg(os.getpgid(self.infer_proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            try:
                self.infer_proc.send_signal(signal.SIGINT)
            except OSError:
                pass
        self._log_write("\n[sent SIGINT — finishing]\n", "warn")
        self._set_status("stopping…", "warn")

    def _stdout_reader(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_queue.put(("data", line))
        except (OSError, ValueError):
            pass
        finally:
            self.log_queue.put(("exit", proc.wait()))

    # ---- background services (camera_client, brainco_hand) ----
    def _launch_bg(self, label: str, cmd_str: str) -> None:
        # Reap any dead bg processes first.
        self.bg_procs = [p for p in self.bg_procs if p.poll() is None]
        # If something with the same label is already running, leave it alone.
        for p in self.bg_procs:
            if getattr(p, "_label", None) == label:
                self._log_write(
                    f"\n[{label}] already running (pid {p.pid}); not relaunching.\n",
                    "warn",
                )
                return
        self._log_write(f"\n[{label}] $ {cmd_str}\n", "cmd")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd_str],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1, text=True,
                start_new_session=True,
                env=env,
            )
        except OSError as e:
            self._log_write(f"[{label}] launch failed: {e}\n", "err")
            self._set_status(f"{label} launch failed", "err")
            return
        proc._label = label  # type: ignore[attr-defined]
        self.bg_procs.append(proc)
        threading.Thread(
            target=self._bg_reader, args=(proc, label), daemon=True,
        ).start()
        self._set_status(f"{label} launched (pid {proc.pid})", "info")

    def _bg_reader(self, proc: subprocess.Popen, label: str) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_queue.put(("data", f"[{label}] {line}"))
        except (OSError, ValueError):
            pass
        finally:
            rc = proc.wait()
            self.log_queue.put(
                ("data", f"[{label}] exited with code {rc}\n")
            )

    def _drain_log(self) -> None:
        drained = 0
        while drained < 200:
            try:
                tag, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if tag == "data":
                self._log_write(payload)
            else:  # exit
                self._log_write(
                    f"\n[process exited with code {payload}]\n",
                    "warn" if payload else "ok",
                )
                self.infer_proc = None
                self._update_inference_buttons()
                self._set_status(
                    f"finished (exit {payload})",
                    "ok" if payload == 0 else "warn",
                )
            drained += 1
        self.after(80, self._drain_log)

    def _log_write(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        if tag:
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _update_inference_buttons(self) -> None:
        alive = bool(self.infer_proc and self.infer_proc.poll() is None)
        self.go_btn.config(state=("disabled" if alive else "normal"))
        self.run_s_btn.config(state=("normal" if alive else "disabled"))
        self.finish_btn.config(state=("normal" if alive else "disabled"))

    def _on_close(self) -> None:
        if self.infer_proc and self.infer_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.infer_proc.pid), signal.SIGINT)
                self.infer_proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.infer_proc.terminate()
                except OSError:
                    pass
        for p in self.bg_procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except OSError:
                    try:
                        p.terminate()
                    except OSError:
                        pass
        self.destroy()

    # ----------------------------------------------------- status
    def _set_status(self, msg: str, kind: str = "ok") -> None:
        self.status.set(msg)
        self.status_kind.set(kind)
        color = {
            "ok":   COLORS["success"],
            "info": COLORS["accent"],
            "warn": COLORS["warn"],
            "err":  COLORS["danger"],
        }.get(kind, COLORS["success"])
        self.status_dot.config(fg=color)


if __name__ == "__main__":
    App().mainloop()
