"""eval_g1_dp.py rewritten to drop lerobot's `EvalRealConfig` / `@parser.wrap()`
infrastructure entirely and use a plain argparse layer.

Features added on top of eval_g1_dp.py:
  - BGR -> RGB conversion (the trained policy expects RGB; default ON)
  - --episode_idx based init pose (fast parquet lookup, no LeRobotDataset load)
  - --use_dataset_hand_pose to also pull hand state from the dataset
  - Slow move to dataset arm pose BEFORE the 's' prompt
  - --system1_cfg_path / --system2_model_path CLI args
  - --no_bgr_to_rgb for A/B comparison

Pose sequence
-------------
  * --episode_idx omitted -> go_initial_pose -> go_initial_pose_3
                             (legacy hand-raising)
  * --episode_idx given   -> go_initial_pose -> go_initial_pose_2 ->
                             move_dual_arm_to_q_with_gravity(dataset arm pose,
                             t_move=--init_move_seconds)
"""
from __future__ import annotations

import argparse
import sys
import traceback
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import SynchronizedArray
from typing import Any

import cv2
import numpy as np
import torch

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils_g1 import (
    cleanup_resources,
    to_list,
    to_scalar,
    DualProcess_VLA,
)
from unitree_lerobot.eval_robot.utils.init_pose import (
    load_init_state_from_parquet,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


# ----------------------------------------------------------------------------
# Config (plain dataclass populated by argparse — no lerobot parser involved)
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # Robot hardware
    arm: str = "G1_29"             # G1_29 | G1_23
    ee: str = "brainco"            # brainco | dex3 | dex1 | inspire1
    motion: bool = False
    sim: bool = False
    visualization: bool = False

    # Closed-loop control
    frequency: float = 30.0

    # Initial pose
    episode_idx: int | None = None
    use_dataset_hand_pose: bool = False
    init_root: str = "/mnt/ssd/config_lerobot/config_lerobot_final_brainco_av1_260611"
    init_move_seconds: float = 3.0

    # Color
    bgr_to_rgb: bool = True

    # Dual process VLA model paths
    system1_cfg_path: str = (
        "/home/goodman/unitree_v030/act_validation/B300_backup/"
        "lerobot_train_model/260511/"
        "dpact_dinov3_variance_config_final_260511/"
        "checkpoints/0040000/pretrained_model"
    )
    system2_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Task / instruction
    # Priority: --task_text  >  --prompt_path file  >  dataset task (only if
    # --episode_idx is set and the above two fail/are absent).
    prompt_path: str = "/home/goodman/g1_eval_protocol/language_instruction.txt"
    task_text: str | None = None  # if set, override prompt_path & dataset


def parse_cli() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee", default="brainco", choices=["brainco", "dex3", "dex1", "inspire1"])
    p.add_argument("--motion", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--visualization", action="store_true")
    p.add_argument("--frequency", type=float, default=30.0)

    p.add_argument("--episode_idx", type=int, default=None,
                   help="Episode in --init_root to seed the initial pose. "
                        "If omitted, use legacy go_initial_pose_3.")
    p.add_argument("--use_dataset_hand_pose", action="store_true",
                   help="Also pull the hand state (12 dims) from the dataset.")
    p.add_argument("--init_root", type=str,
                   default="/mnt/ssd/config_lerobot/config_lerobot_final_brainco_av1_260611",
                   help="Dataset root used for --episode_idx lookup.")
    p.add_argument("--init_move_seconds", type=float, default=3.0,
                   help="How slowly to move to the dataset arm pose.")

    p.add_argument("--no_bgr_to_rgb", action="store_true",
                   help="Skip the BGR->RGB conversion (default is to apply it).")

    p.add_argument("--system1_cfg_path", type=str,
                   default=Config.system1_cfg_path,
                   help="Path to the system1 (lerobot) policy checkpoint directory.")
    p.add_argument("--system2_model_path", type=str,
                   default=Config.system2_model_path,
                   help="HF id or local path for the system2 VLM.")

    p.add_argument("--prompt_path", type=str,
                   default=Config.prompt_path,
                   help="Plain-text file containing the language instruction.")
    p.add_argument("--task_text", type=str, default=None,
                   help="Override the language instruction sent to the policy.")

    a = p.parse_args()
    return Config(
        arm=a.arm,
        ee=a.ee,
        motion=a.motion,
        sim=a.sim,
        visualization=a.visualization,
        frequency=a.frequency,
        episode_idx=a.episode_idx,
        use_dataset_hand_pose=a.use_dataset_hand_pose,
        init_root=a.init_root,
        init_move_seconds=a.init_move_seconds,
        bgr_to_rgb=not a.no_bgr_to_rgb,
        system1_cfg_path=a.system1_cfg_path,
        system2_model_path=a.system2_model_path,
        prompt_path=a.prompt_path,
        task_text=a.task_text,
    )


# ----------------------------------------------------------------------------
def _bgr_to_rgb_inplace(observation: dict) -> None:
    """Swap channel order BGR -> RGB on every image observation tensor.

    process_images_and_observations() wraps the cv2.imdecode output (BGR uint8,
    shape (H, W, 3)) directly into a torch tensor without color conversion.
    The trained policy expects RGB (convert_unitree_json_to_lerobot.py runs
    cv2.cvtColor(BGR2RGB) before writing frames; lerobot then loads videos
    with torchcodec which also yields RGB). This helper restores that missing
    inference-side conversion.
    """
    for k in list(observation.keys()):
        if "images" not in k:
            continue
        v = observation[k]
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            observation[k] = v[..., [2, 1, 0]].contiguous()
        else:
            observation[k] = np.ascontiguousarray(np.asarray(v)[..., ::-1])


def eval_policy(cfg: Config) -> None:
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key]
            for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # --- Initial pose ---
        init_arm_pose = None
        init_left_hand = None
        init_right_hand = None
        if cfg.episode_idx is None:
            logger_mp.info("Init pose: legacy mode (go_initial_pose_3, no dataset lookup).")
            if cfg.use_dataset_hand_pose:
                logger_mp.warning(
                    "--use_dataset_hand_pose ignored: requires --episode_idx."
                )
        else:
            init_state = load_init_state_from_parquet(cfg.init_root, episode_idx=cfg.episode_idx)
            init_arm_pose = init_state[:arm_dof].astype(np.float32)
            if cfg.use_dataset_hand_pose:
                init_left_hand  = init_state[arm_dof:arm_dof + ee_dof].tolist()
                init_right_hand = init_state[arm_dof + ee_dof:arm_dof + 2 * ee_dof].tolist()
            logger_mp.info(
                f"Init pose: episode={cfg.episode_idx}, root={cfg.init_root}, "
                f"arm_pose[:3]={init_arm_pose[:3].tolist()}, "
                f"hand_src={'dataset' if cfg.use_dataset_hand_pose else 'hardcoded [0.0]*' + str(ee_dof)}"
            )

        arm_ctrl.speed_gradual_max()
        arm_ctrl.go_initial_pose(arm_ik)
        if cfg.episode_idx is None:
            arm_ctrl.go_initial_pose_3(arm_ik)
        else:
            arm_ctrl.go_initial_pose_2(arm_ik)
            logger_mp.info(
                f"Slow-moving arms to dataset starting pose over "
                f"{cfg.init_move_seconds:.1f} s ..."
            )
            arm_ctrl.move_dual_arm_to_q_with_gravity(
                arm_ik, init_arm_pose, t_move=cfg.init_move_seconds,
            )

        # Hand init
        left_hand_init  = init_left_hand  if init_left_hand  is not None else [0.0] * ee_dof
        right_hand_init = init_right_hand if init_right_hand is not None else [0.0] * ee_dof
        if cfg.ee:
            ee_shared_mem["left"][:]  = to_list(left_hand_init)
            ee_shared_mem["right"][:] = to_list(right_hand_init)
        time.sleep(0.8)

        # --- Instruction source ---
        # The task instruction lives in a single text file. --task_text, if
        # given, overrides the file content. DualProcess_VLA reads the file
        # itself on every System-2 tick, so updates to the file (e.g. from
        # the GUI) take effect mid-run.
        if cfg.task_text:
            Path(cfg.prompt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.prompt_path).write_text(cfg.task_text.strip() + "\n")
            logger_mp.info(f"Wrote --task_text to {cfg.prompt_path}")
        logger_mp.info(f"Using prompt file: {cfg.prompt_path}")

        # --- DualProcess VLA ---
        dp_vla = DualProcess_VLA(
            system1_cfg_path=cfg.system1_cfg_path,
            system2_model_path=cfg.system2_model_path,
            prompt_path=cfg.prompt_path,
        )
        is_first = True

        # Robot is at its initial pose — wait for the operator before
        # entering the evaluation loop (mirrors eval_g1_groot's flow).
        # When stdout is a pipe (web UI), input()'s implicit prompt flush
        # can stay buffered behind tqdm / rich / accelerate wrappers — so
        # we print on its OWN line with flush=True (a trailing newline
        # forces line-buffered streams to flush immediately) and read the
        # answer separately.
        print("Press 'Run inference' button to start the evaluation.", flush=True)
        user_input = input()
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() != "s":
            logger_mp.info("Aborted by user.")
            return

        logger_mp.info("Starting evaluation loop ...")
        time.sleep(1.0)
        logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
        _last_printed_instr = None   # only re-print when the instruction changes
        while True:
            loop_start_time = time.perf_counter()

            # 1. Get Observations
            observation, current_arm_q = process_images_and_observations(
                tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape,
                is_binocular, has_wrist_cam, arm_ctrl,
            )

            # 2. BGR -> RGB color fix (default ON; disable with --no_bgr_to_rgb).
            if cfg.bgr_to_rgb:
                _bgr_to_rgb_inplace(observation)

            left_ee_state = right_ee_state = np.array([])
            if cfg.ee:
                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state  = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]
            state_tensor = torch.from_numpy(
                np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
            ).float()
            observation["observation.state"] = state_tensor
            # NOTE: task text is read inside DualProcess_VLA.extract_latent
            # from self.prompt_path, so no need to inject it here.

            # Per-step log: print the instruction on the FIRST step and on
            # every subsequent change. Keeps the console quiet during long
            # stretches where the operator hasn't edited the file.
            try:
                with open(cfg.prompt_path, "r") as _pf:
                    _cur_instr = _pf.read().strip()
            except OSError:
                _cur_instr = ""
            if _cur_instr != _last_printed_instr:
                print(f"[{idx}] {_cur_instr}", flush=True)
                _last_printed_instr = _cur_instr
            elif idx > 0 and idx % 500 == 0:
                # Heartbeat — print just the step index every 500 steps.
                print(f"[{idx}]", flush=True)

            # 3. Get Action from Policy
            if is_first:
                dp_vla.start_system2_thread()
                is_first = False
            action_np = dp_vla.forward_system1(obs_dict=observation)

            # 4. Execute Action
            arm_action = action_np[:arm_dof]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action  = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:]  = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value  = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
            idx += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger_mp.error("Eval failed with traceback:\n%s", tb)
    finally:
        if image_info:
            cleanup_resources(image_info)
        try:
            arm_ctrl.go_exit_pose(arm_ik)
        except Exception:
            pass
        logger_mp.info("Finally, exiting program…")


def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = parse_cli()
    eval_policy(cfg)


if __name__ == "__main__":
    main()
