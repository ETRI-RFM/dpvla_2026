"""Closed-loop G1 evaluation with a remote GR00T (N1.7) policy server.

Structure mirrors eval_g1.py, but the lerobot policy is replaced by a
ZMQ client that talks to a gr00t.eval.run_gr00t_server instance.

Run the server on the GPU machine first, e.g.:

    /mnt/ssd/Isaac-GR00T/.venv/bin/python -m gr00t.eval.run_gr00t_server \\
      --model-path /mnt/ssd/GROOT/outputs/train_g1_groot/<run>/checkpoint-<step> \\
      --embodiment-tag new_embodiment \\
      --device cuda --host 0.0.0.0 --port 5555
"""
from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np
import tyro

from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.utils.gr00t_client import (
    Gr00tClient,
    build_gr00t_observation,
    parse_gr00t_action_chunk,
)
from unitree_lerobot.eval_robot.utils.init_pose import (
    load_init_arm_pose_from_parquet,
    load_init_state_from_parquet,
)
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources, to_list

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


@dataclass
class EvalGr00tConfig:
    # Robot hardware (must match training data collection).
    arm: str = "G1_29"
    ee: str = "brainco"
    motion: bool = False
    sim: bool = False

    # Dataset path: used ONLY to recover the initial robot pose.
    root: str = "/mnt/ssd/config_lerobot/config_lerobot_final_brainco_av1_260611"
    # Which episode of the dataset to take the starting pose from.
    # If None (default), skip dataset lookup and use the legacy
    # `go_initial_pose_3` (hand-raising) pose from eval_g1_dp.py.
    episode_idx: int | None = None

    # Path to a plain-text file containing the task instruction
    # (must match the instruction style seen during training).
    prompt_path: str = "/home/goodman/g1_eval_protocol/language_instruction.txt"

    # GR00T inference server.
    server_host: str = "localhost"
    server_port: int = 5555
    server_timeout_ms: int = 60000

    # Closed-loop control.
    frequency: float = 30.0
    # Number of steps to execute from each predicted chunk before triggering
    # a fresh inference (receding horizon). The trained model emits 16-step
    # chunks; 16 = drain the queue fully each time, 8 = re-infer halfway.
    action_steps_per_chunk: int = 16

    # Length of each predicted action chunk produced by the GR00T model.
    # Must match the trained checkpoint (gr00t-n1.7 G1 finetune = 16).
    action_horizon: int = 16

    # If True, swap image channels BGR -> RGB before sending to the server.
    # image_client delivers BGR (cv2.imdecode); the n1.7 G1 checkpoint was
    # trained on RGB videos decoded by torchcodec. Default ON (matches the
    # training color convention); disable with --no-bgr-to-rgb for A/B test.
    bgr_to_rgb: bool = True

    # If True, initialize the hand pose from the dataset's episode start
    # (state[14:20] for left hand, state[20:26] for right hand) instead of
    # forcing fully-open [0.0]*6. Default ON when --episode-idx is set.
    use_dataset_hand_pose: bool = True


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read().strip()


def eval_loop(cfg: EvalGr00tConfig) -> None:
    logger_mp.info(f"Arguments: {cfg}")

    # Log the prompt once so we know what the file currently holds, but the
    # main loop will re-read it every iteration (hot-reload).
    logger_mp.info(f"Initial task prompt: {load_prompt(cfg.prompt_path)!r}")

    assert 1 <= cfg.action_steps_per_chunk <= cfg.action_horizon, (
        f"action_steps_per_chunk ({cfg.action_steps_per_chunk}) must be in "
        f"[1, action_horizon={cfg.action_horizon}]"
    )

    # --- GR00T client ---
    client = Gr00tClient(
        host=cfg.server_host,
        port=cfg.server_port,
        timeout_ms=cfg.server_timeout_ms,
    )
    logger_mp.info("Pinging GR00T server ...")
    logger_mp.info(f"  ping -> {client.ping()}")
    client.reset()
    logger_mp.info("  reset OK")

    image_info = None
    try:
        # --- Setup ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]
        arm_dof = robot_interface["arm_dof"]
        ee_dof = robot_interface["ee_dof"]

        tv_img_array = image_info["tv_img_array"]
        wrist_img_array = image_info["wrist_img_array"]
        tv_img_shape = image_info["tv_img_shape"]
        wrist_img_shape = image_info["wrist_img_shape"]
        is_binocular = image_info["is_binocular"]
        has_wrist_cam = image_info["has_wrist_cam"]

        assert arm_dof == 14, f"expected G1 dual-arm dof=14, got {arm_dof}"
        assert ee_dof == 6, (
            f"trained model expects 6-dim hand (brainco); got ee_dof={ee_dof}. "
            "Check cfg.ee."
        )

        # --- Initial pose ---
        # Two modes:
        #   (a) --episode-idx <N>  : load the arm (and optionally the hand)
        #                            from the dataset's first frame of episode N.
        #                            After user 's', the arms are explicitly
        #                            moved to that dataset pose.
        #   (b) no --episode-idx   : legacy eval_g1_dp.py behavior — use
        #                            go_initial_pose_3 (hand-raising pose) and
        #                            skip the dataset-driven arm move.
        init_arm_pose: np.ndarray | None = None
        init_left_hand  = [0.0] * ee_dof
        init_right_hand = [0.0] * ee_dof
        hand_src = "hardcoded fully-open [0.0]*6"

        if cfg.episode_idx is None:
            if cfg.use_dataset_hand_pose:
                logger_mp.warning(
                    "--use-dataset-hand-pose ignored: requires --episode-idx."
                )
            logger_mp.info(
                "Init pose: legacy mode (go_initial_pose_3, no dataset lookup)."
            )
        else:
            if cfg.use_dataset_hand_pose:
                init_state = load_init_state_from_parquet(cfg.root, episode_idx=cfg.episode_idx)
                init_arm_pose   = init_state[:arm_dof]
                init_left_hand  = init_state[arm_dof:arm_dof + ee_dof].tolist()
                init_right_hand = init_state[arm_dof + ee_dof:arm_dof + 2 * ee_dof].tolist()
                hand_src = f"dataset (left={init_left_hand}, right={init_right_hand})"
            else:
                init_arm_pose = load_init_arm_pose_from_parquet(
                    cfg.root, episode_idx=cfg.episode_idx, arm_dof=arm_dof,
                )
            logger_mp.info(
                f"Init pose: episode={cfg.episode_idx}, root={cfg.root}, "
                f"arm_pose[:3]={init_arm_pose[:3].tolist()}, hand_src={hand_src}"
            )

        arm_ctrl.speed_gradual_max()
        arm_ctrl.go_initial_pose(arm_ik)
        if cfg.episode_idx is None:
            # Legacy mode: hand-raising pose (eval_g1_dp.py style).
            arm_ctrl.go_initial_pose_3(arm_ik)
        else:
            # Dataset mode: shoulders-out pose, then slowly move to dataset start pose.
            arm_ctrl.go_initial_pose_2(arm_ik)
            logger_mp.info("Moving arms to dataset starting pose (slow) ...")
            arm_ctrl.move_dual_arm_to_q_with_gravity(arm_ik, init_arm_pose, t_move=3.0)

        # brainco: 0.0 = fully open, 1.0 = fully closed (see eval_g1.py:101).
        # Hand init source depends on --use_dataset_hand_pose (default: fully open).
        ee_shared_mem["left"][:]  = init_left_hand
        ee_shared_mem["right"][:] = init_right_hand
        time.sleep(0.8)

        # When stdout is a pipe (web UI), input()'s implicit prompt flush
        # can stay buffered behind tqdm / rich / accelerate wrappers — so
        # we print on its own line with flush=True (the trailing newline
        # forces line-buffered streams to flush immediately) and read the
        # answer separately.
        print("Press 'Run inference' button to start the evaluation.", flush=True)
        user_input = input()
        if user_input.lower() != "s":
            logger_mp.info("Aborted by user.")
            return

        logger_mp.info("Starting evaluation loop.")

        # --- Main control loop ---
        logger_mp.info(
            f"Starting evaluation @ {cfg.frequency} Hz, horizon={cfg.action_horizon}, "
            f"action_steps_per_chunk={cfg.action_steps_per_chunk}"
        )
        period = 1.0 / cfg.frequency
        # Refill the queue when the remaining items <= horizon - action_steps_per_chunk.
        # action_steps_per_chunk == horizon → refill only when fully drained.
        refill_threshold = cfg.action_horizon - cfg.action_steps_per_chunk
        action_queue: deque[np.ndarray] = deque()
        chunk_counter = 0
        step_in_chunk = 0
        global_step = 0
        _last_printed_prompt = None   # only re-print when prompt changes

        while True:
            step_start = time.perf_counter()

            # 1) Observation (every 30 Hz tick, always fresh)
            obs_lerobot, current_arm_q = process_images_and_observations(
                tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape,
                is_binocular, has_wrist_cam, arm_ctrl,
            )
            with ee_shared_mem["lock"]:
                full_state = np.array(ee_shared_mem["state"][:])
            left_ee_state  = full_state[:ee_dof]
            right_ee_state = full_state[ee_dof:ee_dof * 2]
            state_26d = np.concatenate(
                [current_arm_q, left_ee_state, right_ee_state], axis=0
            )

            # 2) Refill action queue when drained past receding-horizon threshold.
            #    The prompt is re-read from disk only here (when actually used by
            #    the model), so edits to demo_prompt.txt take effect on the next
            #    chunk inference.
            if len(action_queue) <= refill_threshold:
                prompt = load_prompt(cfg.prompt_path)
                gr_obs = build_gr00t_observation(
                    obs_lerobot, state_26d, prompt,
                    bgr_to_rgb=cfg.bgr_to_rgb,
                )
                t_inf = time.perf_counter()
                action_dict, _info = client.get_action(gr_obs)
                inf_ms = (time.perf_counter() - t_inf) * 1000.0
                chunk = parse_gr00t_action_chunk(action_dict)
                # Stack to (T, 26) rows: [arm14 | left_hand6 | right_hand6]
                stacked = np.concatenate(
                    [chunk["left_arm"], chunk["right_arm"],
                     chunk["left_hand"], chunk["right_hand"]],
                    axis=1,
                )
                action_queue.clear()
                action_queue.extend(stacked)
                chunk_counter += 1
                step_in_chunk = 0

            # 4) Pop one action and execute.
            action_np = action_queue.popleft()           # (26,)
            step_in_chunk += 1                           # 1-indexed within chunk
            arm_action = action_np[:arm_dof]             # (14,)
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)

            left_hand_cmd  = action_np[arm_dof:arm_dof + ee_dof]               # (6,)
            right_hand_cmd = action_np[arm_dof + ee_dof:arm_dof + 2 * ee_dof]  # (6,)
            if isinstance(ee_shared_mem["left"], SynchronizedArray):
                ee_shared_mem["left"][:]  = to_list(left_hand_cmd)
                ee_shared_mem["right"][:] = to_list(right_hand_cmd)
            else:
                # Not used for brainco, kept for parity with dex1.
                ee_shared_mem["left"].value  = float(left_hand_cmd[0])
                ee_shared_mem["right"].value = float(right_hand_cmd[0])

            # 5) Per-step log: print on the FIRST step and on every prompt
            # change after that — keeps the console quiet during long runs.
            if prompt != _last_printed_prompt:
                print(f"[{global_step}] {prompt}", flush=True)
                _last_printed_prompt = prompt
            elif global_step > 0 and global_step % 500 == 0:
                # Heartbeat — print just the step index every 500 steps.
                print(f"[{global_step}]", flush=True)
            global_step += 1

            time.sleep(max(0.0, period - (time.perf_counter() - step_start)))
    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user.")
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger_mp.error("Eval failed with traceback:\n%s", tb)
    finally:
        if image_info is not None:
            cleanup_resources(image_info)
        try:
            arm_ctrl.go_exit_pose(arm_ik)
        except Exception:
            pass
        client.close()
        logger_mp.info("Exiting program.")


def main() -> None:
    cfg = tyro.cli(EvalGr00tConfig)
    eval_loop(cfg)


if __name__ == "__main__":
    main()
