"""eval_g1_dp.py + BGR -> RGB fix on the image observations.

Why this file exists
--------------------
image_client.py decodes JPEGs with cv2.imdecode, which returns BGR
(OpenCV convention). The trained policy, however, was fed RGB during
training because convert_unitree_json_to_lerobot.py explicitly runs
`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` (see line 178 of that script)
before writing frames into the LeRobot dataset. lerobot's training
data loader reads videos with torchcodec, which also returns RGB.

So the training pipeline is RGB end-to-end, but inference via
image_client delivers BGR. This file adds a single channel swap on
the four image observations right after they are produced, so the
ordering finally seen by the policy matches what it saw during
training.

The rest of the script is identical to eval_g1_dp.py.
"""
import traceback

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils_g1 import (
    cleanup_resources,
    # predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
    DualProcess_VLA,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def _bgr_to_rgb_inplace(observation: dict) -> None:
    """Swap channel order BGR -> RGB on every image observation tensor.

    process_images_and_observations() wraps the cv2.imdecode output
    (BGR uint8, shape (H, W, 3)) directly into a torch tensor without
    color conversion. The policy expects RGB, so we swap channels here.
    Operates in-place on the dict.
    """
    for k in list(observation.keys()):
        if "images" not in k:
            continue
        v = observation[k]
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            observation[k] = v[..., [2, 1, 0]].contiguous()
        else:  # numpy fallback
            observation[k] = np.ascontiguousarray(np.asarray(v)[..., ::-1])


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
):
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
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

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][1]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        ### sjh 260309
        arm_ctrl.speed_gradual_max()
        arm_ctrl.go_initial_pose(arm_ik)
        arm_ctrl.go_initial_pose_3(arm_ik)

        # ee_shared_mem["left"][:] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # ee_shared_mem["right"][:] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ###

        dp_vla = DualProcess_VLA(
            # system1_cfg_path="/home/goodman/unitree_v030/act_validation/0820000/pretrained_model",
            #system1_cfg_path="/home/goodman/unitree_v030/act_validation/260427/0420000_r7_260427/pretrained_model",
            system1_cfg_path="/home/goodman/unitree_v030/act_validation/B300_backup/lerobot_train_model/260511/dpact_dinov3_variance_config_final_260511/checkpoints/0040000/pretrained_model",
            ##
            # system1_cfg_path="/home/goodman/unitree_v030/act_validation/0095000_finetune_r012345/pretrained_model",
            system2_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        )
        is_first = True

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            logger_mp.info("Initializing robot to starting pose...")
            time.sleep(1.0)  # Give time for the robot to move

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )

                # ===== BGR -> RGB fix =====
                # image_client.cv2.imdecode gives BGR; the policy was trained
                # on RGB (convert_unitree_json_to_lerobot.py:178 converts the
                # collected frames BGR -> RGB before they enter the LeRobot
                # dataset; lerobot then decodes with torchcodec = RGB).
                _bgr_to_rgb_inplace(observation)
                # ==========================

                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor

                observation["task"] = step["task"]

                # 2. Get Action from Policy
                if is_first:
                    print('#' * 30)
                    print('* task :', step["task"])
                    print('#' * 30)
                    dp_vla.start_system2_thread()
                    is_first = False
                action_np = dp_vla.forward_system1(obs_dict=observation)

                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
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


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=None, root="/home/goodman/unitree_v030/act_validation/1_672/")

    eval_policy(cfg, dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
