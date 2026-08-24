import numpy as np
import torch
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


import logging_mp

### sjh 260309
from threading import Thread, Lock, Event
import queue, time
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Optional

from lerobot.policies.dpact.modeling_dpact import DPACTPolicy
from lerobot.policies.dpact_dino_v3_variance.modeling_dpact_dino_v3_variance import DPACTDINOv3VariancePolicy
from lerobot.policies.factory import make_pre_post_processors
import threading
import queue
import time

from PIL import Image
import time

from copy import deepcopy

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation

class DualProcess_VLA:
    def __init__(self, system1_cfg_path, system2_model_path, prompt_path,
                 system1_hz=30.0, system2_hz=2.0):
        self.system1_hz = system1_hz
        self.system2_hz = system2_hz
        self.prompt_path = prompt_path
        self.current_step = 0
        self.is_first=True

        self.system2_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            system2_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2",
            )
        processor = AutoProcessor.from_pretrained(system2_model_path, use_fast = True)
        processor.tokenizer.padding_side = "left"
        self.processor = processor

        #self.system1_model = DOTPolicy.from_pretrained(
        #    system1_cfg_path, strict=False)
        
        #self.system1_model = DPACTPolicy.from_pretrained(
        #    system1_cfg_path)
        
        self.system1_model = DPACTDINOv3VariancePolicy.from_pretrained(
            system1_cfg_path)

        self.device = next(self.system1_model.parameters()).device

        self.sys1_preprocessor, self.sys1_postprocessor = make_pre_post_processors(self.system1_model.config, pretrained_path=system1_cfg_path)

        #self.system1_model.eval()
        #self.system1_model.training = False
                
        self.current_latent = None

        self.latent_queue = queue.Queue(maxsize=1)
        self.thread_running = False
        # sjh
        self.latent_ready = threading.Event()

        # 💡 [수정] 스레드 간 데이터 공유를 위한 변수와 Lock 추가
        self.latest_obs_dict = None
        self.obs_lock = threading.Lock()

    def system2_thread(self):
        while self.thread_running:
            t0 = time.perf_counter()
            # 💡 [수정] Lock을 사용하여 안전하게 최신 관측값 복사
            with self.obs_lock:
                if self.latest_obs_dict is None:
                    # 아직 관측값이 없으면 잠시 대기
                    time.sleep(0.01)
                    continue
                # 깊은 복사로 데이터 일관성 보장
                current_obs = deepcopy(self.latest_obs_dict)

            # 복사한 최신 관측값으로 latent 추출
            latent = self.extract_latent(current_obs)
            
            if not self.latent_queue.empty():
                _ = self.latent_queue.get()
            self.latent_queue.put(latent)
            self.latent_ready.set()
            
            # System 2의 주기에 맞춰 대기
            elapsed = time.perf_counter() - t0
            #print('sys2_loop_time :', elapsed)
            remain = 1.0 / self.system2_hz - elapsed
            if remain > 0:
                time.sleep(remain)

    def extract_latent(self, obs_dict):
        #print('*'*30)        
        #print(obs_dict.keys())
        #print('*'*30)
        #print(obs_dict["task"])

        #print('* LEFT CAM',obs_dict['observation.images.cam_left_high'].shape)
        #print(obs_dict['observation.images.cam_left_high'])

        pil_img_l = Image.fromarray(np.array(obs_dict['observation.images.cam_left_high']).astype('uint8'))

        images = [
            pil_img_l
            ]
        
        #print('* LEFT CAM',pil_img_l.shape)
        #print(pil_img_l)

        with open(self.prompt_path, "r") as f:
            prompt = f.read().strip()

        query = (
                '<|im_start|>system\n'
                'You are a humanoid robot. You have just received the following image.'
                'Picture: <|vision_start|><|image_pad|><|vision_end|> - Captured by the head camera, showing both the robot and its environment.<|im_end|>\n'
                '<|im_start|>user\n'
                f'How should you move when you need to {prompt}?<|im_end|>\n'
                '<|im_start|>assistant\n'
            )
        
        #print(query)

        inputs = self.processor(text=query, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.system2_model.device)

        output = self.system2_model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True, temperature=None,
            )

        latent = output.hidden_states[0][-1][:, -1, :].float()
        return latent

    # 💡 [수정] start_system2_thread가 obs_dict를 인자로 받지 않도록 변경
    def start_system2_thread(self):
        if not self.thread_running:
            self.thread_running = True
            # 인자 없이 스레드 시작
            thread = threading.Thread(target=self.system2_thread)
            thread.daemon = True
            thread.start()

    def stop_system2_thread(self):
        self.thread_running = False

    def reset(self):
        self.system1_model.reset()
        self.current_step=0

    def forward_system1(self, obs_dict):
        # 💡 [수정] 최신 관측값을 스레드 공유 변수에 업데이트
        with self.obs_lock:
            self.latest_obs_dict = obs_dict
        if self.current_latent is None:
            if hasattr(self, "latent_ready"):
                self.latent_ready.wait()    
        self.current_latent = self.latent_queue.get() if not self.latent_queue.empty() else self.current_latent

        if self.is_first:
            self.system1_model.reset()
            self.sys1_preprocessor.reset()
            self.sys1_postprocessor.reset()
            self.system1_model.eval()
            self.is_first = False
        
        #print('* OBS_DICT', obs_dict.keys())
        #print('* STATE',obs_dict['observation.state'].shape)
        #print(obs_dict['observation.state'])
        #print('* LEFT CAM',obs_dict['observation.images.cam_left_high'].shape)
        #print(obs_dict['observation.images.cam_left_high'])
        #print('* RIGHT CAM',obs_dict['observation.images.cam_right_high'].shape)
        #print(obs_dict['observation.images.cam_right_high'])
        #print('* WRIST LEFT CAM',obs_dict['observation.images.cam_left_wrist'].shape)
        #print(obs_dict['observation.images.cam_left_wrist'])
        #print('* WRIST RIGHT CAM',obs_dict['observation.images.cam_right_wrist'].shape)
        #print(obs_dict['observation.images.cam_right_wrist'])

        with torch.inference_mode():
            observation = {
                "observation.images.cam_left_high": np.array(obs_dict['observation.images.cam_left_high']),
                "observation.images.cam_right_high": np.array(obs_dict['observation.images.cam_right_high']),
                "observation.images.cam_left_wrist": np.array(obs_dict['observation.images.cam_left_wrist']),
                "observation.images.cam_right_wrist":np.array(obs_dict['observation.images.cam_right_wrist']),
                "observation.state": obs_dict['observation.state'].type(torch.float32)
            }
            for name in observation:
                if "image" in name:
                    observation[name] = torch.from_numpy(observation[name]).type(torch.float32) / 255.
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                        
                observation[name] = observation[name].unsqueeze(0).to(self.device)
            observation["observation.latent"] = self.current_latent
            #observation["observation.latent"] = self.current_latent.unsqueeze(0)
            
            observation = self.sys1_preprocessor(observation)
            
            action = self.system1_model.select_action(observation)

            action = self.sys1_postprocessor(action)

            action= action.squeeze(0).to('cpu').numpy()
        # state = obs_dict['observation.state'].type(torch.float32).unsqueeze(0).to(self.system2_model.device)
        
        # img_l = np.array(obs_dict['observation.images.cam_left_high'])
        # img_l = torch.from_numpy(img_l).type(torch.float32) / 255.
        # img_l = img_l.permute(2,0,1).unsqueeze(0).to(self.system2_model.device)
        
        # img_r = np.array(obs_dict['observation.images.cam_right_high'])
        # img_r = torch.from_numpy(img_r).type(torch.float32) / 255.
        # img_r = img_r.permute(2,0,1).unsqueeze(0).to(self.system2_model.device)
        
        # img_wl = np.array(obs_dict['observation.images.cam_left_wrist'])
        # img_wl = torch.from_numpy(img_wl).type(torch.float32) / 255.
        # img_wl = img_wl.permute(2,0,1).unsqueeze(0).to(self.system2_model.device)
        
        # img_wr = np.array(obs_dict['observation.images.cam_right_wrist'])
        # img_wr = torch.from_numpy(img_wr).type(torch.float32) / 255.
        # img_wr = img_wr.permute(2,0,1).unsqueeze(0).to(self.system2_model.device)

        # observation = {
        #     "observation.images.cam_left_high": img_l,
        #     "observation.images.cam_right_high": img_r,
        #     "observation.images.cam_left_wrist": img_wl,
        #     "observation.images.cam_right_wrist": img_wr,
        #     "observation.state": state,
        #     "observation.latent": self.current_latent.unsqueeze(0),
        #     }
        # with torch.inference_mode():
        #     action = self.system1_model.select_action(observation)
        # action = action.squeeze(0)
        # action = action.to("cpu")
        return action

# def predict_action(
#     observation: dict[str, np.ndarray],
#     policy: PreTrainedPolicy,
#     device: torch.device,
#     preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
#     postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
#     use_amp: bool,
#     task: str | None = None,
#     use_dataset: bool | None = False,
#     robot_type: str | None = None,
# ):
#     observation = copy(observation)
#     with (
#         torch.inference_mode(),
#         torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
#     ):
#         # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
#         for name in observation:
#             if not use_dataset:
#                 # Skip non-tensor observations (like task strings)
#                 if not hasattr(observation[name], "unsqueeze"):
#                     continue
#                 if "images" in name:
#                     observation[name] = observation[name].type(torch.float32) / 255
#                     observation[name] = observation[name].permute(2, 0, 1).contiguous()

#             observation[name] = observation[name].unsqueeze(0).to(device)

#         observation["task"] = task if task else ""
#         observation["robot_type"] = robot_type if robot_type else ""

#         observation = preprocessor(observation)

#         # Compute the next action with the policy
#         # based on the current observation
#         action = policy.select_action(observation)
#         action = postprocessor(action)

#         # Remove batch dimension
#         action = action.squeeze(0)

#         # Move to cpu, if not already the case
#         action = action.to("cpu")

#     return action


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
