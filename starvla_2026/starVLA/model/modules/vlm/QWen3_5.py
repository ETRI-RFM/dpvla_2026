# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Optional

import torch
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        n_ro = int(qwenvl_config.get("readout_tokens", 0))
        self._readout_ids = None
        if n_ro > 0:
            names = [f"<|dpvla_ro_{i}|>" for i in range(n_ro)]
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": names})
            old_rows = self.model.get_input_embeddings().weight.shape[0]
            new_vocab = len(self.processor.tokenizer)
            if new_vocab > old_rows:
                self.model.resize_token_embeddings(new_vocab)
            self._readout_ids = self.processor.tokenizer.convert_tokens_to_ids(names)

            with torch.no_grad():
                emb = self.model.get_input_embeddings().weight
                mu = emb[: min(self._readout_ids)].mean(dim=0)
                for tid in self._readout_ids:
                    emb[tid] = mu
            print(f"[QWen3_5] readout tokens x{n_ro} enabled "
                  f"(ids={self._readout_ids[0]}..{self._readout_ids[-1]}, "
                  f"emb_rows={emb.shape[0]}, mean-init applied)")

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        qcfg = self.config.framework.get("qwenvl", {})
        prompt_style = str(qcfg.get("prompt_style", "native")).lower()
        if prompt_style == "dpvla_original":
            prompt_style = "embodiment"
        if prompt_style not in ("native", "embodiment", "detailed"):

            raise ValueError(f"qwenvl.prompt_style='{prompt_style}' not supported by the "
                             f"Qwen3.5 interface (native|embodiment|detailed)")
        persona = str(qcfg.get("system_prompt", "You are a single-arm gripper-type robot."))
        cam_descs = list(qcfg.get("camera_descriptions", [
            "Captured by the camera positioned directly above the robot.",
            "Captured by the wrist-mounted camera.",
        ]))

        if prompt_style == "detailed":
            robot_tag = str(qcfg.get("robot_tag", "a Franka Emika Panda robot arm"))
            arm_type = str(qcfg.get("arm_type", "a single arm"))
            extras = ""
            if bool(qcfg.get("has_waist", False)):
                extras += ", waist"
            if bool(qcfg.get("has_mobile_base", False)):
                extras += ", and mobile base"
            control_hz = qcfg.get("control_hz",
                                  self.config.datasets.vla_data.get("sys1_hz", 10))

            if bool(qcfg.get("prompt_include_chunk", False)):
                chunk_size = self.config.framework.get("action_model", {}).get("action_horizon", 20)
                chunk_phrase = f"the next {chunk_size} control actions"
            else:
                chunk_phrase = "the control actions"

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            if prompt_style in ("embodiment", "detailed"):
                cam_word = {1: "one camera", 2: "two cameras"}.get(
                    len(imgs), f"{len(imgs)} cameras")
                content = []
                for k, img in enumerate(imgs):
                    desc = cam_descs[k] if k < len(cam_descs) else f"Captured by camera {k + 1}."
                    content.append({"type": "text", "text": f"Picture {k + 1}: "})
                    content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": f" - {desc} "})
            else:
                content = [{"type": "image", "image": img} for img in imgs]

            if prompt_style == "detailed":
                prompt = (
                    f"The robot is {robot_tag} with {arm_type}{extras}. "
                    f"The control frequency is {control_hz} Hz. "
                    f"Please predict {chunk_phrase} to "
                    f"execute the following task: {instruction}."
                )
            elif "CoT_prompt" in self.config.datasets.vla_data:  # grounding prompt
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            if prompt_style in ("embodiment", "detailed"):
                system_block = (
                    f"{persona} You have just received the following images from {cam_word}."
                )
                msg = [
                    {"role": "system", "content": [{"type": "text", "text": system_block}]},
                    {"role": "user", "content": content},
                ]
            else:
                msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        template_kwargs = {}
        _qcfg = self.config.framework.get("qwenvl", {})
        if _qcfg.get("enable_thinking", None) is not None:
            template_kwargs["enable_thinking"] = bool(_qcfg.get("enable_thinking"))

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **template_kwargs
        )

        if self._readout_ids is not None and solutions is None:
            ids = batch_inputs["input_ids"]
            ro = torch.tensor(self._readout_ids, dtype=ids.dtype,
                              device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
            batch_inputs["input_ids"] = torch.cat([ids, ro], dim=1)
            batch_inputs["attention_mask"] = torch.cat(
                [batch_inputs["attention_mask"],
                 torch.ones_like(ro, dtype=batch_inputs["attention_mask"].dtype)], dim=1)

            if "mm_token_type_ids" in batch_inputs:
                mm = batch_inputs["mm_token_type_ids"]
                batch_inputs["mm_token_type_ids"] = torch.cat(
                    [mm, mm.new_zeros(ro.shape)], dim=1)

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
