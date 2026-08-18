import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.DPVLA import DPVLA
from starVLA.model.modules.action_model.typed_latent_qformer import (
    TypedLatentQFormer,
    build_typed_masks,
    select_and_pad,
)
from starVLA.model.modules.dino_model.dinov3_encoder import DINOv3Encoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class DPVLAQFormerV1DefaultConfig:
    name: str = "DPVLA_QFormer_V1"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
            "s2_init_ckpt": None,
            "attn_implementation": "sdpa",
            "train_mode": "frozen",
            "enable_thinking": False,
            "latent_mode": "typed",
            "latent_source": "post_norm",
            "prompt_style": "minimal",
            "camera_labels": ["Front view", "Wrist view"],
            "instruction_marker": None,
            "tq_dim": 512,
            "tq_layers": 2,
            "tq_heads": 8,
            "tq_task_queries": 2,
            "tq_vision_queries": 4,
            "tq_fusion_queries": 2,
            "tq_dropout": 0.1,
            "tq_use_view_embed": True,
        }
    )

    dino: dict = field(
        default_factory=lambda: {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "pretrained": False,
            "frozen": True,
            "num_cameras": 2,
            "image_size": 224,
            "resize": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "head_type": "act",
            "action_dim": 7,
            "state_dim": 8,
            "action_horizon": 20,
            "dim_model": 512,
            "n_heads": 8,
            "n_dit_layers": 6,
            "dropout": 0.1,
            "dim_feedforward": 3200,
            "n_encoder_layers": 4,
            "n_decoder_layers": 1,
            "n_vae_encoder_layers": 4,
            "latent_dim": 32,
            "kl_weight": 10.0,
            "use_vae": True,
            "feedforward_activation": "relu",
            "pre_norm": False,
            "temporal_ensemble_coeff": 0.01,
            "latent_norm_mode": "identity",
            "latent_mlp_hidden": [1024, 512],
            "use_age_token": True,
            "age_max": 32,
            "use_zero_latent_aux": False,
            "zero_latent_weight": 0.5,
            "posterior_drop_prob": 0.0,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 5,
            "interleave_self_attention": False,
        }
    )

    future_action: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "n_offsets": 3,      
            "weight": 0.2,
            "head_hidden": 512,
        }
    )

    decision_module: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "horizon": 19,         
            "hist_len": 8,          
            "d_model": 128,
            "loss_weight": 0.1,     
            "use_sensitivity": True, 
            "sens_weight": 0.1,
        }
    )


@FRAMEWORK_REGISTRY.register("DPVLA_QFormer_V1")
class DPVLA_QFormer_V1(DPVLA):
    default_config_cls = DPVLAQFormerV1DefaultConfig

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        from starVLA.model.framework.base_framework import baseframework
        from starVLA.model.framework.share_tools import merge_framework_config

        baseframework.__init__(self)
        self.config = merge_framework_config(self.default_config_cls, config)

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self._apply_system2_train_mode()
        _s2_init = self.config.framework.qwenvl.get("s2_init_ckpt", None)
        if _s2_init:
            _pref = "qwen_vl_interface."
            _raw = torch.load(str(_s2_init), map_location="cpu")
            _sub = {k[len(_pref):]: v for k, v in _raw.items() if k.startswith(_pref)}
            _miss, _unexp = self.qwen_vl_interface.load_state_dict(_sub, strict=False)
            _rids = getattr(self.qwen_vl_interface, "_readout_ids", None)
            if _rids:
                with torch.no_grad():
                    _emb = self.qwen_vl_interface.model.get_input_embeddings().weight
                    _mu = _emb[: min(_rids)].mean(dim=0)
                    for _tid in _rids:
                        _emb[_tid] = _mu
            logger.info("[S2-INIT] robot-pretrained S2 loaded from %s | keys=%d "
                        "missing=%d unexpected=%d readout_reinit=%s",
                        _s2_init, len(_sub), len(_miss), len(_unexp), bool(_rids))

        dino_cfg = self.config.framework.dino
        self.dino_encoder = DINOv3Encoder(
            model_name=str(dino_cfg.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
            pretrained=bool(dino_cfg.get("pretrained", False)),
            frozen=bool(dino_cfg.get("frozen", True)),
            image_size=(lambda _v: [int(_v[0]), int(_v[1])]
                        if (hasattr(_v, "__len__") and not isinstance(_v, str))
                        else int(_v))(dino_cfg.get("image_size", 224)),
            resize=bool(dino_cfg.get("resize", True)),
        )

        qcfg = self.config.framework.qwenvl
        am = self.config.framework.action_model
        self._vlm_hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        am.dino_feat_dim = int(self.dino_encoder.num_channels)
        am.patches_per_view = int(self.dino_encoder.patches_per_view)
        am.num_cameras = int(dino_cfg.get("num_cameras", 2))

        self.latent_mode = str(qcfg.get("latent_mode", "typed")).lower()
        self.typed_qformer = None
        self.latent_qformer = None
        if self.latent_mode == "typed":
            n_task = int(qcfg.get("tq_task_queries", 2))
            n_vision = int(qcfg.get("tq_vision_queries", 4))
            n_fusion = int(qcfg.get("tq_fusion_queries", 2))
            k = n_task + n_vision + n_fusion
            self.typed_qformer = TypedLatentQFormer(
                in_dim=self._vlm_hidden_size,
                out_dim=int(qcfg.get("tq_dim", 512)),
                num_task_queries=n_task,
                num_vision_queries=n_vision,
                num_fusion_queries=n_fusion,
                num_layers=int(qcfg.get("tq_layers", 2)),
                num_heads=int(qcfg.get("tq_heads", 8)),
                num_views=am.num_cameras,
                dropout=float(qcfg.get("tq_dropout", 0.1)),
                use_view_embedding=bool(qcfg.get("tq_use_view_embed", True)),
            )
            am.latent_input_dim = int(qcfg.get("tq_dim", 512))
        elif self.latent_mode == "single_qformer":
            from starVLA.model.modules.action_model.latent_resampler import LatentQFormer
            k = int(qcfg.get("latent_tokens", 8))
            self.latent_qformer = LatentQFormer(
                in_dim=self._vlm_hidden_size,
                out_dim=int(qcfg.get("tq_dim", 512)),
                num_queries=k,
                num_layers=int(qcfg.get("tq_layers", 2)),
                num_heads=int(qcfg.get("tq_heads", 8)),
                num_vlm_layers=1,
            )
            am.latent_input_dim = int(qcfg.get("tq_dim", 512))
        elif self.latent_mode == "readout":
            k = int(qcfg.get("readout_tokens", 0))
            if k <= 0:
                raise ValueError("latent_mode=readout requires qwenvl.readout_tokens > 0")
            if self.qwen_vl_interface._readout_ids is None:
                raise RuntimeError("[QFormerV1] interface did not register readout tokens")
            am.latent_input_dim = self._vlm_hidden_size
        else:
            raise ValueError(
                f"unknown qwenvl.latent_mode='{self.latent_mode}' (typed|single_qformer|readout)")
        am.num_latent_tokens = k
        self.num_latent_tokens = k

        self.num_lang_tokens = (int(qcfg.get("num_lang_tokens", 16))
                                if bool(qcfg.get("use_lang_tokens", False)) else 0)
        if self.num_lang_tokens > 0:
            self.lang_projector = torch.nn.Linear(
                self._vlm_hidden_size, int(am.latent_input_dim))
            am.num_latent_tokens = k + self.num_lang_tokens

        self.prompt_style = str(qcfg.get("prompt_style", "minimal")).lower()
        if self.prompt_style not in ("minimal", "detailed", "embodiment", "jepa"):
            raise ValueError(
                f"qwenvl.prompt_style='{self.prompt_style}' not supported by "
                f"DPVLA_QFormer_V1 (minimal|detailed|embodiment|jepa)")
        if self.prompt_style == "jepa" and self.latent_mode != "readout":
            raise ValueError("prompt_style='jepa' requires latent_mode='readout'")
        marker = qcfg.get("instruction_marker", None)
        if not marker:
            marker = ("Instruction:" if self.prompt_style == "minimal"
                      else "execute the following task:")
        self._marker_text = str(marker)
        tok = self.qwen_vl_interface.processor.tokenizer
        self._marker_variants = []
        for m in (self._marker_text, "\n" + self._marker_text, " " + self._marker_text):
            ids = tok.encode(m, add_special_tokens=False)
            if ids and ids not in self._marker_variants:
                self._marker_variants.append(ids)
        self._boundary_token_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
        self._image_token_id = int(
            getattr(self.qwen_vl_interface.model.config, "image_token_id", 248056))
        self._camera_labels = list(qcfg.get("camera_labels",
                                            ["Front view", "Wrist view"]))

        self.latent_source = str(qcfg.get("latent_source", "post_norm")).lower()
        if self.latent_source not in ("post_norm", "pre_norm"):
            raise ValueError(
                f"unknown qwenvl.latent_source='{self.latent_source}' (post_norm|pre_norm)")
        self._last_decoder_layer = None
        if self.latent_source == "pre_norm":
            self._last_decoder_layer = self._get_qwen_decoder_layers(
                self.qwen_vl_interface.model)[-1]

        self.head_type = str(am.get("head_type", "act")).lower()
        if self.head_type == "act":
            from starVLA.model.modules.action_model.act_cvae_options import get_act_cvae_action_model
            am.n_patch_tokens = am.num_cameras * am.patches_per_view
            self.action_model = get_act_cvae_action_model(self.config)
        elif self.head_type == "dit":
            from starVLA.model.modules.action_model.DPFM_ActionHeader import DPFMActionHead
            self.action_model = DPFMActionHead(self.config)
        else:
            raise ValueError(f"unknown action_model.head_type='{self.head_type}' (act|dit)")
        self.action_horizon = int(am.action_horizon)

        fa = getattr(self.config.framework, "future_action", None)
        self.future_action_head = None
        if fa is not None and bool(fa.get("enabled", False)):
            _n_off = int(fa.get("n_offsets", 3))
            _hid = int(fa.get("head_hidden", 512))
            self.future_action_head = torch.nn.Sequential(
                torch.nn.LayerNorm(int(am.latent_input_dim)),
                torch.nn.Linear(int(am.latent_input_dim), _hid),
                torch.nn.GELU(),
                torch.nn.Linear(_hid, _n_off * int(am.action_dim)),
            )
            self._fa_n = _n_off
            self._fa_w = float(fa.get("weight", 0.2))
            logger.info(f"[QFormerV1] future_action aux ON n_offsets={_n_off} "
                        f"w={self._fa_w} (S2 latent → chunk 너머 행동 의도)")

        dm = getattr(self.config.framework, "decision_module", None)
        self.decision_module = None
        if dm is not None and bool(dm.get("enabled", False)):
            from starVLA.model.modules.decision_model.decision_module import DecisionModule
            n_cam = int(self.config.framework.dino.get("num_cameras", 2))
            d_anchor = (n_cam * int(self.dino_encoder.num_channels)
                        + int(am.latent_input_dim) + int(am.get("state_dim", 8)))
            d_hist = int(am.get("state_dim", 8)) + int(am.action_dim)
            self.decision_module = DecisionModule(
                d_hist=d_hist, d_anchor=d_anchor,
                d_chunk=int(am.action_dim) + 1,
                d_model=int(dm.get("d_model", 128)),
                horizon=int(dm.get("horizon", 19)))
            self._decision_w = float(dm.get("loss_weight", 0.1))
            self._decision_use_sens = bool(dm.get("use_sensitivity", True))
            self._decision_sens_w = float(dm.get("sens_weight", 0.1))
            logger.info(
                f"[QFormerV1] decision_module ON d_hist={d_hist} d_anchor={d_anchor} "
                f"horizon={int(dm.get('horizon', 19))} w={self._decision_w}")

        self._apply_freeze_modes()
        logger.info(
            f"[QFormerV1] latent_mode={self.latent_mode} K={k} "
            f"dim={am.latent_input_dim} source={self.latent_source} "
            f"prompt={self.prompt_style} marker='{self._marker_text}' head={self.head_type}")

    def _append_readout_tokens(self, batch_inputs):
        ro_ids = self.qwen_vl_interface._readout_ids
        ids = batch_inputs["input_ids"]
        ro = torch.tensor(ro_ids, dtype=ids.dtype, device=ids.device
                          ).unsqueeze(0).expand(ids.shape[0], -1)
        batch_inputs["input_ids"] = torch.cat([ids, ro], dim=1)
        batch_inputs["attention_mask"] = torch.cat(
            [batch_inputs["attention_mask"],
             torch.ones_like(ro, dtype=batch_inputs["attention_mask"].dtype)], dim=1)
        if "mm_token_type_ids" in batch_inputs:
            mm = batch_inputs["mm_token_type_ids"]
            batch_inputs["mm_token_type_ids"] = torch.cat([mm, mm.new_zeros(ro.shape)], dim=1)
        return batch_inputs

    def _build_inputs_and_masks(self, batch_images, instructions):
        if self.prompt_style in ("minimal", "jepa"):
            messages = []
            for imgs, instruction in zip(batch_images, instructions):
                content = []
                for i, img in enumerate(imgs):
                    label = (self._camera_labels[i] if i < len(self._camera_labels)
                             else f"View {i + 1}")
                    prefix = "" if i == 0 else "\n"
                    content.append({"type": "text", "text": f"{prefix}{label}: "})
                    content.append({"type": "image", "image": img})
                if self.prompt_style == "jepa":
                    content.append({"type": "text", "text": (
                        f"\nYour task is {instruction}. Infer the temporal dynamics "
                        f"from the frames and produce the corresponding policy actions ")})
                else:
                    content.append({"type": "text",
                                    "text": f"\nInstruction: {instruction}"})
                messages.append([{"role": "user", "content": content}])
            batch_inputs = self.qwen_vl_interface.processor.apply_chat_template(
                messages, tokenize=True, padding=True,
                add_generation_prompt=False,
                return_dict=True, return_tensors="pt",
            ).to(self.qwen_vl_interface.model.device)
            if self.latent_mode == "readout":
                batch_inputs = self._append_readout_tokens(batch_inputs)
        else:
            batch_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions)

        masks = None
        if self.latent_mode == "typed":
            masks = build_typed_masks(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                image_token_id=self._image_token_id,
                marker_id_variants=self._marker_variants,
                boundary_token_id=self._boundary_token_id,
                num_views=self.typed_qformer.num_views,
            )
        return batch_inputs, masks

    def _split_length_groups(self, batch_inputs):
        if not bool(self.config.framework.qwenvl.get("pad_free_groups", True)):
            B = batch_inputs["input_ids"].shape[0]
            return [(list(range(B)), batch_inputs)]
        ids, attn = batch_inputs["input_ids"], batch_inputs["attention_mask"]
        B, L = ids.shape
        lens = attn.sum(dim=1).tolist()
        if len(set(lens)) == 1 and int(lens[0]) == L:
            return [(list(range(B)), batch_inputs)]         
        grid = batch_inputs.get("image_grid_thw", None)
        pv = batch_inputs.get("pixel_values", None)
        n_img_per_sample = grid.shape[0] // B if grid is not None else 0
        rows_per_img = grid.prod(dim=1).tolist() if grid is not None else []
        pv_bounds = [0]
        for b in range(B):
            n = sum(rows_per_img[b * n_img_per_sample:(b + 1) * n_img_per_sample])
            pv_bounds.append(pv_bounds[-1] + int(n))
        groups = {}
        for b, n in enumerate(lens):
            groups.setdefault(int(n), []).append(b)
        out = []
        for n, idxs in sorted(groups.items()):
            g = {}
            g["input_ids"] = torch.stack([ids[b, -n:] for b in idxs])      
            g["attention_mask"] = torch.ones_like(g["input_ids"])
            if "mm_token_type_ids" in batch_inputs:
                g["mm_token_type_ids"] = torch.stack(
                    [batch_inputs["mm_token_type_ids"][b, -n:] for b in idxs])
            if pv is not None:
                g["pixel_values"] = torch.cat([pv[pv_bounds[b]:pv_bounds[b + 1]] for b in idxs], dim=0)
                g["image_grid_thw"] = torch.cat(
                    [grid[b * n_img_per_sample:(b + 1) * n_img_per_sample] for b in idxs], dim=0)
            out.append((idxs, g))
        return out

    def get_latent(self, batch_images, instructions):
        qwen_inputs, _ = self._build_inputs_and_masks(batch_images, instructions)
        B = qwen_inputs["input_ids"].shape[0]
        latents = [None] * B
        for idxs, g in self._split_length_groups(qwen_inputs):
            lat = self._latent_from_inputs(g)
            for j, b in enumerate(idxs):
                latents[b] = lat[j]
        return torch.stack(latents, dim=0)

    def _latent_from_inputs(self, qwen_inputs):
        masks = None
        if self.latent_mode == "typed":
            masks = build_typed_masks(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                image_token_id=self._image_token_id,
                marker_id_variants=self._marker_variants,
                boundary_token_id=self._boundary_token_id,
                num_views=self.typed_qformer.num_views,
            )
        if self.latent_source == "pre_norm":
            captured = []

            def _hook(module, inputs, output):
                captured.append(output[0] if isinstance(output, tuple) else output)

            handle = self._last_decoder_layer.register_forward_hook(_hook)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.qwen_vl_interface(
                        **qwen_inputs, output_attentions=False,
                        output_hidden_states=False, return_dict=True)
            finally:
                handle.remove()
            hidden = captured[0].float()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.qwen_vl_interface(
                    **qwen_inputs, output_attentions=False,
                    output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1].float()

        if self.latent_mode == "typed":
            base = self.typed_qformer(
                hidden,
                attention_mask=qwen_inputs["attention_mask"],
                task_mask=masks["task_mask"],
                vision_mask=masks["vision_mask"],
                fusion_mask=masks["fusion_mask"],
                view_ids=masks["view_ids"],
            )
        elif self.latent_mode == "single_qformer":
            base = self.latent_qformer(
                hidden, attention_mask=qwen_inputs["attention_mask"])
        else:
            base = hidden[:, -self.num_latent_tokens:, :]
        if self.num_lang_tokens > 0:
            base = torch.cat([base, self._lang_tokens(hidden, qwen_inputs)], dim=1)
        return base

    def _lang_tokens(self, hidden, qwen_inputs):
        masks = build_typed_masks(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            image_token_id=self._image_token_id,
            marker_id_variants=self._marker_variants,
            boundary_token_id=self._boundary_token_id,
            num_views=len(self._camera_labels),
        )
        sel, pad_mask, _ = select_and_pad(hidden, masks["task_mask"])
        sel = sel.detach()
        M = self.num_lang_tokens
        S = sel.shape[1]
        keep = ~pad_mask
        if S >= M:
            sel, keep = sel[:, :M], keep[:, :M]
        else:
            sel = torch.nn.functional.pad(sel, (0, 0, 0, M - S))
            keep = torch.nn.functional.pad(keep, (0, M - S))
        return self.lang_projector(sel) * keep.unsqueeze(-1).to(sel.dtype)

    def _age_tensor(self, examples, device):
        if "system2_age" not in examples[0]:
            return None
        return torch.tensor([int(ex["system2_age"]) for ex in examples],
                            device=device, dtype=torch.long)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        cur_images, sys2_images, instructions, state = self.align_model_input(examples)
        latent = self.get_latent(sys2_images, instructions)
        dino_features = self.dino_encoder(cur_images)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array([ex["action"] for ex in examples]),
                device=latent.device, dtype=torch.float32)
            actions_target = actions[:, -self.action_horizon:, :]
            action_is_pad = None
            if "action_is_pad" in examples[0]:
                action_is_pad = torch.as_tensor(
                    np.array([ex["action_is_pad"] for ex in examples]),
                    device=latent.device).bool()[:, -self.action_horizon:]
            st = self._state_tensor(state, latent.device)
            if self.head_type == "act":
                out = self.action_model(
                    latent.float(), dino_features.float(), actions_target,
                    st, action_is_pad)
                ret = {"action_loss": out["loss"],
                       "l1_loss": out["l1"], "kl_loss": out["kl"]}
                if "l1_zero" in out:            
                    ret["l1_zero_loss"] = out["l1_zero"]
                age = None
            else:
                age = self._age_tensor(examples, latent.device)
                out = self.action_model(
                    latent.float(), dino_features.float(), actions_target,
                    state=st, action_is_pad=action_is_pad, age=age)
                ret = {"action_loss": out["loss"], "fm_loss": out["loss"]}

        if self.future_action_head is not None and "future_action" in examples[0]:
            fut = torch.tensor(
                np.array([ex["future_action"] for ex in examples]),
                device=latent.device, dtype=torch.float32)          
            fpad = torch.as_tensor(
                np.array([ex.get("future_action_is_pad",
                                 np.zeros(fut.shape[1], dtype=bool))
                          for ex in examples]),
                device=latent.device).bool()                        
            pooled = latent.float().mean(dim=1)                     
            pred = self.future_action_head(pooled).view(fut.shape)
            m = (~fpad).float().unsqueeze(-1)
            l_fa = (torch.nn.functional.smooth_l1_loss(pred, fut, reduction="none")
                    * m).sum() / m.sum().clamp(min=1.0)
            ret["future_action_loss"] = l_fa.detach()
            ret["action_loss"] = ret["action_loss"] + self._fa_w * l_fa

        if self.decision_module is not None:
            dec_loss = self._decision_step(examples, latent, dino_features, st, age)
            if dec_loss is not None:
                ret["decision_loss"] = dec_loss.detach()
                ret["action_loss"] = ret["action_loss"] + self._decision_w * dec_loss
        return ret

    @staticmethod
    def _pil_jitter(im, rng):
        from PIL import ImageEnhance
        im = ImageEnhance.Brightness(im).enhance(0.85 + 0.30 * float(rng.random()))
        return ImageEnhance.Contrast(im).enhance(0.85 + 0.30 * float(rng.random()))

    def _decision_step(self, examples, cur_latent, cur_dino, cur_st, cur_age):
        if cur_st is None:
            return None
        idx = [i for i, ex in enumerate(examples)
               if int(ex.get("decision_pair_k", 0) or 0) > 0
               and "decision_pair_image" in ex and "decision_state_hist" in ex]
        if not idx:
            return None
        dev = cur_latent.device
        H = int(self.decision_module.horizon)
        ks = [min(int(examples[i]["decision_pair_k"]), H) for i in idx]
        b = len(idx)
        with torch.no_grad():
            p_cur = [to_pil_preserve(examples[i]["decision_pair_image"]) for i in idx]
            p_s2 = [to_pil_preserve(examples[i]["decision_pair_sys2_image"]) for i in idx]
            instr = [examples[i]["lang"] for i in idx]
            pl = self.get_latent(p_s2, instr)                       
            pd = self.dino_encoder(p_cur)                          
            pst = self._state_tensor(
                [examples[i]["decision_pair_state"] for i in idx], dev)
            page = self._age_tensor(
                [{"system2_age": int(examples[i].get("decision_pair_age", 0))}
                 for i in idx], dev)
            if self.head_type == "act":
                chunk_pair = self.action_model.predict_action(
                    pl.float(), pd.float(), pst)
                chunk_cur = self.action_model.predict_action(
                    cur_latent[idx].float(), cur_dino[idx].float(), cur_st[idx])
            else:
                chunk_pair = self.action_model.predict_action(
                    pl.float(), pd.float(), state=pst, age=page)
                chunk_cur = self.action_model.predict_action(
                    cur_latent[idx].float(), cur_dino[idx].float(),
                    state=cur_st[idx],
                    age=(cur_age[idx] if cur_age is not None else None))
            C = chunk_pair.shape[1]
            d_lab = torch.zeros(b, device=dev)
            for j, k in enumerate(ks):
                ov = max(C - k, 1)
                d_lab[j] = ((chunk_pair[j, k:k + ov] - chunk_cur[j, :ov]) ** 2).mean()
            if self._decision_use_sens:
                rng = np.random.RandomState(
                    (int(d_lab.sum().item() * 1e4) + b) % (2 ** 31))
                pd_aug = self.dino_encoder(
                    [[self._pil_jitter(im, rng) for im in ims] for ims in p_cur])
                if self.head_type == "act":
                    chunk_aug = self.action_model.predict_action(
                        pl.float(), pd_aug.float(), pst)
                else:
                    chunk_aug = self.action_model.predict_action(
                        pl.float(), pd_aug.float(), state=pst, age=page)
                s_lab = ((chunk_pair - chunk_aug) ** 2).mean(dim=(1, 2)).unsqueeze(-1)
            else:
                s_lab = torch.zeros(b, 1, device=dev)

            V = int(self.config.framework.dino.get("num_cameras", 2))
            P = pd.shape[1] // V
            anchor = torch.cat([
                pd.float().view(b, V, P, -1).mean(dim=2).flatten(1),  
                pl.float().mean(dim=1),                                
                pst.float().view(b, -1),                              
            ], dim=-1)
            hist = torch.tensor(np.stack([np.concatenate(
                [np.asarray(examples[i]["decision_state_hist"], dtype=np.float32),
                 np.asarray(examples[i]["decision_action_hist"], dtype=np.float32)],
                axis=1) for i in idx]), device=dev, dtype=torch.float32)
            hmask = torch.tensor(np.stack(
                [np.asarray(examples[i]["decision_hist_mask"]) for i in idx]),
                device=dev, dtype=torch.bool)
            csum = torch.cat([chunk_pair.sum(dim=1),
                              torch.ones(b, 1, device=dev)], dim=-1)
        pred = self.decision_module(
            hist, hmask, anchor, anchor, csum,
            torch.zeros(b, dtype=torch.long, device=dev))
        dmask = torch.zeros(b, H, dtype=torch.bool, device=dev)
        tgt = torch.zeros(b, H, device=dev)
        for j, k in enumerate(ks):
            dmask[j, k - 1] = True
            tgt[j, k - 1] = d_lab[j]
        losses = self.decision_module.loss(
            pred, tgt, s_lab, decay_mask=dmask,
            sens_weight=(self._decision_sens_w if self._decision_use_sens else 0.0))
        return losses["loss"]

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        cur_images, sys2_images, instructions, state = self.align_model_input(examples)
        latent = self.get_latent(sys2_images, instructions)
        dino_features = self.dino_encoder(cur_images)
        with torch.autocast("cuda", dtype=torch.float32):
            st = self._state_tensor(state, latent.device)
            if self.head_type == "act":
                pred = self.action_model.predict_action(
                    latent.float(), dino_features.float(), st)
            else:
                age = self._age_tensor(examples, latent.device)
                pred = self.action_model.predict_action(
                    latent.float(), dino_features.float(), state=st, age=age)
        return {"normalized_actions": pred.detach().cpu().numpy()}


    def save_separated(self, out_dir: str) -> str:
        out_path = super().save_separated(out_dir)            
        out = Path(out_path)
        prefix = {"typed": "typed_qformer.", "single_qformer": "latent_qformer."}.get(
            self.latent_mode)
        meta_f = out / "dpvla_meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        if prefix is not None:
            tq = {k: v for k, v in self.state_dict().items() if k.startswith(prefix)}
            fname = prefix.rstrip(".") + ".pt"
            torch.save(tq, out / fname)
            meta[prefix.rstrip(".")] = {
                "n_tensors": len(tq),
                "num_latent_tokens": int(self.num_latent_tokens),
            }
            print(f"[QFormerV1] {fname} ({len(tq)} tensors) added to {out}")
        else:
            meta["readout"] = {"num_latent_tokens": int(self.num_latent_tokens)}
        if self.num_lang_tokens > 0:
            lp = {k: v for k, v in self.state_dict().items()
                  if k.startswith("lang_projector.")}
            torch.save(lp, out / "lang_projector.pt")
            meta["lang_tokens"] = {"num_lang_tokens": int(self.num_lang_tokens),
                                   "n_tensors": len(lp)}
        meta["latent_mode"] = self.latent_mode
        meta_f.write_text(json.dumps(meta, indent=2))
        return out_path


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="examples/LIBERO/train_files/starvla_dpvla_qformer_v1_9b_act_6000pro.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mock_small", action="store_true",
                        help="0.8B VLM + random-init DINOv3로 축소")
    parser.add_argument("--prompt_style", type=str, default=None)
    parser.add_argument("--latent_mode", type=str, default=None,
                        help="typed | single_qformer | readout")
    parser.add_argument("--head_type", type=str, default=None, help="act | dit")
    parser.add_argument("--latent_source", type=str, default=None,
                        help="post_norm | pre_norm")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.prompt_style:
        cfg.framework.qwenvl.prompt_style = args.prompt_style
    if args.latent_mode:
        cfg.framework.qwenvl.latent_mode = args.latent_mode
        if args.latent_mode == "readout":
            cfg.framework.qwenvl.readout_tokens = 32
    if args.head_type:
        cfg.framework.action_model.head_type = args.head_type
    if args.latent_source:
        cfg.framework.qwenvl.latent_source = args.latent_source
    if args.mock_small:
        cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-0.8B"
        cfg.framework.qwenvl.attn_implementation = "sdpa"
        cfg.framework.dino.pretrained = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: DPVLA_QFormer_V1 = DPVLA_QFormer_V1(cfg).to(device)
    H = model._vlm_hidden_size
    _lat_mod = model.typed_qformer or model.latent_qformer
    n_tq = sum(p.numel() for p in _lat_mod.parameters()) if _lat_mod is not None else 0
    n_head = sum(p.numel() for p in model.action_model.parameters())
    print(f"[mock] VLM hidden={H} K={model.num_latent_tokens} mode={model.latent_mode} "
          f"tq_params={n_tq/1e6:.1f}M head_params={n_head/1e6:.1f}M "
          f"prompt={model.prompt_style} head={model.head_type}")

    from starVLA.dataloader.gr00t_lerobot.datasets import hz_parser_dpvla
    _vla = cfg.datasets.vla_data
    s1hz, s2hz = int(_vla.get("sys1_hz", 10)), int(_vla.get("sys2_hz", 2))

    _am = cfg.framework.action_model
    A_DIM, S_DIM = int(_am.action_dim), int(_am.get("state_dim", 8))
    N_CAM = int(cfg.framework.dino.get("num_cameras", 2))

    def make_sample(seed, lang="pick up the black bowl and place it on the plate"):
        rng = np.random.RandomState(seed)
        mk = lambda: Image.fromarray(rng.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        n = int(rng.randint(0, 200))
        T = int(_am.action_horizon)
        pad = np.zeros(T, dtype=bool); pad[T - int(rng.randint(0, 3)):] = True
        return {
            "image": [mk() for _ in range(N_CAM)],
            "system2_image": [mk() for _ in range(N_CAM)],
            "system2_age": n - hz_parser_dpvla(n, s1hz, s2hz),
            "lang": lang,
            "action": rng.uniform(-1, 1, size=(T, A_DIM)).astype(np.float32),
            "action_is_pad": pad,
            "state": rng.uniform(-1, 1, size=(1, S_DIM)).astype(np.float32),
        }

    batch = [make_sample(0), make_sample(1, lang="open the top drawer of the cabinet")]
    imgs = [ex["system2_image"] for ex in batch]
    langs = [ex["lang"] for ex in batch]
    qin, masks = model._build_inputs_and_masks(imgs, langs)
    tok = model.qwen_vl_interface.processor.tokenizer
    if model.latent_mode == "typed":
        for b in range(2):
            ids = qin["input_ids"][b]
            task_txt = tok.decode(ids[masks["task_mask"][b]])
            n_vis = int(masks["vision_mask"][b].sum())
            views = masks["view_ids"][b][masks["vision_mask"][b]]
            print(f"[mock] sample{b}: task_span='{task_txt}' vision_tokens={n_vis} "
                  f"views={sorted(set(views.tolist()))}")
            core = langs[b].split()[2]
            assert core in task_txt, f"task span missing instruction word: {task_txt}"
            assert not (masks["task_mask"][b] & masks["vision_mask"][b]).any()
            assert n_vis > 0 and set(views.tolist()) == set(range(N_CAM))
        print("[mock] M1 mask verification PASSED")
    elif model.latent_mode == "readout":
        ro = model.qwen_vl_interface._readout_ids
        for b in range(2):
            tail = qin["input_ids"][b, -len(ro):].tolist()
            assert tail == list(ro), f"readout tokens not at tail: {tail[:4]}..."
        print(f"[mock] M1 readout-tail verification PASSED (K={len(ro)}, "
              f"prompt={model.prompt_style})")

    model.eval()
    with torch.no_grad():
        l_a = model.get_latent(imgs, [langs[0], langs[0]])
        l_b = model.get_latent(imgs, [langs[0], "push the mug to the left side"])
    assert l_a.shape[1] == model.num_latent_tokens
    cos_ref = torch.nn.functional.cosine_similarity(
        l_a[0], l_b[0], dim=-1).min().item()
    d_swap = (l_a[1] - l_b[1]).norm().item()
    if model.latent_mode == "typed":
        nt, nv = model.typed_qformer.num_task_queries, model.typed_qformer.num_vision_queries
        d_task = (l_a[1, :nt] - l_b[1, :nt]).norm().item()
        d_vis = (l_a[1, nt:nt + nv] - l_b[1, nt:nt + nv]).norm().item()
        print(f"[mock] M2 instruction-swap: Δtask={d_task:.4f} Δvision={d_vis:.4f} "
              f"same-input cos={cos_ref:.6f}")
        assert d_task > 1e-3, "task bank must react to instruction change"
    else:
        print(f"[mock] M2 instruction-swap Δ={d_swap:.4f} same-input cos={cos_ref:.6f} "
              f"(mode={model.latent_mode})")
        assert d_swap > 1e-3, "latent must react to instruction change"
    _cos_thr = 0.995 if model.latent_mode == "readout" else 0.999
    assert cos_ref > _cos_thr, f"same input latent cosine too low: {cos_ref}"

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    for step in range(args.steps):
        out = model(batch)
        loss = out["action_loss"]
        opt.zero_grad()
        loss.backward()
        if step == 0:
            if model.latent_mode == "typed":
                for nm in ("task_queries", "vision_queries", "fusion_queries"):
                    g = getattr(model.typed_qformer, nm).grad
                    assert g is not None and g.norm() > 0, f"{nm} got no gradient"
                print("[mock] M3 typed query gradients flow ✓")
            elif model.latent_mode == "single_qformer":
                g = model.latent_qformer.query.grad
                assert g is not None and g.norm() > 0, "single-bank query got no gradient"
                print("[mock] M3 single-bank query gradients flow ✓")
            else:
                emb = model.qwen_vl_interface.model.get_input_embeddings().weight
                if emb.requires_grad:
                    ro = model.qwen_vl_interface._readout_ids
                    g = emb.grad[list(ro)].abs().sum().item()
                    assert g > 0, "readout embedding rows got no gradient"
                    print(f"[mock] M3 readout embedding gradients flow ✓ (sum {g:.2e})")
        opt.step()
        extra = f" l1={out['l1_loss'].item():.4f} kl={out['kl_loss'].item():.4f}" \
            if "l1_loss" in out else ""
        print(f"step {step}: loss={loss.item():.4f}{extra}")
        assert torch.isfinite(loss), "loss is NaN/Inf"

    model.eval()
    pred = model.predict_action([make_sample(2)])["normalized_actions"]
    print(f"[mock] predict_action shape = {pred.shape}")
    assert pred.shape == (1, model.action_horizon, A_DIM)
    print(f"[mock] DPVLA_QFormer_V1 mock-up PASSED ✅ "
          f"(prompt={model.prompt_style}, head={model.head_type})")
