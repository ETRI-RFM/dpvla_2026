import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.ACT_ActionHeader import get_act_action_model, pool_qwen_latent
from starVLA.model.modules.dino_model.dinov3_encoder import DINOv3Encoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

@dataclass
class DPVLADefaultConfig:

    name: str = "DPVLA"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
            "attn_implementation": "sdpa",
            "lora": False,
            "lora_rank": 16,
            "lora_alpha": 32,
            "enable_thinking": False,
        }
    )

    dino: dict = field(
        default_factory=lambda: {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "pretrained": False,
            "frozen": True,
            "num_cameras": 2,
            "image_size": 224,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "state_dim": 8,
            "action_horizon": 8,
            "dim_model": 512,
            "n_heads": 8,
            "dim_feedforward": 3200,
            "n_encoder_layers": 4,
            "n_decoder_layers": 1,
            "n_vae_encoder_layers": 4,
            "latent_dim": 32,
            "kl_weight": 10.0,
            "dropout": 0.1,
            "use_vae": True,
            "latent_mlp_hidden": [1024, 512],
            "feedforward_activation": "relu",
            "pre_norm": False,
        }
    )


@FRAMEWORK_REGISTRY.register("DPVLA")
class DPVLA(baseframework): 
    default_config_cls = DPVLADefaultConfig

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(self.default_config_cls, config)

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self._apply_system2_train_mode()

        dino_cfg = self.config.framework.dino
        self.dino_encoder = DINOv3Encoder(
            model_name=str(dino_cfg.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
            pretrained=bool(dino_cfg.get("pretrained", False)),
            frozen=bool(dino_cfg.get("frozen", True)),
            image_size=int(dino_cfg.get("image_size", 224)),
        )
        num_cameras = int(dino_cfg.get("num_cameras", 2))

        am = self.config.framework.action_model
        am.latent_input_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        am.dino_feat_dim = int(self.dino_encoder.num_channels)
        am.n_patch_tokens = num_cameras * int(self.dino_encoder.patches_per_view)

        self.action_model = get_act_action_model(self.config)
        self.action_horizon = int(am.action_horizon)

        self._apply_freeze_modes()

    def _apply_freeze_modes(self):
        for module in (self.dino_encoder, self.qwen_vl_interface):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_freeze_modes()
        return self

    def _apply_system2_train_mode(self):
        qcfg = self.config.framework.qwenvl
        mode = str(qcfg.get("train_mode", "frozen")).lower()
        qmodel = self.qwen_vl_interface.model

        if mode == "frozen":
            for p in qmodel.parameters():
                p.requires_grad = False
        elif mode == "full":
            for p in qmodel.parameters():
                p.requires_grad = True
        elif mode == "partial":
            for p in qmodel.parameters():
                p.requires_grad = False
            n = int(qcfg.get("trainable_last_n_layers", 2))
            layers = self._get_qwen_decoder_layers(qmodel)
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            logger.info(f"[System2] partial fine-tune: last {n}/{len(layers)} decoder layers")
        elif mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError("qwenvl.train_mode='lora' requires peft. `pip install peft`.")
            lora = LoraConfig(
                r=int(qcfg.get("lora_rank", 16)),
                lora_alpha=int(qcfg.get("lora_alpha", 32)),
                target_modules=qcfg.get("lora_target_modules", "all-linear"),
                bias="none",
            )
            self.qwen_vl_interface.model = get_peft_model(qmodel, lora)
        else:
            raise ValueError(f"unknown qwenvl.train_mode='{mode}' (frozen|full|partial|lora)")

        n_train = sum(p.numel() for p in self.qwen_vl_interface.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.qwen_vl_interface.parameters())
        logger.info(f"[System2] train_mode={mode}: trainable {n_train/1e6:.1f}M / {n_all/1e6:.1f}M")

    def _get_qwen_decoder_layers(self, qmodel):
        lang, other = [], []
        for name, mod in qmodel.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
                low = name.lower()
                if any(v in low for v in ("visual", "vision", "vit")):
                    continue
                (lang if "language_model" in low else other).append((len(mod), mod))
        pool = lang or other
        if not pool:
            raise RuntimeError("Could not locate Qwen language decoder layers.")
        return max(pool, key=lambda x: x[0])[1]

    def align_model_input(self, examples: List[dict]):
        cur_images = [to_pil_preserve(ex["image"]) for ex in examples]
        sys2_images = (
            [to_pil_preserve(ex["system2_image"]) for ex in examples]
            if "system2_image" in examples[0] else cur_images
        )
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        return cur_images, sys2_images, instructions, state

    def get_latent(self, batch_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
        qcfg = self.config.framework.qwenvl
        return pool_qwen_latent(
            out.hidden_states[-1],
            attention_mask=qwen_inputs.get("attention_mask", None),
            mode=qcfg.get("latent_pool", "last"),
            n=int(qcfg.get("latent_pool_n", 64)),
        )

    def _state_tensor(self, state, device):
        if state is None:
            return None
        return torch.from_numpy(np.array(state)).to(device=device, dtype=torch.float32)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        cur_images, sys2_images, instructions, state = self.align_model_input(examples)
        vl_embs = self.get_latent(sys2_images, instructions)   
        dino_features = self.dino_encoder(cur_images)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array([ex["action"] for ex in examples]), device=vl_embs.device, dtype=torch.float32
            )
            actions_target = actions[:, -self.action_horizon :, :]
            action_is_pad = None
            if "action_is_pad" in examples[0]:
                action_is_pad = torch.as_tensor(
                    np.array([ex["action_is_pad"] for ex in examples]), device=vl_embs.device
                ).bool()[:, -self.action_horizon :]
            st = self._state_tensor(state, vl_embs.device)
            out = self.action_model(vl_embs, dino_features, actions_target, st, action_is_pad)

        return {"action_loss": out["loss"], "l1_loss": out["l1"], "kl_loss": out["kl"]}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        cur_images, sys2_images, instructions, state = self.align_model_input(examples)
        vl_embs = self.get_latent(sys2_images, instructions)   
        dino_features = self.dino_encoder(cur_images)        
        with torch.autocast("cuda", dtype=torch.float32):
            st = self._state_tensor(state, vl_embs.device)
            pred = self.action_model.predict_action(vl_embs, dino_features, st)
        return {"normalized_actions": pred.detach().cpu().numpy()}

    def save_separated(self, out_dir: str) -> str:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sd = self.state_dict()
        sys1 = {
            k: v for k, v in sd.items()
            if k.startswith("dino_encoder.") or k.startswith("action_model.")
        }
        torch.save(sys1, out / "system1.pt")

        is_lora = hasattr(self.qwen_vl_interface.model, "peft_config")
        if is_lora:
            self.qwen_vl_interface.model.save_pretrained(str(out / "system2_lora"))
        sys2_kind = "lora" if is_lora else "base"

        OmegaConf.save(config=self.config, f=str(out / "dpvla_config.yaml"))
        meta = {
            "system2_kind": sys2_kind,
            "base_vlm": str(self.config.framework.qwenvl.base_vlm),
            "n_system1_tensors": len(sys1),
            "action_horizon": int(self.action_horizon),
        }
        with open(out / "dpvla_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DPVLA] separated save -> {out} | system1.pt ({len(sys1)} tensors) + system2[{sys2_kind}]")
        return str(out)

if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="examples/LIBERO/train_files/starvla_dpvla_libero.yaml")
    parser.add_argument("--steps", type=int, default=5)
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: DPVLA = DPVLA(cfg).to(device)
    H = model.qwen_vl_interface.model.config.hidden_size
    print(f"[mock] VLM hidden={H}  dino_ch={model.dino_encoder.num_channels} "
          f"patches/view={model.dino_encoder.patches_per_view}  chunk={model.action_horizon}")

    def make_sample(seed):
        rng = np.random.RandomState(seed)
        img = Image.fromarray(rng.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img2 = Image.fromarray(rng.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        return {
            "image": [img, img],           
            "system2_image": [img2, img2],
            "lang": "pick up the black bowl and place it on the plate",
            "action": rng.uniform(-1, 1, size=(model.action_horizon, 7)).astype(np.float32),
            "state": rng.uniform(-1, 1, size=(1, 8)).astype(np.float32),
        }

    batch = [make_sample(0), make_sample(1)]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    for step in range(args.steps):
        out = model(batch)
        loss = out["action_loss"]
        opt.zero_grad()
        loss.backward()
        grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        opt.step()
        print(f"step {step}: action_loss={loss.item():.4f} "
              f"l1={out['l1_loss'].item():.4f} kl={out['kl_loss'].item():.4f} grad_sum={grad:.2f}")
        assert torch.isfinite(loss), "loss is NaN/Inf"

    model.eval()
    pred = model.predict_action([make_sample(2)])["normalized_actions"]
    print(f"[mock] predict_action shape = {pred.shape}  (expect (1, {model.action_horizon}, 7))")
    assert pred.shape == (1, model.action_horizon, 7), "unexpected action shape"
    print("[mock] DPVLA mock-up PASSED ✅")
