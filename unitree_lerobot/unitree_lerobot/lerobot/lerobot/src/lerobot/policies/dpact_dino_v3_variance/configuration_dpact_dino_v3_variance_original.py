"""
DPACT-DINOv3-Variance Configuration

dpact_dino_v3 기반 ablation 실험용 variant.
두 가지 구조적 변경을 독립적으로 on/off 가능:

  1. use_multi_layer_dino: DINOv3의 layer 4, 8, 12 features를 concat하여 사용
     - False (기본): last layer만 → Linear(768→512)
     - True: 3개 layer concat → Linear(2304→512)

  2. use_residual_action_head: Action head를 Residual MLP로 교체
     - False (기본): Linear(512→action_dim)
     - True: ResBlock(512) → ResBlock(512) → Linear(512→action_dim)

Ablation 실험 조합:
  A. multi_layer=True,  residual=False  → multi-layer only
  B. multi_layer=False, residual=True   → residual action head only
  C. multi_layer=True,  residual=True   → both
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


OBS_LATENT = "observation.latent"


@PreTrainedConfig.register_subclass("dpact_dino_v3_variance")
@dataclass
class DPACTDINOv3VarianceConfig(PreTrainedConfig):
    """Configuration for DPACT-DINOv3 with ablation options."""

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "LATENT": NormalizationMode.MEAN_STD,
        }
    )

    # -- DINOv3 Vision Backbone --
    dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    dino_hidden_size: int = 768
    dino_num_patches: int = 256
    dino_num_registers: int = 4
    dino_patch_size: int = 16

    # -- Ablation Options --
    use_multi_layer_dino: bool = False    # True: layer 4,8,12 concat → Linear(2304→512)
    use_residual_action_head: bool = False # True: ResBlock×2 + Linear
    use_film: bool = False                # True: projector 후 FiLM (기존 옵션 유지)
    use_patch_pooling: bool = False       # True: 256 patches → patch_pool_size patches

    # -- DINOv3 Backbone Training Options --
    # 기본값: 둘 다 False → DINOv3 frozen (기존 동작 유지)
    unfreeze_dinov3: bool = False         # True: DINOv3 backbone 학습 (lr=optimizer_lr_backbone)
    use_dinov3_lora: bool = False         # True: DINOv3에 LoRA adapter 적용 (PEFT 필요)
    dinov3_lora_rank: int = 8             # LoRA rank
    dinov3_lora_alpha: int = 16           # LoRA alpha
    # DINOv3 ViT attention layer 이름: q_proj, k_proj, v_proj (o_proj 제외)
    dinov3_lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )

    # -- Text Encoder Option (Qwen latent 대체) --
    # 기본값: False → Qwen latent (OBS_LATENT) 사용 (기존 동작 유지)
    # True: CLIP text encoder로 task instruction 인코딩 → latent_projector 입력
    use_text_encoder: bool = False
    text_encoder_model: str = "openai/clip-vit-base-patch32"
    text_encoder_hidden_size: int = 512   # CLIP-Base text dim
    text_encoder_max_length: int = 77     # CLIP 표준 token 길이

    # -- Multi-layer DINOv3 settings --
    dino_layer_indices: list[int] = field(default_factory=lambda: [4, 8, 12])

    # -- Residual Action Head settings --
    residual_action_head_n_blocks: int = 2

    # -- Patch Pooling settings --
    patch_pool_size: int = 64             # pooling 후 patch 수 (8×8=64)

    # -- Transformer --
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # -- VAE --
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # -- Temporal Ensemble --
    temporal_ensemble_coeff: float = 0.01

    # -- Training --
    dropout: float = 0.1
    kl_weight: float = 10.0
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 0.0

    # -- Latent (Qwen) -> Encoder Token --
    latent_mlp_hidden: list[int] = field(default_factory=lambda: [1024, 512])

    def __post_init__(self):
        super().__post_init__()
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})"
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got {self.n_obs_steps}")

    @property
    def dino_projector_input_size(self) -> int:
        """Image projector input size: 768 (single layer) or 2304 (3 layers concat)."""
        if self.use_multi_layer_dino:
            return self.dino_hidden_size * len(self.dino_layer_indices)
        return self.dino_hidden_size

    @property
    def effective_num_patches(self) -> int:
        """Positional embedding에 사용할 실제 patch 수."""
        if self.use_patch_pooling:
            return self.patch_pool_size
        return self.dino_num_patches

    @property
    def latent_projector_input_size(self) -> int:
        """Latent projector input dim:
        - use_text_encoder=False: Qwen latent dim (3584)
        - use_text_encoder=True: text encoder hidden size (512 for CLIP-Base)
        """
        if self.use_text_encoder:
            return self.text_encoder_hidden_size
        return self.latent_feature.shape[0]

    @property
    def latent_feature(self) -> PolicyFeature | None:
        if OBS_LATENT in self.input_features:
            return self.input_features[OBS_LATENT]
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("At least one image feature is required.")
        # use_text_encoder=False (default): Qwen latent (OBS_LATENT) 필수
        # use_text_encoder=True: text encoder가 task instruction에서 직접 추출 → OBS_LATENT 불필요
        if not self.use_text_encoder and not self.latent_feature:
            raise ValueError(
                f"'{OBS_LATENT}' is required in input_features when use_text_encoder=False."
            )
        if not self.latent_mlp_hidden:
            raise ValueError("latent_mlp_hidden must be set.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
