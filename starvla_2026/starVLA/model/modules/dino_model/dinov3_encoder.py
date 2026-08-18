import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor

from transformers import DINOv3ViTConfig, DINOv3ViTModel


def resize_with_pad(img: Tensor, height: int, width: int, pad_value: float = 0.0) -> Tensor:
    if img.ndim != 4:
        raise ValueError(f"(B,C,H,W) expected, got {tuple(img.shape)}")
    cur_h, cur_w = img.shape[2:]
    if cur_h == height and cur_w == width:
        return img
    ratio = max(cur_w / width, cur_h / height)
    new_h = int(round(cur_h / ratio))
    new_w = int(round(cur_w / ratio))
    resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = max(0, height - new_h)
    pad_w = max(0, width - new_w)
    if pad_h == 0 and pad_w == 0:
        return resized
    pad_w0, pad_h0 = pad_w // 2, pad_h // 2
    return F.pad(resized, (pad_w0, pad_w - pad_w0, pad_h0, pad_h - pad_h0), value=pad_value)


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        pretrained: bool = True,
        frozen: bool = True,
        image_size: int = 224,
        resize: bool = True,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        patch_size: int = 16,
        num_register_tokens: int = 4,
    ) -> None:
        super().__init__()
        if pretrained:
            self.body = DINOv3ViTModel.from_pretrained(model_name)
        else:
            cfg = DINOv3ViTConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                patch_size=patch_size,
                image_size=image_size,
                num_register_tokens=num_register_tokens,
            )
            self.body = DINOv3ViTModel(cfg)

        self.num_channels = self.body.config.hidden_size
        self.skip_tokens = 1 + int(getattr(self.body.config, "num_register_tokens", 0))
        ps = int(self.body.config.patch_size)
        self.patch_size = ps
        if isinstance(image_size, (list, tuple)):
            self.img_h, self.img_w = int(image_size[0]), int(image_size[1])
        else:
            self.img_h = self.img_w = int(image_size)
        self.image_size = image_size
        self.resize = bool(resize)
        self.patches_per_view = (self.img_h // ps) * (self.img_w // ps)

        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.frozen = frozen
        if frozen:
            self.body.eval()
            for p in self.body.parameters():
                p.requires_grad = False

    def _prepare(self, batch_images):
        B = len(batch_images)
        V = len(batch_images[0])
        flat = []
        for views in batch_images:
            for im in views:
                t = to_tensor(im).unsqueeze(0)
                if self.resize:
                    t = resize_with_pad(t, self.img_h, self.img_w)
                else:
                    h, w = t.shape[-2], t.shape[-1]
                    if h % self.patch_size or w % self.patch_size:
                        raise ValueError(
                            f"resize=False: 입력 {h}x{w}가 patch_size({self.patch_size}) "
                            f"배수가 아님 — native 입력을 patch_size 배수로 맞추세요")
                flat.append(t.squeeze(0))
        return torch.stack(flat, dim=0), B, V

    def forward(self, batch_images) -> Tensor:
        x, B, V = self._prepare(batch_images)
        dev = next(self.body.parameters()).device
        dtype = next(self.body.parameters()).dtype
        x = x.to(device=dev, dtype=dtype)
        x = (x - self.dino_mean.to(dtype)) / self.dino_std.to(dtype)

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.body(pixel_values=x)
        feats = out.last_hidden_state[:, self.skip_tokens :, :]
        return feats.reshape(B, V * feats.shape[1], feats.shape[2])


def get_dinov3_model(config=None, **kwargs):
    if config is None:
        return DINOv3Encoder(**kwargs)
    return DINOv3Encoder(
        model_name=str(config.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
        pretrained=bool(config.get("pretrained", True)),
        frozen=bool(config.get("frozen", True)),
        image_size=_parse_image_size(config.get("image_size", 224)),
        resize=bool(config.get("resize", True)),
    )


def _parse_image_size(v):
    if hasattr(v, "__len__") and not isinstance(v, str):
        return [int(v[0]), int(v[1])]
    return int(v)
