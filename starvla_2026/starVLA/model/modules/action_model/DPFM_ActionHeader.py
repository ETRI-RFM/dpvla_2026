from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.ACT_ActionHeader import make_mlp
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import ActionEncoder
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


def extract_lastk_latent(hidden, attention_mask=None, k=8):
    if hidden.shape[1] < k:
        raise ValueError(f"sequence length {hidden.shape[1]} < latent_tokens {k}")
    if attention_mask is not None:
        n_real = attention_mask.sum(dim=1)
        if bool((n_real < k).any()):
            raise ValueError(f"a sample has fewer than {k} real tokens: {n_real.tolist()}")
    return hidden[:, -k:, :]


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return ((x * rms) * self.weight).to(dtype)


def _build_latent_norm(kind, dim):
    kind = str(kind).lower()
    if kind in ("layernorm", "ln"):
        return nn.LayerNorm(dim)
    if kind in ("rmsnorm", "rms"):
        return _RMSNorm(dim)
    if kind in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"unknown latent_norm='{kind}' (layernorm|rmsnorm|none)")


class DPFMActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        cfg = full_config.framework.action_model

        self.latent_input_dim = int(cfg.latent_input_dim)
        self.dino_feat_dim = int(cfg.dino_feat_dim)
        self.patches_per_view = int(cfg.patches_per_view)
        self.num_cameras = int(cfg.num_cameras)

        self.dim_model = int(cfg.get("dim_model", 512))
        self.n_heads = int(cfg.get("n_heads", 8))
        self.n_dit_layers = int(cfg.get("n_dit_layers", 6))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.action_dim = int(cfg.action_dim)
        self.state_dim = int(cfg.get("state_dim", 0))
        self.action_horizon = int(cfg.action_horizon)

        self.num_latent_tokens = int(cfg.get("num_latent_tokens", 8))
        self.use_age_token = bool(cfg.get("use_age_token", True))
        self.age_max = int(cfg.get("age_max", 32))

        self.noise_s = float(cfg.get("noise_s", 0.999))
        self.beta_dist = Beta(float(cfg.get("noise_beta_alpha", 1.5)),
                              float(cfg.get("noise_beta_beta", 1.0)))
        self.num_timestep_buckets = int(cfg.get("num_timestep_buckets", 1000))
        self.num_inference_timesteps = int(cfg.get("num_inference_timesteps", 5))

        D = self.dim_model
        
        self.image_projector = nn.Linear(self.dino_feat_dim, D)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.patches_per_view, D))
        nn.init.normal_(self.patch_pos_embed, std=0.02)
        self.view_embed = nn.Embedding(self.num_cameras, D)
        nn.init.normal_(self.view_embed.weight, std=0.02)

        self.latent_norm = _build_latent_norm(cfg.get("latent_norm", "layernorm"),
                                              self.latent_input_dim)
        hidden_dims = list(cfg.get("latent_mlp_hidden", [1024, 512]))
        self.latent_projector = make_mlp(self.latent_input_dim, hidden_dims, D,
                                         p_drop=self.dropout)
        if self.use_age_token:
            self.age_embed = nn.Embedding(self.age_max + 1, D)
            nn.init.normal_(self.age_embed.weight, std=0.02)

        self.state_proj = nn.Linear(self.state_dim, D) if self.state_dim else None

        self.action_encoder = ActionEncoder(action_dim=self.action_dim, hidden_size=D)
        self.action_pos_embed = nn.Embedding(self.action_horizon, D)
        nn.init.normal_(self.action_pos_embed.weight, std=0.02)

        self.interleave_self_attention = bool(cfg.get("interleave_self_attention", False))
        self.dit = DiT(
            num_attention_heads=self.n_heads,
            attention_head_dim=D // self.n_heads,
            output_dim=D,
            num_layers=self.n_dit_layers,
            dropout=self.dropout,
            cross_attention_dim=D,
            num_embeds_ada_norm=self.num_timestep_buckets,
            interleave_self_attention=self.interleave_self_attention,
        )
        self.action_decoder = make_mlp(D, [D], self.action_dim, p_drop=0.0)

    @staticmethod
    def _norm_state(state):
        if state is None:
            return None
        if state.dim() == 2:
            return state.unsqueeze(1)
        return state.reshape(state.shape[0], 1, -1)

    def _age_vec(self, age, batch, device):
        if not self.use_age_token:
            return None
        if age is None:
            age = torch.zeros(batch, dtype=torch.long, device=device)
        age = age.to(device=device, dtype=torch.long).clamp_(0, self.age_max)
        return self.age_embed(age)

    def build_memory(self, latent_tokens, dino_features, state, age=None):
        B = latent_tokens.shape[0]
        D = self.dim_model
        state = self._norm_state(state)

        p = self.image_projector(dino_features)                        
        V, P = self.num_cameras, self.patches_per_view
        p = p.view(B, V, P, D) + self.patch_pos_embed.unsqueeze(0)
        p = p + self.view_embed.weight[:V].view(1, V, 1, D)
        p = p.reshape(B, V * P, D)

        z = self.latent_projector(self.latent_norm(latent_tokens))
        a = self._age_vec(age, B, z.device)
        if a is not None:
            z = z + a.unsqueeze(1)

        mem = [p, z]
        if self.state_proj is not None and state is not None:
            mem.append(self.state_proj(state))
        return torch.cat(mem, dim=1)

    def _hidden_seq(self, noised_actions, t_disc, state):
        state = self._norm_state(state)
        af = self.action_encoder(noised_actions, t_disc)
        pos = torch.arange(af.shape[1], device=af.device)
        af = af + self.action_pos_embed(pos).unsqueeze(0)
        if self.state_proj is not None and state is not None:
            sf = self.state_proj(state)
            return torch.cat([sf, af], dim=1)
        return af

    def sample_time(self, batch, device, dtype):
        s = self.noise_s
        sample = self.beta_dist.sample([batch]).to(device, dtype=dtype).clamp(max=s)
        return (s - sample) / s

    def forward(self, latent_tokens, dino_features, actions, state=None,
                action_is_pad=None, age=None):
        B, T, _ = actions.shape
        device = actions.device

        noise = torch.randn_like(actions)
        t = self.sample_time(B, device, actions.dtype)[:, None, None]
        x_t = t * actions + (1.0 - t) * noise
        velocity = actions - noise
        t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()

        memory = self.build_memory(latent_tokens, dino_features, state, age)
        h = self._hidden_seq(x_t, t_disc, state)
        out = self.dit(hidden_states=h, encoder_hidden_states=memory, timestep=t_disc)
        pred = self.action_decoder(out)[:, -T:]

        mse = (pred - velocity) ** 2
        if action_is_pad is not None:
            keep = (~action_is_pad).unsqueeze(-1).to(mse.dtype)
            loss = (mse * keep).sum() / keep.sum().clamp(min=1.0) / mse.shape[-1]
        else:
            loss = mse.mean()
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, latent_tokens, dino_features, state=None, age=None,
                       num_steps=None, noise=None):
        B = latent_tokens.shape[0]
        device = latent_tokens.device
        N = int(num_steps or self.num_inference_timesteps)
        dt = 1.0 / N

        memory = self.build_memory(latent_tokens, dino_features, state, age)
        actions = noise if noise is not None else torch.randn(
            B, self.action_horizon, self.action_dim, device=device, dtype=memory.dtype)

        for k in range(N):
            t_disc = torch.full((B,), int(k / N * self.num_timestep_buckets),
                                device=device, dtype=torch.long)
            h = self._hidden_seq(actions, t_disc, state)
            v = self.action_decoder(
                self.dit(hidden_states=h, encoder_hidden_states=memory, timestep=t_disc)
            )[:, -self.action_horizon:]
            actions = actions + dt * v
        return actions

    @torch.no_grad()
    def score_action_candidates(self, candidates, latent_tokens, dino_features,
                                state=None, age=None, t_star=0.9, n_eps=4, seed=None):
        assert latent_tokens.shape[0] == 1, "candidate scoring is per-observation (B=1)"
        C, T, A = candidates.shape
        device = candidates.device
        R = C * n_eps

        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(int(seed))
        eps = torch.randn(n_eps, C, T, A, device=device, generator=g,
                          dtype=candidates.dtype)

        cand = candidates.unsqueeze(0).expand(n_eps, C, T, A).reshape(R, T, A)
        eps = eps.reshape(R, T, A)
        x_t = t_star * cand + (1.0 - t_star) * eps
        target_v = cand - eps

        memory = self.build_memory(latent_tokens, dino_features, state, age)
        memory = memory.expand(R, -1, -1)
        state = self._norm_state(state)
        state_r = state.expand(R, -1, -1) if state is not None else None
        t_disc = torch.full((R,), int(t_star * self.num_timestep_buckets),
                            device=device, dtype=torch.long)

        h = self._hidden_seq(x_t, t_disc, state_r)
        v = self.action_decoder(
            self.dit(hidden_states=h, encoder_hidden_states=memory, timestep=t_disc)
        )[:, -T:]

        err = ((v - target_v) ** 2).mean(dim=(1, 2)).view(n_eps, C).mean(dim=0)
        return -err


def get_dpfm_action_model(full_config):
    return DPFMActionHead(full_config)
