from typing import List, Optional, Tuple

import torch
from torch import nn

def find_subsequence(row: torch.Tensor, pattern: List[int], start: int = 0) -> int:
    n, m = row.shape[0], len(pattern)
    if m == 0 or m > n:
        return -1
    pat = torch.tensor(pattern, dtype=row.dtype, device=row.device)
    windows = row[start:].unfold(0, m, 1)
    hits = (windows == pat).all(dim=1).nonzero().flatten()
    return int(hits[0].item()) + start if len(hits) else -1


def build_typed_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    marker_id_variants: List[List[int]],
    boundary_token_id: int,
    num_views: int,
) -> dict:
    B, L = input_ids.shape
    attn = attention_mask.to(dtype=torch.bool, device=input_ids.device)

    vision_mask = input_ids.eq(int(image_token_id)) & attn

    task_mask = torch.zeros_like(vision_mask)
    for b in range(B):
        row = input_ids[b]
        start = end_of_marker = -1
        for pat in marker_id_variants:                  
            pos = find_subsequence(row, pat)
            if pos >= 0:
                start, end_of_marker = pos, pos + len(pat)
                break
        if start < 0:
            raise RuntimeError(
                "[typed_qformer] instruction marker not found in sample "
                f"{b} — prompt/marker mismatch (variants tried: {len(marker_id_variants)})")
        bpos = (row[end_of_marker:] == int(boundary_token_id)).nonzero().flatten()
        end = int(bpos[0].item()) + end_of_marker if len(bpos) else int(attn[b].sum().item())
        if end <= end_of_marker:
            raise RuntimeError(f"[typed_qformer] empty instruction span in sample {b}")
        task_mask[b, end_of_marker:end] = True
    task_mask &= attn

    view_ids = torch.full_like(input_ids, -1)
    for b in range(B):
        v = vision_mask[b]
        run = -1
        prev = False
        for i in range(L):
            cur = bool(v[i])
            if cur and not prev:
                run += 1
            if cur:
                view_ids[b, i] = min(run, num_views - 1)
            prev = cur

    fusion_mask = vision_mask | task_mask

    n_vis = vision_mask.sum(dim=1)
    n_task = task_mask.sum(dim=1)
    if (n_vis == 0).any() or (n_task == 0).any():
        raise RuntimeError(
            f"[typed_qformer] empty mask — vision per-sample {n_vis.tolist()}, "
            f"task per-sample {n_task.tolist()} (image_token_id={image_token_id})")

    return {
        "task_mask": task_mask,
        "vision_mask": vision_mask,
        "fusion_mask": fusion_mask,
        "view_ids": view_ids,
    }


def select_and_pad(
    kv: torch.Tensor, select_mask: torch.Tensor, view_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    B, L, D = kv.shape
    counts = select_mask.sum(dim=1)                          
    if (counts == 0).any():
        raise RuntimeError(f"[typed_qformer] select_and_pad got an empty selection: {counts.tolist()}")
    S = int(counts.max().item())

    selected = kv.new_zeros(B, S, D)
    key_padding_mask = torch.ones(B, S, dtype=torch.bool, device=kv.device)
    sel_views = (torch.zeros(B, S, dtype=torch.long, device=kv.device)
                 if view_ids is not None else None)
    for b in range(B):
        idx = select_mask[b].nonzero().flatten()
        n = idx.shape[0]
        selected[b, :n] = kv[b, idx]
        key_padding_mask[b, :n] = False
        if sel_views is not None:
            sel_views[b, :n] = view_ids[b, idx].clamp(min=0)
    return selected, key_padding_mask, sel_views

class _TypedResamplerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.drop = nn.Dropout(dropout)

    def _cross(self, q, kv, key_padding_mask):
        h = self.cross_norm(q)
        out, _ = self.cross_attn(h, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return q + self.drop(out)

    def forward(self, task_q, vision_q, fusion_q,
                task_kv, vision_kv, fusion_kv,
                task_pad, vision_pad, fusion_pad):
        task_q = self._cross(task_q, task_kv, task_pad)
        vision_q = self._cross(vision_q, vision_kv, vision_pad)
        fusion_q = self._cross(fusion_q, fusion_kv, fusion_pad)

        q = torch.cat([task_q, vision_q, fusion_q], dim=1)
        h = self.self_norm(q)
        out, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + self.drop(out)
        q = q + self.drop(self.ffn(self.ffn_norm(q)))

        nt, nv = task_q.shape[1], vision_q.shape[1]
        return q[:, :nt], q[:, nt:nt + nv], q[:, nt + nv:]


class TypedLatentQFormer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int = 512,
                 num_task_queries: int = 2, num_vision_queries: int = 4,
                 num_fusion_queries: int = 2, num_layers: int = 2,
                 num_heads: int = 8, num_views: int = 2, dropout: float = 0.1,
                 use_view_embedding: bool = True):
        super().__init__()
        self.num_task_queries = int(num_task_queries)
        self.num_vision_queries = int(num_vision_queries)
        self.num_fusion_queries = int(num_fusion_queries)
        self.num_queries = self.num_task_queries + self.num_vision_queries + self.num_fusion_queries
        self.out_dim = int(out_dim)
        self.num_views = int(num_views)
        self.use_view_embedding = bool(use_view_embedding)

        self.task_queries = nn.Parameter(torch.randn(self.num_task_queries, out_dim) * 0.02)
        self.vision_queries = nn.Parameter(torch.randn(self.num_vision_queries, out_dim) * 0.02)
        self.fusion_queries = nn.Parameter(torch.randn(self.num_fusion_queries, out_dim) * 0.02)
        self.query_type_embedding = nn.Parameter(torch.zeros(3, out_dim))
        if self.use_view_embedding:
            self.view_embedding = nn.Embedding(self.num_views, out_dim)

        self.kv_proj = nn.Linear(in_dim, out_dim)
        self.kv_norm = nn.LayerNorm(out_dim)
        self.blocks = nn.ModuleList([
            _TypedResamplerBlock(out_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, hidden, attention_mask, task_mask, vision_mask, fusion_mask,
                view_ids=None):
        B = hidden.shape[0]
        dev = hidden.device
        attn = attention_mask.to(device=dev, dtype=torch.bool)
        task_mask = task_mask.to(device=dev, dtype=torch.bool) & attn
        vision_mask = vision_mask.to(device=dev, dtype=torch.bool) & attn
        fusion_mask = fusion_mask.to(device=dev, dtype=torch.bool) & attn

        kv = self.kv_norm(self.kv_proj(hidden))

        task_kv, task_pad, _ = select_and_pad(kv, task_mask)
        vids = view_ids.to(dev) if (view_ids is not None and self.use_view_embedding) else None
        vision_kv, vision_pad, sel_views = select_and_pad(kv, vision_mask, vids)
        fusion_kv, fusion_pad, _ = select_and_pad(kv, fusion_mask)
        if sel_views is not None:
            vision_kv = vision_kv + self.view_embedding(sel_views.clamp(0, self.num_views - 1))

        task_q = self.task_queries.unsqueeze(0).expand(B, -1, -1) + self.query_type_embedding[0]
        vision_q = self.vision_queries.unsqueeze(0).expand(B, -1, -1) + self.query_type_embedding[1]
        fusion_q = self.fusion_queries.unsqueeze(0).expand(B, -1, -1) + self.query_type_embedding[2]

        for blk in self.blocks:
            task_q, vision_q, fusion_q = blk(
                task_q, vision_q, fusion_q,
                task_kv, vision_kv, fusion_kv,
                task_pad, vision_pad, fusion_pad)

        latent = torch.cat([task_q, vision_q, fusion_q], dim=1)
        return self.out_norm(latent)
