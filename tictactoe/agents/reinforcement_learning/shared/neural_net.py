"""PyTorch neural network implementations for RL agents.

Provides three large PolicyValueNetwork variants for AlphaZero and related
agents, plus QNetwork for DQN.  All policy-value networks share the same
public API: forward(), train_step(), train_batch(), copy(), save(), load().

Variants
--------
PolicyValueNetwork          : float32 conv residual (10 blocks, 256 filters, ~11.8M params)
QuantizedPolicyValueNetwork : int8 conv residual    (10 blocks, 256 filters, ~11.8M params)
BitNetPolicyValueNetwork    : BitNet 1.58-bit Transformer (15 layers, d=256, ~11.8M params)

Board encoding functions produce 3-channel representations; conv networks
internally append a 4th k/n constant plane so the trunk can condition on the
winning-run-length.

Dependency chain position: types → state → board → game → agents → benchmark.

How to add a new network
------------------------
1. Subclass nn.Module — implement forward(), copy(), save(), load().
2. Implement train_step(x, target_policy, target_value) -> float.
3. Implement train_batch(examples, lr=None) -> float.
4. Register in AlphaZeroAgent.__init__ with a new network_type string.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tictactoe.core.types import Board2D, Cell, Player

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Activation functions and losses (torch-native)
# ---------------------------------------------------------------------------

def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit."""
    return F.relu(x)


def softmax(x: Tensor) -> Tensor:
    """Numerically stable softmax along last dimension."""
    return F.softmax(x, dim=-1)


def tanh_act(x: Tensor) -> Tensor:
    """Hyperbolic tangent activation."""
    return torch.tanh(x)


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss between logits and soft probability targets."""
    return -(targets * F.log_softmax(logits, dim=-1)).sum()


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(predictions, targets)


def squared_relu(x: Tensor) -> Tensor:
    """Squared ReLU activation: ReLU(x)^2 (used in BitNet FFN sublayers)."""
    return F.relu(x).pow(2)


# ---------------------------------------------------------------------------
# BitNet 1.58-bit quantization helpers
# ---------------------------------------------------------------------------

def absmean_quantize_weights(w_latent: Tensor) -> tuple[Tensor, float]:
    """Absmean ternary weight quantization (W1.58A8 — weight side).

    Returns (w_ternary_float32, alpha) where w_ternary uses values in
    {-alpha, 0, +alpha} — effectively ternary scaled by alpha.
    """
    alpha = float(w_latent.abs().mean()) + 1e-8
    w_ternary = (w_latent / alpha).round().clamp(-1.0, 1.0)
    return w_ternary * alpha, alpha


def absmax_quantize_activations(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-token absmax 8-bit activation quantization (W1.58A8 — activation side).

    Args:
        x: 2-D float32 tensor of shape (T, d) — one row per token.

    Returns:
        (x_q, gamma) where x_q is float32 (dequantized), gamma is (T, 1).
    """
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_q = (x / gamma * 127.0).round() / 127.0 * gamma
    return x_q.float(), gamma


def rms_norm_bitnet(x: Tensor) -> Tensor:
    """RMS normalisation without learnable parameters (SubLN in BitNet).

    Normalises along the last axis.
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    return (x / rms).float()


# ---------------------------------------------------------------------------
# STE autograd Functions for quantization-aware training
# ---------------------------------------------------------------------------

class _AbsmeanTernaryWeightFn(torch.autograd.Function):
    """W1.58 absmean ternary weight quantization with Straight-Through Estimator."""

    @staticmethod
    def forward(ctx, w_latent: Tensor) -> Tensor:
        alpha = w_latent.abs().mean() + 1e-8
        w_q = (w_latent / alpha).round().clamp(-1.0, 1.0)
        ctx.save_for_backward(w_latent)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output  # STE: gradient passes through unchanged


class _AbsmaxInt8ActivationFn(torch.autograd.Function):
    """Per-token absmax 8-bit activation quantization with STE."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return ((x / gamma * 127.0).round() / 127.0 * gamma).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output  # STE


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) helpers
# ---------------------------------------------------------------------------

def rope_frequencies(d_head: int, max_seq: int = 512) -> tuple[Tensor, Tensor]:
    """Precompute interleaved cos/sin tables for RoPE.

    Returns:
        (cos_table, sin_table), each of shape (max_seq, d_head) on DEVICE.
    """
    half = d_head // 2
    theta = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(pos, theta)                   # (max_seq, half)
    cos_table = torch.cos(freqs).repeat_interleave(2, dim=-1).to(DEVICE)
    sin_table = torch.sin(freqs).repeat_interleave(2, dim=-1).to(DEVICE)
    return cos_table, sin_table


def rope_rotate_half(x: Tensor) -> Tensor:
    """Rotate interleaved pairs: (x0, x1) → (-x1, x0)."""
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    return torch.stack([-x_odd, x_even], dim=-1).reshape(x.shape)


def rope_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: Query tensor, shape (T, n_heads, d_head).
        k: Key tensor, shape (T, n_heads, d_head).
        cos, sin: Tables from rope_frequencies, shape (max_seq, d_head).
    """
    T = q.shape[0]
    c = cos[:T].unsqueeze(1)   # (T, 1, d_head)
    s = sin[:T].unsqueeze(1)
    q_rot = (q * c + rope_rotate_half(q) * s).float()
    k_rot = (k * c + rope_rotate_half(k) * s).float()
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board: Board2D, current_player: Player, n: int) -> Tensor:
    """Encode board as a (3, n, n) float32 tensor on DEVICE.

    Channel 0: 1.0 where current_player has a piece.
    Channel 1: 1.0 where the opponent has a piece.
    Channel 2: all 1.0 (constant plane marking board size).
    """
    own_cell = current_player.to_cell()
    opp_cell = current_player.opponent().to_cell()
    arr = torch.zeros(3, n, n, dtype=torch.float32, device=DEVICE)
    arr[2, :, :] = 1.0
    for r in range(n):
        for c in range(n):
            cell = board[r][c]
            if cell is own_cell:
                arr[0, r, c] = 1.0
            elif cell is opp_cell:
                arr[1, r, c] = 1.0
    return arr


def encode_board_flat(board: Board2D, current_player: Player, n: int) -> Tensor:
    """Encode board as a flat (3*n*n,) float32 tensor on DEVICE."""
    return encode_board(board, current_player, n).flatten()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, device=None) -> Tensor:
    """Convert x to a float32 Tensor on the target device."""
    dev = device or DEVICE
    if isinstance(x, Tensor):
        return x.to(dtype=torch.float32, device=dev)
    return torch.tensor(x, dtype=torch.float32, device=dev)


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def _he_init(module: nn.Module) -> None:
    """Apply Kaiming normal initialisation to Conv2d and Linear layers."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# PolicyValueNetwork: AlphaZero-style convolutional residual network (float32)
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """One residual block: Conv→BN-free→ReLU→Conv, then add skip→ReLU."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class PolicyValueNetwork(nn.Module):
    """AlphaZero-inspired policy-value network with conv residual blocks.

    Architecture:
        Input:   3*n*n flat vector → reshaped to (4, n, n) with k/n plane
        Initial: Conv(4→256, 3×3, same) → ReLU
        Tower:   10 × ResBlock(256 filters)
        Policy:  Conv(256→2, 1×1) → ReLU → flatten(2n²) → Linear(2n²→n²)
        Value:   Conv(256→1, 1×1) → ReLU → flatten(n²) → Linear(n²→256) → ReLU
                 → Linear(256→1) → tanh

    Weight precision: float32 throughout (~11.8M parameters).
    Training:        Adam on all parameters.
    """

    NUM_RES_BLOCKS: int = 10
    FILTERS: int = 256

    def __init__(self, n: int, k: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.k = k if k is not None else n
        C = self.FILTERS

        self.conv_init = nn.Conv2d(4, C, 3, padding=1)
        self.res_blocks = nn.ModuleList([_ResBlock(C) for _ in range(self.NUM_RES_BLOCKS)])

        # Policy head
        self.conv_p = nn.Conv2d(C, 2, 1)
        self.fc_p   = nn.Linear(2 * n * n, n * n)

        # Value head
        self.conv_v = nn.Conv2d(C, 1, 1)
        self.fc_v1  = nn.Linear(n * n, 256)
        self.fc_v2  = nn.Linear(256, 1)

        # He initialisation
        _he_init(self)

        # Optimizer: all parameters
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(DEVICE)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_4channel(self, x: Tensor) -> Tensor:
        """Reshape flat (3n²,) → (1, 4, n, n) with k/n plane appended."""
        spatial = x.reshape(1, 3, self.n, self.n)
        k_plane = torch.full((1, 1, self.n, self.n), self.k / self.n, device=x.device)
        return torch.cat([spatial, k_plane], dim=1)

    def _forward_trunk(self, feat: Tensor) -> Tensor:
        feat = F.relu(self.conv_init(feat))
        for blk in self.res_blocks:
            feat = blk(feat)
        return feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Flat board encoding of length 3*n*n (torch.Tensor on DEVICE).

        Returns:
            (policy_logits, value) both as tensors. policy_logits has length
            n*n; value is a scalar tensor in [-1, 1].
        """
        feat = self._forward_trunk(self._to_4channel(x))

        # Policy head
        p_h = F.relu(self.conv_p(feat))        # (1, 2, n, n)
        flat_p = p_h.reshape(1, -1)            # (1, 2n²)
        policy_logits = self.fc_p(flat_p).squeeze(0)   # (n²,)

        # Value head
        v_h = F.relu(self.conv_v(feat))        # (1, 1, n, n)
        flat_v = v_h.reshape(1, -1)            # (1, n²)
        h_v = F.relu(self.fc_v1(flat_v))       # (1, 256)
        value = torch.tanh(self.fc_v2(h_v)).squeeze()  # scalar tensor

        return policy_logits, value

    def train_step(
        self,
        x: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
    ) -> float:
        """One supervised update step on a single example.

        Args:
            x: Flat encoded state (torch.Tensor of length 3*n*n).
            target_policy: Target policy distribution (sums to 1, length n*n).
            target_value: Target value scalar in [-1, 1].

        Returns:
            Combined loss (policy cross-entropy + value MSE).
        """
        return self.train_batch([(x, target_policy, target_value)])

    def train_batch(
        self,
        examples: list[tuple],
        lr: float | None = None,
    ) -> float:
        """Mini-batch training step.

        Args:
            examples: List of (x, target_policy, target_value) tuples.
            lr: Learning rate override (optional).

        Returns:
            Mean combined loss over the batch.
        """
        if not examples:
            raise ValueError("examples must be non-empty.")
        if lr is not None:
            _set_lr(self._optimizer, lr)
        self._optimizer.zero_grad()
        total_loss = 0.0
        n = len(examples)
        for x, tp, tv in examples:
            x  = _to_tensor(x)
            tp = _to_tensor(tp)
            tv = _to_tensor(tv).reshape(1)
            logits, val = self.forward(x)
            p_loss = cross_entropy_loss(logits, tp)
            v_loss = mse_loss(val.reshape(1), tv)
            loss = (p_loss + v_loss) / n
            loss.backward()
            total_loss += loss.item()
        self._optimizer.step()
        return total_loss

    def copy(self) -> "PolicyValueNetwork":
        return copy.deepcopy(self)

    def save(self, path: str) -> None:
        """Save weights to a .pt file (path without extension)."""
        torch.save(self.state_dict(), path + '.pt')

    def load(self, path: str) -> None:
        """Load weights from a .pt file."""
        pt = path if path.endswith('.pt') else path + '.pt'
        self.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))


# ---------------------------------------------------------------------------
# QuantizedPolicyValueNetwork: Deep conv int8 network (10 blocks, 256 filters)
# ---------------------------------------------------------------------------

class QuantizedPolicyValueNetwork(nn.Module):
    """AlphaZero-style policy-value network with int8 quantized weights.

    Identical architecture to PolicyValueNetwork (same layer counts and widths);
    only difference is precision: all conv and FC weights are int8 and frozen,
    and only the output head biases are trained.

    Architecture:
        Input:   3*n*n flat vector → reshaped to (4, n, n) with k/n plane
        Initial: Conv(4→256, 3×3, int8) → ReLU
        Tower:   10 × ResBlock(256 filters, int8)
        Policy:  Conv(256→2, 1×1, int8) → ReLU → flatten(2n²) → Linear(2n²→n², int8)
        Value:   Conv(256→1, 1×1, int8) → ReLU → flatten(n²) → Linear(n²→256, int8)
                 → ReLU → Linear(256→1, int8) → tanh

    Weight precision: int8 (~11.8M parameters, ~11.8 MB).
    Training:        Adam on output biases bp (n²,) and bv2 (1,) only.
    """

    NUM_RES_BLOCKS: int = 10
    FILTERS: int = 256

    def __init__(self, n: int, k: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.k = k if k is not None else n
        self._build()

    @staticmethod
    def _quantize(w_fp32: Tensor) -> tuple[Tensor, Tensor]:
        abs_max = w_fp32.abs().max().clamp(min=1e-8)
        scale = abs_max / 127.0
        return (w_fp32 / scale).round().to(torch.int8), scale.reshape(1)

    def _build(self) -> None:
        gen = torch.Generator().manual_seed(0)
        n, C, N = self.n, self.FILTERS, self.NUM_RES_BLOCKS

        def _he_conv_q(c_in: int, c_out: int, kH: int = 3, kW: int = 3):
            std = math.sqrt(2.0 / (c_in * kH * kW))
            w = torch.empty(c_out, c_in, kH, kW).normal_(0.0, std, generator=gen)
            return self._quantize(w)

        def _he_fc_q(fan_in: int, fan_out: int):
            std = math.sqrt(2.0 / fan_in)
            w = torch.empty(fan_in, fan_out).normal_(0.0, std, generator=gen)
            return self._quantize(w)

        # Initial conv
        w_init, s_init = _he_conv_q(4, C)
        self.register_buffer('conv_init_w', w_init)
        self.register_buffer('conv_init_s', s_init)
        self.register_buffer('conv_init_b', torch.zeros(C))

        # Residual blocks (batched buffers)
        w1_list, s1_list, w2_list, s2_list = [], [], [], []
        for _ in range(N):
            w, s = _he_conv_q(C, C); w1_list.append(w); s1_list.append(s)
            w, s = _he_conv_q(C, C); w2_list.append(w); s2_list.append(s)
        self.register_buffer('res_w1', torch.stack(w1_list))   # (N, C, C, 3, 3) int8
        self.register_buffer('res_w2', torch.stack(w2_list))
        self.register_buffer('res_s1', torch.cat(s1_list))     # (N,) float32
        self.register_buffer('res_s2', torch.cat(s2_list))
        self.register_buffer('res_b1', torch.zeros(N, C))
        self.register_buffer('res_b2', torch.zeros(N, C))

        # Policy head conv
        w_cp, s_cp = _he_conv_q(C, 2, 1, 1)
        self.register_buffer('conv_p_w', w_cp)
        self.register_buffer('conv_p_s', s_cp)
        self.register_buffer('conv_p_b', torch.zeros(2))
        # Policy FC
        w_fp, s_fp = _he_fc_q(2 * n * n, n * n)
        self.register_buffer('fc_p_w', w_fp)
        self.register_buffer('fc_p_s', s_fp)
        # Trainable policy bias
        self.bp = nn.Parameter(torch.zeros(n * n))

        # Value head conv
        w_cv, s_cv = _he_conv_q(C, 1, 1, 1)
        self.register_buffer('conv_v_w', w_cv)
        self.register_buffer('conv_v_s', s_cv)
        self.register_buffer('conv_v_b', torch.zeros(1))
        # Value FC1
        w_fv1, s_fv1 = _he_fc_q(n * n, 256)
        self.register_buffer('fc_v1_w', w_fv1)
        self.register_buffer('fc_v1_s', s_fv1)
        self.register_buffer('bv1', torch.zeros(256))
        # Value FC2
        w_fv2, s_fv2 = _he_fc_q(256, 1)
        self.register_buffer('fc_v2_w', w_fv2)
        self.register_buffer('fc_v2_s', s_fv2)
        # Trainable value bias
        self.bv2 = nn.Parameter(torch.zeros(1))

        self._optimizer = torch.optim.Adam([self.bp, self.bv2], lr=1e-3)
        self.to(DEVICE)

    def _dq(self, w: Tensor, s: Tensor) -> Tensor:
        return w.float() * s.item()

    def _to_4channel(self, x: Tensor) -> Tensor:
        spatial = x.reshape(1, 3, self.n, self.n)
        k_plane = torch.full((1, 1, self.n, self.n), self.k / self.n, device=x.device)
        return torch.cat([spatial, k_plane], dim=1)

    def _forward_trunk(self, feat: Tensor) -> Tensor:
        feat = F.relu(F.conv2d(feat, self._dq(self.conv_init_w, self.conv_init_s),
                               self.conv_init_b, padding=1))
        N = self.NUM_RES_BLOCKS
        for i in range(N):
            skip = feat
            w1 = self._dq(self.res_w1[i], self.res_s1[i:i+1])
            w2 = self._dq(self.res_w2[i], self.res_s2[i:i+1])
            b1 = self.res_b1[i]
            b2 = self.res_b2[i]
            out = F.relu(F.conv2d(feat, w1, b1, padding=1))
            out = F.conv2d(out, w2, b2, padding=1)
            feat = F.relu(out + skip)
        return feat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self._forward_trunk(self._to_4channel(x))

        # Policy head
        p_h = F.relu(F.conv2d(feat, self._dq(self.conv_p_w, self.conv_p_s),
                               self.conv_p_b))
        flat_p = p_h.reshape(1, -1)
        policy_logits = (flat_p @ self._dq(self.fc_p_w, self.fc_p_s) + self.bp).squeeze(0)

        # Value head
        v_h = F.relu(F.conv2d(feat, self._dq(self.conv_v_w, self.conv_v_s),
                               self.conv_v_b))
        flat_v = v_h.reshape(1, -1)
        h_v = F.relu(flat_v @ self._dq(self.fc_v1_w, self.fc_v1_s) + self.bv1)
        value = torch.tanh(h_v @ self._dq(self.fc_v2_w, self.fc_v2_s) + self.bv2).squeeze()

        return policy_logits, value

    def train_step(self, x: Tensor, target_policy: Tensor, target_value: Tensor) -> float:
        return self.train_batch([(x, target_policy, target_value)])

    def train_batch(self, examples: list[tuple], lr: float | None = None) -> float:
        if not examples:
            raise ValueError("examples must be non-empty.")
        if lr is not None:
            _set_lr(self._optimizer, lr)
        self._optimizer.zero_grad()
        total_loss = 0.0
        n = len(examples)
        for x, tp, tv in examples:
            x  = _to_tensor(x)
            tp = _to_tensor(tp)
            tv = _to_tensor(tv).reshape(1)
            logits, val = self.forward(x)
            loss = (cross_entropy_loss(logits, tp) + mse_loss(val.reshape(1), tv)) / n
            loss.backward()
            total_loss += loss.item()
        self._optimizer.step()
        return total_loss

    def copy(self) -> "QuantizedPolicyValueNetwork":
        return copy.deepcopy(self)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + '.pt')

    def load(self, path: str) -> None:
        pt = path if path.endswith('.pt') else path + '.pt'
        self.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))

    @property
    def memory_usage_mb(self) -> float:
        int8_bytes = sum(b.numel() for b in [
            self.conv_init_w,
            self.res_w1, self.res_w2,
            self.conv_p_w, self.fc_p_w,
            self.conv_v_w, self.fc_v1_w, self.fc_v2_w,
        ])
        float32_bytes = sum(p.numel() * 4 for p in [
            self.conv_init_b, self.res_b1, self.res_b2,
            self.conv_p_b, self.bp,
            self.conv_v_b, self.bv1, self.bv2,
        ])
        return (int8_bytes + float32_bytes) / (1024 * 1024)


# ---------------------------------------------------------------------------
# QNetwork: action-value network for DQN
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Q-network: maps board state to action-values for each cell.

    Architecture:
        Input:  3*n*n + 1 flat vector (k/n appended as extra scalar)
        Hidden: Linear(3n²+1, 128) → ReLU → Linear(128, 64) → ReLU
        Output: Linear(64, n²)  (Q-values for each cell)

    All weights trainable via Adam.
    """

    def __init__(self, n: int, k: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.k = k if k is not None else n
        input_size = 3 * n * n + 1
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n * n)
        _he_init(self)
        self.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Flat input vector of length 3*n*n (torch.Tensor on DEVICE).

        Returns:
            Q-value tensor of length n*n.
        """
        k_s = torch.tensor([self.k / self.n], dtype=torch.float32, device=x.device)
        x = torch.cat([x, k_s])
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def copy(self) -> "QNetwork":
        """Return a deep copy (same weights, fresh optimizer state)."""
        net = QNetwork(self.n, self.k)
        net.load_state_dict(self.state_dict())
        return net

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + '.pt')

    def load(self, path: str) -> None:
        pt = path if path.endswith('.pt') else path + '.pt'
        self.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))


# ---------------------------------------------------------------------------
# BitLinear: 1.58-bit linear layer using STE (no bias per BitNet spec)
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """BitNet 1.58-bit linear layer with STE-based training.

    Forward pass:
        1. Absmean ternary weight quantization via STE.
        2. Per-token absmax int8 activation quantization via STE.
        3. Output = x_q @ W_eff.T

    Gradients flow through both quantization operations unchanged (STE).
    Latent float32 weights are `nn.Parameter` and updated via the enclosing
    network's optimizer.
    """

    def __init__(self, fan_in: int, fan_out: int) -> None:
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in).normal_(0.0, std))

    def forward(self, x: Tensor) -> Tensor:
        """Quantized forward pass.

        Args:
            x: Input of shape (..., fan_in).

        Returns:
            Output of shape (..., fan_out).
        """
        w_eff = _AbsmeanTernaryWeightFn.apply(self.weight)  # (fan_out, fan_in)
        original_shape = x.shape
        x2d = x.reshape(-1, original_shape[-1])
        x_q = _AbsmaxInt8ActivationFn.apply(x2d)
        out = x_q @ w_eff.T
        return out.reshape(*original_shape[:-1], self.weight.shape[0]).float()


# ---------------------------------------------------------------------------
# BitMultiHeadAttention: full MHA with BitLinear projections and RoPE
# ---------------------------------------------------------------------------

class BitMultiHeadAttention(nn.Module):
    """Multi-head attention using BitLinear projections and RoPE.

    SubLN is applied by the enclosing BitTransformerBlock before input.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = BitLinear(d_model, d_model)
        self.wk = BitLinear(d_model, d_model)
        self.wv = BitLinear(d_model, d_model)
        self.wo = BitLinear(d_model, d_model)
        cos, sin = rope_frequencies(self.d_head)
        self.register_buffer('cos_table', cos)
        self.register_buffer('sin_table', sin)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (T, d_model)

        Returns:
            (T, d_model) attention output.
        """
        T, d = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.wq(x).reshape(T, H, dh)
        k = self.wk(x).reshape(T, H, dh)
        v = self.wv(x).reshape(T, H, dh)

        q_rot, k_rot = rope_apply(q, k, self.cos_table, self.sin_table)

        scale = 1.0 / math.sqrt(dh)
        scores = torch.einsum("thd,shd->hts", q_rot, k_rot) * scale
        attn = F.softmax(scores, dim=-1)   # (H, T, T)

        context = torch.einsum("hts,shd->thd", attn, v).reshape(T, d)
        return self.wo(context).float()


# ---------------------------------------------------------------------------
# BitFFN: feed-forward sublayer with squared ReLU
# ---------------------------------------------------------------------------

class BitFFN(nn.Module):
    """BitNet FFN: BitLinear(up) → squared_relu → BitLinear(down). No bias."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up   = BitLinear(d_model, d_ff)
        self.down = BitLinear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(squared_relu(self.up(x))).float()


# ---------------------------------------------------------------------------
# BitTransformerBlock: SubLN + MHA + SubLN + FFN with residual connections
# ---------------------------------------------------------------------------

class BitTransformerBlock(nn.Module):
    """One BitNet Transformer block (SubLN style — RMSNorm before each sublayer).

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = BitMultiHeadAttention(d_model, n_heads)
        self.ffn  = BitFFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(rms_norm_bitnet(x))
        x = x + self.ffn(rms_norm_bitnet(x))
        return x.float()


# ---------------------------------------------------------------------------
# BitNetPolicyValueNetwork: BitNet 1.58-bit Transformer (15 layers, d=256)
# ---------------------------------------------------------------------------

class BitNetPolicyValueNetwork(nn.Module):
    """BitNet 1.58-bit Transformer policy-value network (~11.8M parameters).

    Treats the board as a sequence of n² position tokens, each described by
    4 features (own pieces, opponent pieces, constant 1.0 plane, k/n scalar).
    A BitLinear embedding projects to d_model dimensions, then passes through
    15 BitTransformerBlocks.

    Architecture:
        Input:       3*n*n flat → reshape (n², 3) → append k/n → (n², 4)
        Embedding:   BitLinear(4, 256)
        15 × BitTransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        Final RMSNorm
        Policy head: BitLinear(256, 1) per token → (n²,) logits
        Value head:  mean-pool → BitLinear(256, 1) → tanh → scalar

    Weight precision: W1.58A8 — ternary weights {-1,0,+1}, 8-bit activations.
    Training:        STE-based Adam on all BitLinear latent float32 weights.
    """

    _N_LAYERS: int = 15
    _D_MODEL:  int = 256
    _N_HEADS:  int = 8
    _D_FF:     int = 1024

    def __init__(self, n: int, k: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.k = k if k is not None else n
        d = self._D_MODEL
        self._embed       = BitLinear(4, d)
        self._blocks      = nn.ModuleList([
            BitTransformerBlock(d, self._N_HEADS, self._D_FF)
            for _ in range(self._N_LAYERS)
        ])
        self._policy_head = BitLinear(d, 1)
        self._value_head  = BitLinear(d, 1)
        self._optimizer   = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(DEVICE)

    def _encode_input(self, x: Tensor) -> Tensor:
        """Convert flat (3n²,) to token matrix (n², 4)."""
        n2 = self.n * self.n
        ch = x.reshape(3, n2).T   # (n², 3)
        kn = torch.full((n2, 1), self.k / self.n, dtype=torch.float32, device=x.device)
        return torch.cat([ch, kn], dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns:
            (policy_logits, value) — policy_logits length n*n, value scalar tensor.
        """
        tokens = self._encode_input(x)         # (n², 4)
        h = self._embed(tokens)                # (n², d_model)
        for blk in self._blocks:
            h = blk(h)
        h = rms_norm_bitnet(h)                 # final SubLN

        policy_logits = self._policy_head(h).squeeze(-1)   # (n²,)
        mean_h = h.mean(dim=0, keepdim=True)               # (1, d_model)
        value = torch.tanh(self._value_head(mean_h)).squeeze()  # scalar tensor

        return policy_logits, value

    def train_step(self, x: Tensor, target_policy: Tensor, target_value: Tensor) -> float:
        return self.train_batch([(x, target_policy, target_value)])

    def train_batch(self, examples: list[tuple], lr: float | None = None) -> float:
        if not examples:
            raise ValueError("examples must be non-empty.")
        if lr is not None:
            _set_lr(self._optimizer, lr)
        self._optimizer.zero_grad()
        total_loss = 0.0
        n = len(examples)
        for x, tp, tv in examples:
            x  = _to_tensor(x)
            tp = _to_tensor(tp)
            tv = _to_tensor(tv).reshape(1)
            logits, val = self.forward(x)
            loss = (cross_entropy_loss(logits, tp) + mse_loss(val.reshape(1), tv)) / n
            loss.backward()
            total_loss += loss.item()
        self._optimizer.step()
        return total_loss

    def copy(self) -> "BitNetPolicyValueNetwork":
        return copy.deepcopy(self)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + '.pt')

    def load(self, path: str) -> None:
        pt = path if path.endswith('.pt') else path + '.pt'
        self.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))
