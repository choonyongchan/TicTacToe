"""Numpy-only neural network implementations for RL agents.

Provides several PolicyValueNetwork variants (conv residual + quantized/ternary
flavours) and QNetwork (action-value). All networks expose the same public API:
forward(), backward(), backward_batch(), copy(), save(), load().

Board encoding functions produce 3-channel representations; conv networks
internally append a 4th k/n constant plane so the trunk can condition on the
winning-run-length.

Requires numpy.

Dependency chain position: types → state → board → game → agents → benchmark.

How to add a new network
------------------------
1. Subclass nothing — implement forward(), backward(), backward_batch(),
   copy(), save(), load() directly.
2. backward() should call backward_batch([(x, dp, dv)], lr).
3. Register in AlphaZeroAgent.__init__ with a new network_type string.
"""
from __future__ import annotations

import numpy as np

from tictactoe.core.types import Board2D, Cell, Player


# ---------------------------------------------------------------------------
# Activation functions and losses
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def tanh_act(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss between logits and probability targets."""
    probs = softmax(logits)
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(targets * np.log(probs)))


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((predictions - targets) ** 2))


def squared_relu(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation: ReLU(x)^2 (used in BitNet FFN sublayers)."""
    r = np.maximum(0.0, x)
    return r * r


# ---------------------------------------------------------------------------
# BitNet 1.58-bit quantization helpers
# ---------------------------------------------------------------------------

def absmean_quantize_weights(
    w_latent: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Absmean ternary weight quantization (W1.58A8 — weight side).

    Computes scale alpha = mean(|W|), then clips round(W/alpha) to {-1, 0, +1}.

    Args:
        w_latent: Float32 latent weight matrix of any shape.

    Returns:
        (w_ternary, alpha) where w_ternary is int8 in {-1, 0, +1} and alpha
        is the float32 absmean scale factor.
    """
    alpha = float(np.mean(np.abs(w_latent))) + 1e-8
    w_ternary = np.clip(np.round(w_latent / alpha), -1.0, 1.0).astype(np.int8)
    return w_ternary, alpha


def absmax_quantize_activations(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token absmax 8-bit activation quantization (W1.58A8 — activation side).

    Quantizes each row (token) independently to the range [-127, 127] and then
    dequantizes back to float32. Returned gamma can be used by callers that need
    the scale for further processing.

    Args:
        x: 2-D float32 array of shape (T, d) — one row per token.

    Returns:
        (x_q, gamma) where x_q is float32 (dequantized), gamma is (T, 1)
        absmax scale per token.
    """
    gamma = np.max(np.abs(x), axis=-1, keepdims=True) + 1e-8   # (T, 1)
    x_q = np.round(x / gamma * 127.0) / 127.0 * gamma
    return x_q.astype(np.float32), gamma


def rms_norm_bitnet(x: np.ndarray) -> np.ndarray:
    """RMS normalisation without learnable parameters (SubLN in BitNet).

    Applied before each attention and FFN sublayer. Works on any shape;
    normalises along the last axis.

    Args:
        x: Input array of shape (..., d).

    Returns:
        Normalised array of the same shape.
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-8
    return (x / rms).astype(np.float32)


def rms_norm_bitnet_backward(
    x: np.ndarray, d_out: np.ndarray
) -> np.ndarray:
    """Analytical backward pass through rms_norm_bitnet (no learnable params).

    Derived from the chain rule applied to y = x / rms(x).

    Args:
        x: Original input to rms_norm_bitnet, shape (T, d).
        d_out: Upstream gradient w.r.t. the normalised output, shape (T, d).

    Returns:
        Gradient w.r.t. x, same shape.
    """
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-8
    normed = x / rms
    # d(rms)/d(x_i) = x_i / (rms * d)
    # d_x_i = d_out_i / rms - normed_i * sum(d_out * normed) / d
    d_rms_coeff = np.sum(d_out * normed, axis=-1, keepdims=True) / d
    return ((d_out / rms) - (d_rms_coeff * normed / rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) helpers
# ---------------------------------------------------------------------------

def rope_frequencies(
    d_head: int, max_seq: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute interleaved cos/sin tables for RoPE.

    Each (position, dimension) pair gets the same angle as the canonical
    implementation: theta_i = 10000^(-2i/d_head) for i in [0, d_head//2).
    Pairs of adjacent dimensions share one angle.

    Args:
        d_head: Head dimension (must be even).
        max_seq: Maximum sequence length to precompute.

    Returns:
        (cos_table, sin_table), each of shape (max_seq, d_head) float32.
    """
    half = d_head // 2
    theta = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    pos = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(pos, theta)           # (max_seq, half)
    cos_half = np.cos(freqs)
    sin_half = np.sin(freqs)
    # Interleave: [c0, c0, c1, c1, ...] so adjacent pairs share an angle.
    cos_table = np.repeat(cos_half, 2, axis=-1).astype(np.float32)
    sin_table = np.repeat(sin_half, 2, axis=-1).astype(np.float32)
    return cos_table, sin_table


def rope_rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate interleaved pairs: (x0, x1) → (-x1, x0).

    Works on any leading dimensions; operates on the last axis.
    Requires the last dimension to be even.

    Args:
        x: Array with even last dimension.

    Returns:
        Rotated array of the same shape.
    """
    x_even = x[..., 0::2]    # x0, x2, x4, ...
    x_odd  = x[..., 1::2]    # x1, x3, x5, ...
    rotated = np.stack([-x_odd, x_even], axis=-1)
    return rotated.reshape(x.shape)


def rope_apply(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE to query and key tensors.

    Args:
        q: Query tensor, shape (T, n_heads, d_head).
        k: Key tensor, shape (T, n_heads, d_head).
        cos: Cosine table, shape (max_seq, d_head).
        sin: Sine table, shape (max_seq, d_head).

    Returns:
        (q_rot, k_rot), same shapes as inputs.
    """
    T = q.shape[0]
    c = cos[:T][:, np.newaxis, :]   # (T, 1, d_head)
    s = sin[:T][:, np.newaxis, :]
    q_rot = q * c + rope_rotate_half(q) * s
    k_rot = k * c + rope_rotate_half(k) * s
    return q_rot.astype(np.float32), k_rot.astype(np.float32)


def rope_apply_backward(
    d_q_rot: np.ndarray,
    d_k_rot: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward pass through rope_apply (STE — orthogonal inverse).

    The RoPE rotation R satisfies R^T = R^{-1}, so the gradient
    w.r.t. the pre-rotation input is obtained by applying the inverse:
        d_q = d_q_rot * cos - rotate_half(d_q_rot) * sin

    Args:
        d_q_rot: Upstream gradient w.r.t. q_rot, shape (T, n_heads, d_head).
        d_k_rot: Upstream gradient w.r.t. k_rot, shape (T, n_heads, d_head).
        cos: Same cosine table used in the forward pass.
        sin: Same sine table used in the forward pass.

    Returns:
        (d_q, d_k), gradients w.r.t. the pre-rotation q and k.
    """
    T = d_q_rot.shape[0]
    c = cos[:T][:, np.newaxis, :]
    s = sin[:T][:, np.newaxis, :]
    d_q = d_q_rot * c - rope_rotate_half(d_q_rot) * s
    d_k = d_k_rot * c - rope_rotate_half(d_k_rot) * s
    return d_q.astype(np.float32), d_k.astype(np.float32)


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board: Board2D, current_player: Player, n: int) -> np.ndarray:
    """Encode board as 3×n×n float32 array (3-channel representation).

    Channel 0: 1.0 where current_player has a piece.
    Channel 1: 1.0 where the opponent has a piece.
    Channel 2: all 1.0 (constant plane marking board size).

    Args:
        board: The current board.
        current_player: The player whose turn it is.
        n: Board dimension.

    Returns:
        A (3, n, n) float32 numpy array.
    """
    own_cell = current_player.to_cell()
    opp_cell = current_player.opponent().to_cell()
    arr = np.zeros((3, n, n), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            cell = board[r][c]
            if cell is own_cell:
                arr[0, r, c] = 1.0
            elif cell is opp_cell:
                arr[1, r, c] = 1.0
        arr[2, :, :] = 1.0
    return arr


def encode_board_flat(board: Board2D, current_player: Player, n: int) -> np.ndarray:
    """Encode board as flat 3*n*n float32 vector.

    Same channel layout as encode_board but flattened to 1-D.

    Args:
        board: The current board.
        current_player: The player whose turn it is.
        n: Board dimension.

    Returns:
        A (3*n*n,) float32 numpy array.
    """
    return encode_board(board, current_player, n).flatten()


# ---------------------------------------------------------------------------
# Convolution helper
# ---------------------------------------------------------------------------

def conv2d(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray,
    padding: int = 1,
) -> np.ndarray:
    """Pure-numpy 2-D convolution using sliding-window views.

    Uses ``np.lib.stride_tricks.sliding_window_view`` (numpy ≥ 1.20) to
    extract overlapping patches and then computes the output via
    ``np.tensordot``, which delegates to BLAS for efficiency.

    Args:
        x: Input feature map of shape (C_in, H, W).
        w: Convolution kernel of shape (C_out, C_in, kH, kW).
        bias: Bias vector of shape (C_out,).
        padding: Zero-padding size applied symmetrically on H and W axes.
            Use padding=1 for same-size output with a 3×3 kernel, and
            padding=0 for 1×1 convolutions.

    Returns:
        Output feature map of shape (C_out, H_out, W_out) as float32.
        With same padding H_out == H, W_out == W.
    """
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape

    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    else:
        x_padded = x

    H_out = H
    W_out = W

    # sliding_window_view produces shape (1, H_out, W_out, C_in, kH, kW)
    # because the window covers all C_in channels at once.
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, (C_in, kH, kW))
    # Drop the first (trivial) dimension → (H_out, W_out, C_in, kH, kW)
    windows = windows[0]

    # Reshape for matrix multiply
    windows_flat = windows.reshape(H_out, W_out, -1)    # (H, W, C_in*kH*kW)
    w_flat = w.reshape(C_out, -1)                        # (C_out, C_in*kH*kW)

    # tensordot: (H, W, f) × (C_out, f) → (H, W, C_out)
    out = np.tensordot(windows_flat, w_flat, axes=([-1], [-1]))
    # Reorder to (C_out, H, W) and add bias
    out = out.transpose(2, 0, 1) + bias.reshape(C_out, 1, 1)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# PolicyValueNetwork: AlphaZero-style convolutional residual network (float32)
# ---------------------------------------------------------------------------


class PolicyValueNetwork:
    """AlphaZero-inspired policy-value network with conv residual blocks.

    Follows the general structure of Google DeepMind's AlphaZero, adapted
    for n×n Tic-Tac-Toe. A shared convolutional trunk feeds separate policy
    and value heads.

    Architecture:
        Input:   3*n*n flat vector → reshaped to (4, n, n) with k/n plane
        Initial: Conv(4→64, 3×3, same) → ReLU
        Tower:   4 × ResBlock(64 filters)
                   Conv(64→64, 3×3, same) → ReLU → Conv(64→64, 3×3, same)
                   → add skip → ReLU
        Policy:  Conv(64→2, 1×1) → ReLU → flatten(2n²) → Linear(2n²→n²)
        Value:   Conv(64→1, 1×1) → ReLU → flatten(n²) → Linear(n²→64) → ReLU
                 → Linear(64→1) → tanh

    Weight precision: float32 throughout.
    Training:        Adam optimizer applied to output FC heads; conv trunk
                     weights are frozen at He-normal initialisation.
    k-encoding:      A 4th constant plane filled with k/n is appended to the
                     3-channel board input internally, so the network can
                     condition on the winning-run-length without changing the
                     external interface.

    Args:
        n: Board dimension (e.g. 3 for standard Tic-Tac-Toe).
        k: Winning-run length (default: n). Normalised to k/n and encoded as
           a constant 4th input plane.
    """

    NUM_RES_BLOCKS: int = 4
    FILTERS: int = 64

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Initialise all weight arrays with He-normal initialisation."""
        rng = np.random.default_rng(seed=0)
        n = self.n
        C = self.FILTERS

        def he_conv(c_in: int, c_out: int, kH: int = 3, kW: int = 3) -> np.ndarray:
            std = np.sqrt(2.0 / (c_in * kH * kW))
            return rng.normal(0.0, std, (c_out, c_in, kH, kW)).astype(np.float32)

        def he_fc(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        # Initial conv: 4 channels (3 board channels + 1 k-plane)
        self._conv_init_w = he_conv(4, C)                    # (C, 4, 3, 3)
        self._conv_init_b = np.zeros(C, dtype=np.float32)

        # Residual blocks: list of (w1, b1, w2, b2)
        self._res_blocks: list = []
        for _ in range(self.NUM_RES_BLOCKS):
            self._res_blocks.append((
                he_conv(C, C),                               # (C, C, 3, 3)
                np.zeros(C, dtype=np.float32),
                he_conv(C, C),                               # (C, C, 3, 3)
                np.zeros(C, dtype=np.float32),
            ))

        # Policy head: 1×1 conv + FC
        self._conv_p_w = he_conv(C, 2, kH=1, kW=1)          # (2, C, 1, 1)
        self._conv_p_b = np.zeros(2, dtype=np.float32)
        self._wp = he_fc(2 * n * n, n * n)                   # (2n², n²)
        self._bp = np.zeros(n * n, dtype=np.float32)

        # Value head: 1×1 conv + 2-layer FC
        self._conv_v_w = he_conv(C, 1, kH=1, kW=1)          # (1, C, 1, 1)
        self._conv_v_b = np.zeros(1, dtype=np.float32)
        self._wv1 = he_fc(n * n, 64)                         # (n², 64)
        self._bv1 = np.zeros(64, dtype=np.float32)
        self._wv2 = he_fc(64, 1)                             # (64, 1)
        self._bv2 = np.zeros(1, dtype=np.float32)

        # Adam optimizer state (output FC heads only)
        self._t: int = 0
        self._m_wp = np.zeros_like(self._wp)
        self._v_wp = np.zeros_like(self._wp)
        self._m_bp = np.zeros_like(self._bp)
        self._v_bp = np.zeros_like(self._bp)
        self._m_wv1 = np.zeros_like(self._wv1)
        self._v_wv1 = np.zeros_like(self._wv1)
        self._m_bv1 = np.zeros_like(self._bv1)
        self._v_bv1 = np.zeros_like(self._bv1)
        self._m_wv2 = np.zeros_like(self._wv2)
        self._v_wv2 = np.zeros_like(self._wv2)
        self._m_bv2 = np.zeros_like(self._bv2)
        self._v_bv2 = np.zeros_like(self._bv2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_4channel(self, x: np.ndarray) -> np.ndarray:
        """Append a k/n constant plane to a 3-channel flat board encoding.

        Args:
            x: Flat board encoding of length 3*n*n.

        Returns:
            4-channel feature map of shape (4, n, n).
        """
        spatial = x.reshape(3, self.n, self.n)
        k_plane = np.full((1, self.n, self.n), self.k / self.n, dtype=np.float32)
        return np.concatenate([spatial, k_plane], axis=0)

    def _forward_trunk(self, feat: np.ndarray) -> np.ndarray:
        """Run the convolutional trunk and return the final feature map.

        Args:
            feat: 4-channel input feature map of shape (4, n, n).

        Returns:
            Feature map of shape (FILTERS, n, n) after the residual tower.
        """
        feat = relu(conv2d(feat, self._conv_init_w, self._conv_init_b, padding=1))
        for w1, b1, w2, b2 in self._res_blocks:
            skip = feat
            out = relu(conv2d(feat, w1, b1, padding=1))
            out = conv2d(out, w2, b2, padding=1)
            feat = relu(out + skip)
        return feat

    def _adam_step(
        self,
        m: np.ndarray,
        v: np.ndarray,
        grad: np.ndarray,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute one Adam parameter update (modifies m and v in-place).

        Must be called after incrementing self._t.

        Args:
            m: First-moment estimate (modified in-place).
            v: Second-moment estimate (modified in-place).
            grad: Gradient for this parameter.
            lr: Learning rate.
            beta1: Exponential decay for first moment.
            beta2: Exponential decay for second moment.
            eps: Numerical stability constant.

        Returns:
            Parameter update ∆ (subtract from parameter: param -= ∆).
        """
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * grad ** 2
        m_hat = m / (1.0 - beta1 ** self._t)
        v_hat = v / (1.0 - beta2 ** self._t)
        return lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass.

        Args:
            x: Flat board encoding of length 3*n*n. The network appends
               a k/n plane internally before processing.

        Returns:
            (policy_logits, value) where policy_logits has length n*n and
            value is a scalar in [-1, 1].
        """
        feat = self._forward_trunk(self._to_4channel(x))

        # Policy head
        p_h = relu(conv2d(feat, self._conv_p_w, self._conv_p_b, padding=0))
        flat_p = p_h.flatten()                               # (2n²,)
        policy_logits = flat_p @ self._wp + self._bp         # (n²,)

        # Value head
        v_h = relu(conv2d(feat, self._conv_v_w, self._conv_v_b, padding=0))
        flat_v = v_h.flatten()                               # (n²,)
        h_v = relu(flat_v @ self._wv1 + self._bv1)          # (64,)
        value = float(tanh_act(h_v @ self._wv2 + self._bv2)[0])

        return policy_logits, value

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Apply one Adam update over a mini-batch of examples.

        Accumulates output-head gradients across all examples in the batch,
        then applies a single Adam update. Convolutional trunk weights are
        frozen; only the FC output heads (_wp, _bp, _wv1, _bv1, _wv2, _bv2)
        are updated.

        Args:
            batch: List of (state_enc, dp, dv) tuples where:
                state_enc: Flat input of length 3*n*n.
                dp: Gradient w.r.t. policy logits (length n*n), e.g.
                    softmax(logits) - target_policy.
                dv: Gradient w.r.t. value output (length 1), e.g.
                    2*(value - target_value).
            lr: Learning rate for the Adam optimizer.
        """
        g_wp = np.zeros_like(self._wp)
        g_bp = np.zeros_like(self._bp)
        g_wv1 = np.zeros_like(self._wv1)
        g_bv1 = np.zeros_like(self._bv1)
        g_wv2 = np.zeros_like(self._wv2)
        g_bv2 = np.zeros_like(self._bv2)

        for x, dp, dv in batch:
            feat = self._forward_trunk(self._to_4channel(x))

            # Policy head activations
            p_h = relu(conv2d(feat, self._conv_p_w, self._conv_p_b, padding=0))
            flat_p = p_h.flatten()

            # Value head activations (full forward for backprop)
            v_h = relu(conv2d(feat, self._conv_v_w, self._conv_v_b, padding=0))
            flat_v = v_h.flatten()
            h_v_pre = flat_v @ self._wv1 + self._bv1        # (64,)
            h_v = relu(h_v_pre)                              # (64,)
            v_pre = h_v @ self._wv2 + self._bv2             # (1,)
            value = float(tanh_act(v_pre)[0])

            # Policy gradients
            g_wp += np.outer(flat_p, dp)
            g_bp += dp

            # Value gradients (backprop through tanh + 2-layer FC)
            dv_pre = dv * (1.0 - value ** 2)                # tanh derivative
            g_wv2 += np.outer(h_v, dv_pre)
            g_bv2 += dv_pre
            d_h_v = (self._wv2 @ dv_pre)                    # (64,)
            d_h_v_pre = d_h_v * (h_v_pre > 0)               # ReLU backward
            g_wv1 += np.outer(flat_v, d_h_v_pre)
            g_bv1 += d_h_v_pre

        n_b = len(batch)
        self._t += 1
        self._wp -= self._adam_step(self._m_wp, self._v_wp, g_wp / n_b, lr)
        self._bp -= self._adam_step(self._m_bp, self._v_bp, g_bp / n_b, lr)
        self._wv1 -= self._adam_step(self._m_wv1, self._v_wv1, g_wv1 / n_b, lr)
        self._bv1 -= self._adam_step(self._m_bv1, self._v_bv1, g_bv1 / n_b, lr)
        self._wv2 -= self._adam_step(self._m_wv2, self._v_wv2, g_wv2 / n_b, lr)
        self._bv2 -= self._adam_step(self._m_bv2, self._v_bv2, g_bv2 / n_b, lr)

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Perform one Adam update for a single example.

        Convenience wrapper around backward_batch for single-example use.

        Args:
            x: Flat input vector of length 3*n*n.
            dp: Gradient w.r.t. policy logits (length n*n).
            dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays as a flat list (for get/set roundtrip).

        Returns:
            Flat list of numpy arrays in a fixed order matching set_weights().
        """
        weights: list[np.ndarray] = [self._conv_init_w, self._conv_init_b]
        for w1, b1, w2, b2 in self._res_blocks:
            weights.extend([w1, b1, w2, b2])
        weights.extend([
            self._conv_p_w, self._conv_p_b, self._wp, self._bp,
            self._conv_v_w, self._conv_v_b,
            self._wv1, self._bv1, self._wv2, self._bv2,
        ])
        return weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set all weight arrays from a flat list (same order as get_weights).

        Args:
            weights: Flat list of numpy arrays returned by get_weights().
        """
        idx = 0
        self._conv_init_w = weights[idx]; idx += 1
        self._conv_init_b = weights[idx]; idx += 1
        self._res_blocks = []
        for _ in range(self.NUM_RES_BLOCKS):
            w1 = weights[idx]; idx += 1
            b1 = weights[idx]; idx += 1
            w2 = weights[idx]; idx += 1
            b2 = weights[idx]; idx += 1
            self._res_blocks.append((w1, b1, w2, b2))
        self._conv_p_w = weights[idx]; idx += 1
        self._conv_p_b = weights[idx]; idx += 1
        self._wp = weights[idx]; idx += 1
        self._bp = weights[idx]; idx += 1
        self._conv_v_w = weights[idx]; idx += 1
        self._conv_v_b = weights[idx]; idx += 1
        self._wv1 = weights[idx]; idx += 1
        self._bv1 = weights[idx]; idx += 1
        self._wv2 = weights[idx]; idx += 1
        self._bv2 = weights[idx]; idx += 1

    def copy(self) -> PolicyValueNetwork:
        """Return a deep copy of this network (weights and optimizer state).

        Returns:
            New PolicyValueNetwork with all weights and Adam state copied.
        """
        net = PolicyValueNetwork.__new__(PolicyValueNetwork)
        net.n = self.n
        net.k = self.k
        net._conv_init_w = self._conv_init_w.copy()
        net._conv_init_b = self._conv_init_b.copy()
        net._res_blocks = [
            (w1.copy(), b1.copy(), w2.copy(), b2.copy())
            for w1, b1, w2, b2 in self._res_blocks
        ]
        net._conv_p_w = self._conv_p_w.copy()
        net._conv_p_b = self._conv_p_b.copy()
        net._wp = self._wp.copy()
        net._bp = self._bp.copy()
        net._conv_v_w = self._conv_v_w.copy()
        net._conv_v_b = self._conv_v_b.copy()
        net._wv1 = self._wv1.copy()
        net._bv1 = self._bv1.copy()
        net._wv2 = self._wv2.copy()
        net._bv2 = self._bv2.copy()
        net._t = self._t
        net._m_wp = self._m_wp.copy(); net._v_wp = self._v_wp.copy()
        net._m_bp = self._m_bp.copy(); net._v_bp = self._v_bp.copy()
        net._m_wv1 = self._m_wv1.copy(); net._v_wv1 = self._v_wv1.copy()
        net._m_bv1 = self._m_bv1.copy(); net._v_bv1 = self._v_bv1.copy()
        net._m_wv2 = self._m_wv2.copy(); net._v_wv2 = self._v_wv2.copy()
        net._m_bv2 = self._m_bv2.copy(); net._v_bv2 = self._v_bv2.copy()
        return net

    def save(self, path: str) -> None:
        """Save all weights to a .npz file (path without extension).

        Args:
            path: Destination path without .npz extension.
        """
        arrays: dict = {
            'conv_init_w': self._conv_init_w,
            'conv_init_b': self._conv_init_b,
            'conv_p_w': self._conv_p_w,
            'conv_p_b': self._conv_p_b,
            'wp': self._wp, 'bp': self._bp,
            'conv_v_w': self._conv_v_w,
            'conv_v_b': self._conv_v_b,
            'wv1': self._wv1, 'bv1': self._bv1,
            'wv2': self._wv2, 'bv2': self._bv2,
        }
        for i, (w1, b1, w2, b2) in enumerate(self._res_blocks):
            arrays[f'res{i}_w1'] = w1
            arrays[f'res{i}_b1'] = b1
            arrays[f'res{i}_w2'] = w2
            arrays[f'res{i}_b2'] = b2
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load weights from a .npz file.

        Args:
            path: Source path with .npz extension.
        """
        data = np.load(path)
        self._conv_init_w = data['conv_init_w']
        self._conv_init_b = data['conv_init_b']
        self._conv_p_w = data['conv_p_w']
        self._conv_p_b = data['conv_p_b']
        self._wp = data['wp']
        self._bp = data['bp']
        self._conv_v_w = data['conv_v_w']
        self._conv_v_b = data['conv_v_b']
        self._wv1 = data['wv1']
        self._bv1 = data['bv1']
        self._wv2 = data['wv2']
        self._bv2 = data['bv2']
        self._res_blocks = []
        for i in range(self.NUM_RES_BLOCKS):
            self._res_blocks.append((
                data[f'res{i}_w1'], data[f'res{i}_b1'],
                data[f'res{i}_w2'], data[f'res{i}_b2'],
            ))


# ---------------------------------------------------------------------------
# QuantizedPolicyValueNetwork: Memory-optimized version with 8-bit quantization
# ---------------------------------------------------------------------------


class QuantizedPolicyValueNetwork:
    """Memory-optimized PolicyValueNetwork with 8-bit weight quantization.

    Uses symmetric quantization: weights stored as int8 with per-tensor scale.
    Activations remain float32 for precision. Reduces memory by ~4x vs float32.

    Architecture (optimized for TicTacToe):
        Input:  3*n*n + 1 flat vector (k/n appended as extra scalar)
        Hidden: Linear(3n²+1, 128) → ReLU → Linear(128, 64) → ReLU
        Policy: Linear(64, n²)   (logits; apply softmax externally)
        Value:  Linear(64, 1)    → tanh

    Weights quantized to int8, biases remain float32.
    """

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        input_size = 3 * n * n + 1
        self._build(input_size, n * n)

    def _build(self, input_size: int, output_size: int) -> None:
        rng = np.random.default_rng(seed=0)

        def he(fan_in: int, fan_out: int) -> tuple[np.ndarray, float]:
            """Return (quantized_weights, scale) for He initialization."""
            std = np.sqrt(2.0 / fan_in)
            weights_fp32 = rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
            # Symmetric quantization: find scale
            abs_max = np.max(np.abs(weights_fp32))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            weights_int8 = np.round(weights_fp32 / scale).astype(np.int8)
            return weights_int8, scale

        # Layer 1: 3n² → 128
        self._w1_int8, self._w1_scale = he(input_size, 128)
        self._b1 = np.zeros(128, dtype=np.float32)

        # Layer 2: 128 → 64
        self._w2_int8, self._w2_scale = he(128, 64)
        self._b2 = np.zeros(64, dtype=np.float32)

        # Policy head: 64 → n²
        self._wp_int8, self._wp_scale = he(64, output_size)
        self._bp = np.zeros(output_size, dtype=np.float32)

        # Value head: 64 → 1
        self._wv_int8, self._wv_scale = he(64, 1)
        self._bv = np.zeros(1, dtype=np.float32)

    def _dequantize(self, w_int8: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize int8 weights back to float32."""
        return w_int8.astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass with on-demand dequantization.

        Args:
            x: Flat input vector of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits is length n*n and
            value is a scalar in [-1, 1].
        """
        x = np.append(x, np.float32(self.k / self.n))
        # Layer 1
        w1_fp32 = self._dequantize(self._w1_int8, self._w1_scale)
        h1 = relu(x @ w1_fp32 + self._b1)

        # Layer 2
        w2_fp32 = self._dequantize(self._w2_int8, self._w2_scale)
        h2 = relu(h1 @ w2_fp32 + self._b2)

        # Policy head
        wp_fp32 = self._dequantize(self._wp_int8, self._wp_scale)
        policy_logits = h2 @ wp_fp32 + self._bp

        # Value head
        wv_fp32 = self._dequantize(self._wv_int8, self._wv_scale)
        value = float(tanh_act(h2 @ wv_fp32 + self._bv)[0])

        return policy_logits, value

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays (quantized int8 + scales + float32 biases)."""
        return [
            self._w1_int8, np.array([self._w1_scale], dtype=np.float32), self._b1,
            self._w2_int8, np.array([self._w2_scale], dtype=np.float32), self._b2,
            self._wp_int8, np.array([self._wp_scale], dtype=np.float32), self._bp,
            self._wv_int8, np.array([self._wv_scale], dtype=np.float32), self._bv,
        ]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set weights from quantized format."""
        (
            self._w1_int8, w1_scale_arr, self._b1,
            self._w2_int8, w2_scale_arr, self._b2,
            self._wp_int8, wp_scale_arr, self._bp,
            self._wv_int8, wv_scale_arr, self._bv,
        ) = weights
        self._w1_scale = float(w1_scale_arr[0])
        self._w2_scale = float(w2_scale_arr[0])
        self._wp_scale = float(wp_scale_arr[0])
        self._wv_scale = float(wv_scale_arr[0])

    def copy(self) -> QuantizedPolicyValueNetwork:
        """Return a deep copy of this network."""
        net = QuantizedPolicyValueNetwork.__new__(QuantizedPolicyValueNetwork)
        net.n = self.n
        net.k = self.k
        net._w1_int8 = self._w1_int8.copy()
        net._w1_scale = self._w1_scale
        net._b1 = self._b1.copy()
        net._w2_int8 = self._w2_int8.copy()
        net._w2_scale = self._w2_scale
        net._b2 = self._b2.copy()
        net._wp_int8 = self._wp_int8.copy()
        net._wp_scale = self._wp_scale
        net._bp = self._bp.copy()
        net._wv_int8 = self._wv_int8.copy()
        net._wv_scale = self._wv_scale
        net._bv = self._bv.copy()
        return net

    def save(self, path: str) -> None:
        """Save quantized weights to a .npz file."""
        np.savez(
            path,
            w1_int8=self._w1_int8, w1_scale=self._w1_scale, b1=self._b1,
            w2_int8=self._w2_int8, w2_scale=self._w2_scale, b2=self._b2,
            wp_int8=self._wp_int8, wp_scale=self._wp_scale, bp=self._bp,
            wv_int8=self._wv_int8, wv_scale=self._wv_scale, bv=self._bv,
        )

    def load(self, path: str) -> None:
        """Load quantized weights from a .npz file."""
        data = np.load(path)
        self._w1_int8 = data['w1_int8']
        self._w1_scale = float(data['w1_scale'])
        self._b1 = data['b1']
        self._w2_int8 = data['w2_int8']
        self._w2_scale = float(data['w2_scale'])
        self._b2 = data['b2']
        self._wp_int8 = data['wp_int8']
        self._wp_scale = float(data['wp_scale'])
        self._bp = data['bp']
        self._wv_int8 = data['wv_int8']
        self._wv_scale = float(data['wv_scale'])
        self._bv = data['bv']

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Apply one SGD update over a mini-batch (bias-only).

        Averages dp and dv across the batch and applies an SGD step to
        the float32 policy and value biases. The int8 weight matrices
        remain frozen throughout training.

        Args:
            batch: List of (state_enc, dp, dv) tuples where:
                state_enc: Flat input of length 3*n*n.
                dp: Gradient w.r.t. policy logits (length n*n).
                dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        g_bp = np.zeros_like(self._bp)
        g_bv = np.zeros_like(self._bv)
        for _, dp, dv in batch:
            g_bp += dp
            g_bv += dv
        n_b = len(batch)
        self._bp -= lr * g_bp / n_b
        self._bv -= lr * g_bv / n_b

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Perform one SGD update step given pre-computed output gradients.

        Dequantizes weights to compute hidden activations, then updates the
        float32 bias terms. The int8 weight matrices are left unchanged —
        gradient-based learning is applied only to the trainable float32 biases.

        Args:
            x: Flat input vector of length 3*n*n.
            dp: Gradient w.r.t. policy logits (length n*n).
            dv: Gradient w.r.t. value scalar (length 1).
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB (quantized vs float32)."""
        # Count int8 weights + float32 scales/biases
        int8_bytes = (
            self._w1_int8.nbytes + self._w2_int8.nbytes +
            self._wp_int8.nbytes + self._wv_int8.nbytes
        )
        float32_bytes = (
            self._b1.nbytes + self._b2.nbytes + self._bp.nbytes + self._bv.nbytes +
            4 * 4  # 4 scales as float32
        )
        return (int8_bytes + float32_bytes) / (1024 * 1024)


# ---------------------------------------------------------------------------
# TernaryPolicyValueNetwork: BitNet-style ternary weights for extreme efficiency
# ---------------------------------------------------------------------------


class TernaryPolicyValueNetwork:
    """BitNet-style ternary policy-value network with weights in {-1, 0, +1}.

    Ternary quantization achieves 4x memory reduction vs float32 and 2x vs int8,
    while maintaining comparable accuracy in RL settings. This implementation
    uses symmetric ternary quantization with learnable scaling per layer.

    Architecture (optimized for TicTacToe):
        Input:  3*n*n + 1 flat vector (k/n appended as extra scalar)
        Hidden: Linear(3n²+1, 128) → ReLU → Linear(128, 64) → ReLU
        Policy: Linear(64, n²)   (logits; apply softmax externally)
        Value:  Linear(64, 1)    → tanh

    Weights constrained to {-1, 0, +1} stored as int8. Biases remain float32.
    Scale factors enable training flexibility while keeping weights ternary.

    Memory efficiency (for 3x3 board, n=3):
        - float32: 3×27×256 + 256×128 + 128×64 + 64×9 + 64×1 ≈ 88KB weights
        - Ternary:  (same) ÷ 4 ≈ 22KB weights (4x reduction)
    """

    def __init__(self, n: int, k: int | None = None) -> None:
        """Initialize ternary network.

        Args:
            n: Board dimension (3, 4, 5, ...).
            k: Winning-run-length (3 ≤ k ≤ n). Defaults to n.
        """
        self.n = n
        self.k = k if k is not None else n
        input_size = 3 * n * n + 1
        self._build(input_size, n * n)

    def _build(self, input_size: int, output_size: int) -> None:
        """Build ternary network with random ternary initialization.

        Ternary weights are initialized from {-1, 0, +1} with sparsity ~50%
        (approximately half the weights are 0). This balances expressivity
        with the constraints of ternary quantization.

        Per-layer scale factors (stored separately) are initialized to 1.0
        and can be adapted during training through gradient-based updates
        applied to scale while keeping weights strictly ternary.
        """
        rng = np.random.default_rng(seed=0)

        def ternary(fan_in: int, fan_out: int) -> np.ndarray:
            """Initialize ternary weight matrix from {-1, 0, +1}.

            Sparsity: ~50% of weights are 0 (uniformly random).
            Non-zero weights: equally split between +1 and -1.
            """
            weights = np.zeros((fan_in, fan_out), dtype=np.int8)
            num_nonzero = rng.integers(0, fan_in * fan_out // 2 + 1)
            nonzero_indices = rng.choice(
                fan_in * fan_out, size=num_nonzero, replace=False
            )
            for idx in nonzero_indices:
                weights.flat[idx] = rng.choice([-1, 1])
            return weights

        # Layer 1: 3n² → 128
        self._w1_ternary = ternary(input_size, 128)
        self._w1_scale = 1.0
        self._b1 = np.zeros(128, dtype=np.float32)

        # Layer 2: 128 → 64
        self._w2_ternary = ternary(128, 64)
        self._w2_scale = 1.0
        self._b2 = np.zeros(64, dtype=np.float32)

        # Policy head: 64 → n²
        self._wp_ternary = ternary(64, output_size)
        self._wp_scale = 1.0
        self._bp = np.zeros(output_size, dtype=np.float32)

        # Value head: 64 → 1
        self._wv_ternary = ternary(64, 1)
        self._wv_scale = 1.0
        self._bv = np.zeros(1, dtype=np.float32)

    def _ternary_to_float(
        self, w_ternary: np.ndarray, scale: float
    ) -> np.ndarray:
        """Convert ternary weights (int8: {-1, 0, +1}) to float32.

        Args:
            w_ternary: Integer matrix with values in {-1, 0, +1}.
            scale: Learnable scale factor (typically ~1.0).

        Returns:
            float32 weight matrix after scaling.
        """
        return w_ternary.astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass with ternary weights.

        Args:
            x: Flat input vector of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits is length n*n and
            value is a scalar in [-1, 1].
        """
        x = np.append(x, np.float32(self.k / self.n))
        # Layer 1: convert ternary → float32, compute
        w1_fp32 = self._ternary_to_float(self._w1_ternary, self._w1_scale)
        h1 = relu(x @ w1_fp32 + self._b1)

        # Layer 2: convert ternary → float32, compute
        w2_fp32 = self._ternary_to_float(self._w2_ternary, self._w2_scale)
        h2 = relu(h1 @ w2_fp32 + self._b2)

        # Policy head: convert ternary → float32, compute logits
        wp_fp32 = self._ternary_to_float(self._wp_ternary, self._wp_scale)
        policy_logits = h2 @ wp_fp32 + self._bp

        # Value head: convert ternary → float32, compute value
        wv_fp32 = self._ternary_to_float(self._wv_ternary, self._wv_scale)
        value = float(tanh_act(h2 @ wv_fp32 + self._bv)[0])

        return policy_logits, value

    def _ensure_ternary(self, w: np.ndarray) -> np.ndarray:
        """Enforce ternary constraint: round to nearest value in {-1, 0, +1}.

        Used after weight updates to maintain ternary structure.
        This is a simple rounding approach; advanced methods could use
        learned thresholds or straight-through estimators for gradients.
        """
        return np.round(np.clip(w, -1, 1)).astype(np.int8)

    def quantize_to_ternary(self, w_float: np.ndarray) -> np.ndarray:
        """Quantize floating-point weights to ternary {-1, 0, +1}.

        Simple ternary quantization: find threshold via percentile method.

        Args:
            w_float: Float32 weight matrix (e.g., from a pre-trained network).

        Returns:
            Ternary int8 weight matrix.
        """
        # Use percentile thresholding: assign to nearest ternary value
        w_abs = np.abs(w_float)
        # Threshold at 50th percentile of absolute values
        threshold = np.percentile(w_abs, 50)

        ternary = np.zeros_like(w_float, dtype=np.int8)
        mask_zero = w_abs < threshold
        mask_pos = (w_float >= threshold)
        mask_neg = (w_float < -threshold)

        ternary[mask_zero] = 0
        ternary[mask_pos] = 1
        ternary[mask_neg] = -1

        return ternary

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Apply one SGD update over a mini-batch (bias-only).

        Averages dp and dv across the batch and applies an SGD step to
        the float32 policy and value biases. Ternary weight matrices are
        frozen — gradient-based updates apply only to the trainable biases.

        Args:
            batch: List of (state_enc, dp, dv) tuples where:
                state_enc: Flat input of length 3*n*n.
                dp: Gradient w.r.t. policy logits (length n*n).
                dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        g_bp = np.zeros_like(self._bp)
        g_bv = np.zeros_like(self._bv)
        for _, dp, dv in batch:
            g_bp += dp
            g_bv += dv
        n_b = len(batch)
        self._bp -= lr * g_bp / n_b
        self._bv -= lr * g_bv / n_b

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Perform one SGD update step given pre-computed output gradients.

        Converts ternary weights to float32 to compute hidden activations, then
        updates the float32 bias terms. Ternary weight matrices are frozen —
        gradient-based updates apply only to the trainable float32 biases.

        Args:
            x: Flat input vector of length 3*n*n.
            dp: Gradient w.r.t. policy logits (length n*n).
            dv: Gradient w.r.t. value scalar (length 1).
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays (ternary int8 + scale factors + float32 biases).

        Format mirrors QuantizedPolicyValueNetwork for compatibility.
        """
        return [
            self._w1_ternary,
            np.array([self._w1_scale], dtype=np.float32),
            self._b1,
            self._w2_ternary,
            np.array([self._w2_scale], dtype=np.float32),
            self._b2,
            self._wp_ternary,
            np.array([self._wp_scale], dtype=np.float32),
            self._bp,
            self._wv_ternary,
            np.array([self._wv_scale], dtype=np.float32),
            self._bv,
        ]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set all weights from quantized format.

        Args:
            weights: List in order [w1_ternary, w1_scale, b1, w2_ternary, ...].
        """
        (
            self._w1_ternary,
            w1_scale_arr,
            self._b1,
            self._w2_ternary,
            w2_scale_arr,
            self._b2,
            self._wp_ternary,
            wp_scale_arr,
            self._bp,
            self._wv_ternary,
            wv_scale_arr,
            self._bv,
        ) = weights
        self._w1_scale = float(w1_scale_arr[0])
        self._w2_scale = float(w2_scale_arr[0])
        self._wp_scale = float(wp_scale_arr[0])
        self._wv_scale = float(wv_scale_arr[0])

    def copy(self) -> TernaryPolicyValueNetwork:
        """Return a deep copy of this network."""
        net = TernaryPolicyValueNetwork.__new__(TernaryPolicyValueNetwork)
        net.n = self.n
        net.k = self.k
        net._w1_ternary = self._w1_ternary.copy()
        net._w1_scale = self._w1_scale
        net._b1 = self._b1.copy()
        net._w2_ternary = self._w2_ternary.copy()
        net._w2_scale = self._w2_scale
        net._b2 = self._b2.copy()
        net._wp_ternary = self._wp_ternary.copy()
        net._wp_scale = self._wp_scale
        net._bp = self._bp.copy()
        net._wv_ternary = self._wv_ternary.copy()
        net._wv_scale = self._wv_scale
        net._bv = self._bv.copy()
        return net

    def save(self, path: str) -> None:
        """Save ternary weights to a .npz file."""
        np.savez(
            path,
            w1_ternary=self._w1_ternary,
            w1_scale=self._w1_scale,
            b1=self._b1,
            w2_ternary=self._w2_ternary,
            w2_scale=self._w2_scale,
            b2=self._b2,
            wp_ternary=self._wp_ternary,
            wp_scale=self._wp_scale,
            bp=self._bp,
            wv_ternary=self._wv_ternary,
            wv_scale=self._wv_scale,
            bv=self._bv,
        )

    def load(self, path: str) -> None:
        """Load ternary weights from a .npz file."""
        data = np.load(path)
        self._w1_ternary = data["w1_ternary"]
        self._w1_scale = float(data["w1_scale"])
        self._b1 = data["b1"]
        self._w2_ternary = data["w2_ternary"]
        self._w2_scale = float(data["w2_scale"])
        self._b2 = data["b2"]
        self._wp_ternary = data["wp_ternary"]
        self._wp_scale = float(data["wp_scale"])
        self._bp = data["bp"]
        self._wv_ternary = data["wv_ternary"]
        self._wv_scale = float(data["wv_scale"])
        self._bv = data["bv"]

    def estimate_memory_bytes(self) -> dict[str, int]:
        """Estimate memory usage of this network.

        Returns:
            Dictionary with memory breakdown in bytes.
        """
        # Count ternary weights (int8, but logically ternary)
        w1_size = self._w1_ternary.size
        w2_size = self._w2_ternary.size
        wp_size = self._wp_ternary.size
        wv_size = self._wv_ternary.size
        total_ternary_params = w1_size + w2_size + wp_size + wv_size

        # Each ternary weight: 1 byte (stored as int8)
        ternary_bytes = total_ternary_params * 1

        # Scale factors: 4 float32s (4 bytes each)
        scale_bytes = 4 * 4

        # Biases: all float32
        bias_params = 128 + 64 + self.n * self.n + 1
        bias_bytes = bias_params * 4

        total_bytes = ternary_bytes + scale_bytes + bias_bytes

        return {
            "ternary_weights_bytes": ternary_bytes,
            "scale_factors_bytes": scale_bytes,
            "biases_bytes": bias_bytes,
            "total_bytes": total_bytes,
            "total_ternary_params": total_ternary_params,
        }

    def compare_memory_to_float32(self) -> dict[str, float]:
        """Compare memory usage to standard PolicyValueNetwork (float32).

        Returns:
            Dictionary with size comparisons in KB and reduction ratios.
        """
        # Estimate equivalent float32 network
        w1_size = self._w1_ternary.size
        w2_size = self._w2_ternary.size
        wp_size = self._wp_ternary.size
        wv_size = self._wv_ternary.size
        total_ternary_params = w1_size + w2_size + wp_size + wv_size

        float32_weights_bytes = total_ternary_params * 4
        bias_params = 128 + 64 + self.n * self.n + 1
        float32_bias_bytes = bias_params * 4
        float32_total = float32_weights_bytes + float32_bias_bytes

        ternary_stats = self.estimate_memory_bytes()
        ternary_total = ternary_stats["total_bytes"]

        return {
            "float32_total_kb": float32_total / 1024,
            "ternary_total_kb": ternary_total / 1024,
            "reduction_ratio": float32_total / ternary_total,
            "savings_percent": 100 * (1 - ternary_total / float32_total),
        }


# ---------------------------------------------------------------------------
# LargeQuantizedPolicyValueNetwork: Deep conv int8 network (AlphaZero-scale)
# ---------------------------------------------------------------------------


class LargeQuantizedPolicyValueNetwork:
    """Deep AlphaZero-style network with 10 residual blocks and int8 weights.

    Maximises network capacity within an int8 weight budget. With 128 filters
    and 10 residual blocks, this network has ~3 million parameters (~2.9 MB
    as int8, ~11 MB as float32) — roughly 36× more than the small
    QuantizedPolicyValueNetwork, at 4× float32 compression.

    Architecture:
        Input:   3*n*n flat vector → reshaped to (4, n, n) with k/n plane
        Initial: Conv(4→128, 3×3, same), weights int8
        Tower:   10 × ResBlock(128 filters), weights int8
                   Conv(128→128, 3×3) → ReLU → Conv(128→128, 3×3) → + skip → ReLU
        Policy:  Conv(128→2, 1×1, int8) → ReLU → flatten(2n²) → Linear(2n²→n², int8)
        Value:   Conv(128→1, 1×1, int8) → ReLU → flatten(n²) → Linear(n²→256, int8)
                 → ReLU → Linear(256→1, int8) → tanh

    Weight precision: int8 with per-tensor float32 scale factors.
    Biases:          float32 (updated via SGD mini-batch).
    Trunk:           Frozen after He-normal initialisation + int8 quantisation.

    Args:
        n: Board dimension.
        k: Winning-run length (default: n). Encoded as k/n in the 4th channel.
    """

    NUM_RES_BLOCKS: int = 10
    FILTERS: int = 128

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        self._build()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize(w_fp32: np.ndarray) -> tuple[np.ndarray, float]:
        """Symmetric int8 quantisation (per-tensor scale).

        Args:
            w_fp32: Float32 weight array.

        Returns:
            (int8_weights, scale) tuple.
        """
        abs_max = float(np.max(np.abs(w_fp32)))
        scale = abs_max / 127.0 if abs_max > 0.0 else 1.0
        return np.round(w_fp32 / scale).astype(np.int8), scale

    def _build(self) -> None:
        """Initialise all int8 conv weights and float32 biases."""
        rng = np.random.default_rng(seed=0)
        n = self.n
        C = self.FILTERS

        def he_conv_q(c_in: int, c_out: int, kH: int = 3, kW: int = 3):
            std = np.sqrt(2.0 / (c_in * kH * kW))
            w = rng.normal(0.0, std, (c_out, c_in, kH, kW)).astype(np.float32)
            return self._quantize(w)

        def he_fc_q(fan_in: int, fan_out: int):
            std = np.sqrt(2.0 / fan_in)
            w = rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
            return self._quantize(w)

        # Initial conv: 4 input channels
        self._conv_init_int8, self._conv_init_scale = he_conv_q(4, C)
        self._conv_init_b = np.zeros(C, dtype=np.float32)

        # Residual blocks: list of (w1_int8, s1, b1, w2_int8, s2, b2)
        self._res_blocks: list = []
        for _ in range(self.NUM_RES_BLOCKS):
            w1, s1 = he_conv_q(C, C)
            w2, s2 = he_conv_q(C, C)
            self._res_blocks.append((
                w1, s1, np.zeros(C, dtype=np.float32),
                w2, s2, np.zeros(C, dtype=np.float32),
            ))

        # Policy head conv + FC
        self._conv_p_int8, self._conv_p_scale = he_conv_q(C, 2, kH=1, kW=1)
        self._conv_p_b = np.zeros(2, dtype=np.float32)
        self._wp_int8, self._wp_scale = he_fc_q(2 * n * n, n * n)
        self._bp = np.zeros(n * n, dtype=np.float32)

        # Value head conv + 2-layer FC
        self._conv_v_int8, self._conv_v_scale = he_conv_q(C, 1, kH=1, kW=1)
        self._conv_v_b = np.zeros(1, dtype=np.float32)
        self._wv1_int8, self._wv1_scale = he_fc_q(n * n, 256)
        self._bv1 = np.zeros(256, dtype=np.float32)
        self._wv2_int8, self._wv2_scale = he_fc_q(256, 1)
        self._bv2 = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dq(self, w_int8: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize int8 weights to float32."""
        return w_int8.astype(np.float32) * scale

    def _to_4channel(self, x: np.ndarray) -> np.ndarray:
        """Append k/n constant plane to 3-channel flat encoding.

        Args:
            x: Flat input of length 3*n*n.

        Returns:
            Feature map of shape (4, n, n).
        """
        spatial = x.reshape(3, self.n, self.n)
        k_plane = np.full((1, self.n, self.n), self.k / self.n, dtype=np.float32)
        return np.concatenate([spatial, k_plane], axis=0)

    def _forward_trunk(self, feat: np.ndarray) -> np.ndarray:
        """Run the 10-block conv trunk and return the feature map.

        Args:
            feat: Input of shape (4, n, n).

        Returns:
            Feature map of shape (FILTERS, n, n).
        """
        feat = relu(conv2d(
            feat,
            self._dq(self._conv_init_int8, self._conv_init_scale),
            self._conv_init_b,
            padding=1,
        ))
        for w1, s1, b1, w2, s2, b2 in self._res_blocks:
            skip = feat
            out = relu(conv2d(feat, self._dq(w1, s1), b1, padding=1))
            out = conv2d(out, self._dq(w2, s2), b2, padding=1)
            feat = relu(out + skip)
        return feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass with on-demand int8 dequantisation.

        Args:
            x: Flat board encoding of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits has length n*n and
            value is a scalar in [-1, 1].
        """
        feat = self._forward_trunk(self._to_4channel(x))

        # Policy head
        p_h = relu(conv2d(feat,
                          self._dq(self._conv_p_int8, self._conv_p_scale),
                          self._conv_p_b, padding=0))
        flat_p = p_h.flatten()
        policy_logits = flat_p @ self._dq(self._wp_int8, self._wp_scale) + self._bp

        # Value head
        v_h = relu(conv2d(feat,
                          self._dq(self._conv_v_int8, self._conv_v_scale),
                          self._conv_v_b, padding=0))
        flat_v = v_h.flatten()
        h_v = relu(flat_v @ self._dq(self._wv1_int8, self._wv1_scale) + self._bv1)
        value = float(tanh_act(h_v @ self._dq(self._wv2_int8, self._wv2_scale) + self._bv2)[0])

        return policy_logits, value

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Apply one SGD update over a mini-batch (bias-only).

        Averages dp and dv across the batch and applies an SGD step to
        the float32 policy and value biases. All int8 conv and FC weights
        remain frozen.

        Args:
            batch: List of (state_enc, dp, dv) tuples where:
                state_enc: Flat input of length 3*n*n.
                dp: Gradient w.r.t. policy logits (length n*n).
                dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        g_bp = np.zeros_like(self._bp)
        g_bv2 = np.zeros_like(self._bv2)
        for _, dp, dv in batch:
            g_bp += dp
            g_bv2 += dv
        n_b = len(batch)
        self._bp -= lr * g_bp / n_b
        self._bv2 -= lr * g_bv2 / n_b

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Perform one SGD update for a single example (bias-only).

        Convenience wrapper around backward_batch.

        Args:
            x: Flat input of length 3*n*n.
            dp: Gradient w.r.t. policy logits (length n*n).
            dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    @property
    def memory_usage_mb(self) -> float:
        """Estimate int8 weight memory usage in MB.

        Returns:
            Memory in megabytes (int8 weights + float32 biases).
        """
        int8_bytes = (
            self._conv_init_int8.nbytes
            + sum(w1.nbytes + w2.nbytes for w1, _, _, w2, _, _ in self._res_blocks)
            + self._conv_p_int8.nbytes + self._wp_int8.nbytes
            + self._conv_v_int8.nbytes + self._wv1_int8.nbytes + self._wv2_int8.nbytes
        )
        bias_bytes = (
            self._conv_init_b.nbytes
            + sum(b1.nbytes + b2.nbytes for _, _, b1, _, _, b2 in self._res_blocks)
            + self._conv_p_b.nbytes + self._bp.nbytes
            + self._conv_v_b.nbytes + self._bv1.nbytes + self._bv2.nbytes
        )
        return (int8_bytes + bias_bytes) / (1024 * 1024)

    def copy(self) -> LargeQuantizedPolicyValueNetwork:
        """Return a deep copy of this network.

        Returns:
            New LargeQuantizedPolicyValueNetwork with all weights copied.
        """
        net = LargeQuantizedPolicyValueNetwork.__new__(LargeQuantizedPolicyValueNetwork)
        net.n = self.n
        net.k = self.k
        net._conv_init_int8 = self._conv_init_int8.copy()
        net._conv_init_scale = self._conv_init_scale
        net._conv_init_b = self._conv_init_b.copy()
        net._res_blocks = [
            (w1.copy(), s1, b1.copy(), w2.copy(), s2, b2.copy())
            for w1, s1, b1, w2, s2, b2 in self._res_blocks
        ]
        net._conv_p_int8 = self._conv_p_int8.copy()
        net._conv_p_scale = self._conv_p_scale
        net._conv_p_b = self._conv_p_b.copy()
        net._wp_int8 = self._wp_int8.copy()
        net._wp_scale = self._wp_scale
        net._bp = self._bp.copy()
        net._conv_v_int8 = self._conv_v_int8.copy()
        net._conv_v_scale = self._conv_v_scale
        net._conv_v_b = self._conv_v_b.copy()
        net._wv1_int8 = self._wv1_int8.copy()
        net._wv1_scale = self._wv1_scale
        net._bv1 = self._bv1.copy()
        net._wv2_int8 = self._wv2_int8.copy()
        net._wv2_scale = self._wv2_scale
        net._bv2 = self._bv2.copy()
        return net

    def save(self, path: str) -> None:
        """Save all weights to a .npz file (path without extension).

        Args:
            path: Destination path without .npz extension.
        """
        arrays: dict = {
            'conv_init_int8': self._conv_init_int8,
            'conv_init_scale': self._conv_init_scale,
            'conv_init_b': self._conv_init_b,
            'conv_p_int8': self._conv_p_int8,
            'conv_p_scale': self._conv_p_scale,
            'conv_p_b': self._conv_p_b,
            'wp_int8': self._wp_int8, 'wp_scale': self._wp_scale, 'bp': self._bp,
            'conv_v_int8': self._conv_v_int8,
            'conv_v_scale': self._conv_v_scale,
            'conv_v_b': self._conv_v_b,
            'wv1_int8': self._wv1_int8, 'wv1_scale': self._wv1_scale, 'bv1': self._bv1,
            'wv2_int8': self._wv2_int8, 'wv2_scale': self._wv2_scale, 'bv2': self._bv2,
        }
        for i, (w1, s1, b1, w2, s2, b2) in enumerate(self._res_blocks):
            arrays[f'res{i}_w1'] = w1
            arrays[f'res{i}_s1'] = s1
            arrays[f'res{i}_b1'] = b1
            arrays[f'res{i}_w2'] = w2
            arrays[f'res{i}_s2'] = s2
            arrays[f'res{i}_b2'] = b2
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load weights from a .npz file.

        Args:
            path: Source path with .npz extension.
        """
        data = np.load(path)
        self._conv_init_int8 = data['conv_init_int8']
        self._conv_init_scale = float(data['conv_init_scale'])
        self._conv_init_b = data['conv_init_b']
        self._conv_p_int8 = data['conv_p_int8']
        self._conv_p_scale = float(data['conv_p_scale'])
        self._conv_p_b = data['conv_p_b']
        self._wp_int8 = data['wp_int8']
        self._wp_scale = float(data['wp_scale'])
        self._bp = data['bp']
        self._conv_v_int8 = data['conv_v_int8']
        self._conv_v_scale = float(data['conv_v_scale'])
        self._conv_v_b = data['conv_v_b']
        self._wv1_int8 = data['wv1_int8']
        self._wv1_scale = float(data['wv1_scale'])
        self._bv1 = data['bv1']
        self._wv2_int8 = data['wv2_int8']
        self._wv2_scale = float(data['wv2_scale'])
        self._bv2 = data['bv2']
        self._res_blocks = []
        for i in range(self.NUM_RES_BLOCKS):
            self._res_blocks.append((
                data[f'res{i}_w1'], float(data[f'res{i}_s1']), data[f'res{i}_b1'],
                data[f'res{i}_w2'], float(data[f'res{i}_s2']), data[f'res{i}_b2'],
            ))


# ---------------------------------------------------------------------------
# LargeTernaryPolicyValueNetwork: 20-block ternary conv (AlphaZero depth)
# ---------------------------------------------------------------------------


class LargeTernaryPolicyValueNetwork:
    """AlphaZero-depth ternary network with 20 residual blocks, 256 filters.

    Matches the residual tower depth of Google's original AlphaZero (20 blocks
    for Chess/Shogi). All conv and FC weights are strictly ternary {-1, 0, +1}
    stored as int8 with per-layer float32 scale factors, achieving ~4× memory
    compression versus float32 at the same architecture.

    Architecture:
        Input:   3*n*n flat vector → reshaped to (4, n, n) with k/n plane
        Initial: Conv(4→256, 3×3, same), weights ternary
        Tower:   20 × ResBlock(256 filters), weights ternary
                   Conv(256→256, 3×3) → ReLU → Conv(256→256, 3×3)
                   → add skip → ReLU
        Policy:  Conv(256→2, 1×1, ternary) → ReLU → flatten(2n²)
                 → Linear(2n²→n², ternary)
        Value:   Conv(256→1, 1×1, ternary) → ReLU → flatten(n²)
                 → Linear(n²→256, ternary) → ReLU → Linear(256→1, ternary) → tanh

    Weight precision: int8 ternary {-1, 0, +1} with float32 scale per layer.
    Biases:          float32 (updated via SGD mini-batch).
    Memory (n=3):    ~22.5 MB int8 vs ~90 MB float32 (4× compression).

    Args:
        n: Board dimension.
        k: Winning-run length (default: n). Encoded as k/n in the 4th channel.
    """

    NUM_RES_BLOCKS: int = 20
    FILTERS: int = 256

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Initialise all ternary conv weights and float32 biases."""
        rng = np.random.default_rng(seed=0)
        n = self.n
        C = self.FILTERS

        def ternary_conv(c_in: int, c_out: int, kH: int = 3, kW: int = 3):
            """Random ternary initialisation for conv weights (50% sparsity)."""
            total = c_out * c_in * kH * kW
            weights = np.zeros(total, dtype=np.int8)
            num_nz = int(rng.integers(0, total // 2 + 1))
            indices = rng.choice(total, size=num_nz, replace=False)
            for idx in indices:
                weights[idx] = rng.choice([-1, 1])
            return weights.reshape(c_out, c_in, kH, kW), 1.0

        def ternary_fc(fan_in: int, fan_out: int):
            total = fan_in * fan_out
            weights = np.zeros(total, dtype=np.int8)
            num_nz = int(rng.integers(0, total // 2 + 1))
            indices = rng.choice(total, size=num_nz, replace=False)
            for idx in indices:
                weights[idx] = rng.choice([-1, 1])
            return weights.reshape(fan_in, fan_out), 1.0

        # Initial conv: 4 input channels
        self._conv_init_t, self._conv_init_scale = ternary_conv(4, C)
        self._conv_init_b = np.zeros(C, dtype=np.float32)

        # Residual blocks: list of (w1_t, s1, b1, w2_t, s2, b2)
        self._res_blocks: list = []
        for _ in range(self.NUM_RES_BLOCKS):
            w1, s1 = ternary_conv(C, C)
            w2, s2 = ternary_conv(C, C)
            self._res_blocks.append((
                w1, s1, np.zeros(C, dtype=np.float32),
                w2, s2, np.zeros(C, dtype=np.float32),
            ))

        # Policy head conv + FC
        self._conv_p_t, self._conv_p_scale = ternary_conv(C, 2, kH=1, kW=1)
        self._conv_p_b = np.zeros(2, dtype=np.float32)
        self._wp_t, self._wp_scale = ternary_fc(2 * n * n, n * n)
        self._bp = np.zeros(n * n, dtype=np.float32)

        # Value head conv + 2-layer FC
        self._conv_v_t, self._conv_v_scale = ternary_conv(C, 1, kH=1, kW=1)
        self._conv_v_b = np.zeros(1, dtype=np.float32)
        self._wv1_t, self._wv1_scale = ternary_fc(n * n, 256)
        self._bv1 = np.zeros(256, dtype=np.float32)
        self._wv2_t, self._wv2_scale = ternary_fc(256, 1)
        self._bv2 = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tf(self, w_t: np.ndarray, scale: float) -> np.ndarray:
        """Convert ternary int8 weights to float32.

        Args:
            w_t: Ternary weight array (values in {-1, 0, +1}).
            scale: Per-layer float32 scale factor.

        Returns:
            float32 weight array after applying scale.
        """
        return w_t.astype(np.float32) * scale

    def quantize_to_ternary(self, w_float: np.ndarray) -> np.ndarray:
        """Quantize floating-point weights to ternary {-1, 0, +1}.

        Uses the 50th-percentile of absolute values as the zero-threshold.

        Args:
            w_float: Float32 weight matrix.

        Returns:
            Ternary int8 weight matrix with values in {-1, 0, +1}.
        """
        w_abs = np.abs(w_float)
        threshold = np.percentile(w_abs, 50)
        out = np.zeros_like(w_float, dtype=np.int8)
        out[w_float >= threshold] = 1
        out[w_float < -threshold] = -1
        return out

    def _to_4channel(self, x: np.ndarray) -> np.ndarray:
        """Append k/n constant plane to 3-channel flat encoding.

        Args:
            x: Flat input of length 3*n*n.

        Returns:
            Feature map of shape (4, n, n).
        """
        spatial = x.reshape(3, self.n, self.n)
        k_plane = np.full((1, self.n, self.n), self.k / self.n, dtype=np.float32)
        return np.concatenate([spatial, k_plane], axis=0)

    def _forward_trunk(self, feat: np.ndarray) -> np.ndarray:
        """Run the 20-block ternary conv trunk with RMS normalization.

        Deep ternary towers without batch normalization suffer from exploding
        activations. RMS normalization is applied after each residual block
        (following the BitNet convention of normalizing before linear ops) and
        once more after the full tower to keep activations in a stable range
        for the policy and value heads.

        Args:
            feat: Input of shape (4, n, n).

        Returns:
            Feature map of shape (FILTERS, n, n) with unit RMS magnitude.
        """
        def _rms_norm(x: np.ndarray) -> np.ndarray:
            rms = float(np.sqrt(np.mean(x ** 2)))
            return x / (rms + 1e-8)

        feat = relu(conv2d(
            feat,
            self._tf(self._conv_init_t, self._conv_init_scale),
            self._conv_init_b,
            padding=1,
        ))
        for w1, s1, b1, w2, s2, b2 in self._res_blocks:
            skip = feat
            out = relu(conv2d(_rms_norm(feat), self._tf(w1, s1), b1, padding=1))
            out = conv2d(out, self._tf(w2, s2), b2, padding=1)
            feat = relu(out + skip)
        return _rms_norm(feat)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass with ternary weights.

        Args:
            x: Flat board encoding of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits has length n*n and
            value is a scalar in [-1, 1].
        """
        feat = self._forward_trunk(self._to_4channel(x))

        # Policy head
        p_h = relu(conv2d(feat,
                          self._tf(self._conv_p_t, self._conv_p_scale),
                          self._conv_p_b, padding=0))
        flat_p = p_h.flatten()
        policy_logits = flat_p @ self._tf(self._wp_t, self._wp_scale) + self._bp

        # Value head
        v_h = relu(conv2d(feat,
                          self._tf(self._conv_v_t, self._conv_v_scale),
                          self._conv_v_b, padding=0))
        flat_v = v_h.flatten()
        h_v = relu(flat_v @ self._tf(self._wv1_t, self._wv1_scale) + self._bv1)
        value = float(tanh_act(h_v @ self._tf(self._wv2_t, self._wv2_scale) + self._bv2)[0])

        return policy_logits, value

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Apply one SGD update over a mini-batch (bias-only).

        Averages dp and dv across the batch and applies an SGD step to
        the float32 policy and value biases. All ternary weights remain frozen.

        Args:
            batch: List of (state_enc, dp, dv) tuples where:
                state_enc: Flat input of length 3*n*n.
                dp: Gradient w.r.t. policy logits (length n*n).
                dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        g_bp = np.zeros_like(self._bp)
        g_bv2 = np.zeros_like(self._bv2)
        for _, dp, dv in batch:
            g_bp += dp
            g_bv2 += dv
        n_b = len(batch)
        self._bp -= lr * g_bp / n_b
        self._bv2 -= lr * g_bv2 / n_b

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Perform one SGD update for a single example (bias-only).

        Convenience wrapper around backward_batch.

        Args:
            x: Flat input of length 3*n*n.
            dp: Gradient w.r.t. policy logits (length n*n).
            dv: Gradient w.r.t. value output (length 1).
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    def estimate_memory_bytes(self) -> dict[str, int]:
        """Estimate memory usage broken down by component type.

        Returns:
            Dictionary with keys: ternary_weights_bytes, scale_factors_bytes,
            biases_bytes, total_bytes, total_ternary_params.
        """
        C = self.FILTERS
        n = self.n
        # Count all ternary weight arrays
        total_t = (
            self._conv_init_t.size
            + sum(w1.size + w2.size for w1, _, _, w2, _, _ in self._res_blocks)
            + self._conv_p_t.size + self._wp_t.size
            + self._conv_v_t.size + self._wv1_t.size + self._wv2_t.size
        )
        ternary_bytes = total_t  # 1 byte per int8
        # Scale factors: 2 per block + initial + heads
        num_scales = 2 * self.NUM_RES_BLOCKS + 6
        scale_bytes = num_scales * 4
        # Biases
        bias_bytes = (
            self._conv_init_b.nbytes
            + sum(b1.nbytes + b2.nbytes for _, _, b1, _, _, b2 in self._res_blocks)
            + self._conv_p_b.nbytes + self._bp.nbytes
            + self._conv_v_b.nbytes + self._bv1.nbytes + self._bv2.nbytes
        )
        total_bytes = ternary_bytes + scale_bytes + bias_bytes
        return {
            "ternary_weights_bytes": ternary_bytes,
            "scale_factors_bytes": scale_bytes,
            "biases_bytes": bias_bytes,
            "total_bytes": total_bytes,
            "total_ternary_params": total_t,
        }

    def compare_memory_to_float32(self) -> dict[str, float]:
        """Compare ternary memory usage to float32 equivalent.

        Returns:
            Dictionary with float32_total_kb, ternary_total_kb,
            reduction_ratio, savings_percent.
        """
        stats = self.estimate_memory_bytes()
        total_t = stats["total_ternary_params"]
        float32_total = total_t * 4 + stats["biases_bytes"]
        ternary_total = stats["total_bytes"]
        return {
            "float32_total_kb": float32_total / 1024,
            "ternary_total_kb": ternary_total / 1024,
            "reduction_ratio": float32_total / ternary_total,
            "savings_percent": 100.0 * (1.0 - ternary_total / float32_total),
        }

    def copy(self) -> LargeTernaryPolicyValueNetwork:
        """Return a deep copy of this network.

        Returns:
            New LargeTernaryPolicyValueNetwork with all weights copied.
        """
        net = LargeTernaryPolicyValueNetwork.__new__(LargeTernaryPolicyValueNetwork)
        net.n = self.n
        net.k = self.k
        net._conv_init_t = self._conv_init_t.copy()
        net._conv_init_scale = self._conv_init_scale
        net._conv_init_b = self._conv_init_b.copy()
        net._res_blocks = [
            (w1.copy(), s1, b1.copy(), w2.copy(), s2, b2.copy())
            for w1, s1, b1, w2, s2, b2 in self._res_blocks
        ]
        net._conv_p_t = self._conv_p_t.copy()
        net._conv_p_scale = self._conv_p_scale
        net._conv_p_b = self._conv_p_b.copy()
        net._wp_t = self._wp_t.copy()
        net._wp_scale = self._wp_scale
        net._bp = self._bp.copy()
        net._conv_v_t = self._conv_v_t.copy()
        net._conv_v_scale = self._conv_v_scale
        net._conv_v_b = self._conv_v_b.copy()
        net._wv1_t = self._wv1_t.copy()
        net._wv1_scale = self._wv1_scale
        net._bv1 = self._bv1.copy()
        net._wv2_t = self._wv2_t.copy()
        net._wv2_scale = self._wv2_scale
        net._bv2 = self._bv2.copy()
        return net

    def save(self, path: str) -> None:
        """Save all weights to a .npz file (path without extension).

        Args:
            path: Destination path without .npz extension.
        """
        arrays: dict = {
            'conv_init_t': self._conv_init_t,
            'conv_init_scale': self._conv_init_scale,
            'conv_init_b': self._conv_init_b,
            'conv_p_t': self._conv_p_t,
            'conv_p_scale': self._conv_p_scale,
            'conv_p_b': self._conv_p_b,
            'wp_t': self._wp_t, 'wp_scale': self._wp_scale, 'bp': self._bp,
            'conv_v_t': self._conv_v_t,
            'conv_v_scale': self._conv_v_scale,
            'conv_v_b': self._conv_v_b,
            'wv1_t': self._wv1_t, 'wv1_scale': self._wv1_scale, 'bv1': self._bv1,
            'wv2_t': self._wv2_t, 'wv2_scale': self._wv2_scale, 'bv2': self._bv2,
        }
        for i, (w1, s1, b1, w2, s2, b2) in enumerate(self._res_blocks):
            arrays[f'res{i}_w1'] = w1
            arrays[f'res{i}_s1'] = s1
            arrays[f'res{i}_b1'] = b1
            arrays[f'res{i}_w2'] = w2
            arrays[f'res{i}_s2'] = s2
            arrays[f'res{i}_b2'] = b2
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load weights from a .npz file.

        Args:
            path: Source path with .npz extension.
        """
        data = np.load(path)
        self._conv_init_t = data['conv_init_t']
        self._conv_init_scale = float(data['conv_init_scale'])
        self._conv_init_b = data['conv_init_b']
        self._conv_p_t = data['conv_p_t']
        self._conv_p_scale = float(data['conv_p_scale'])
        self._conv_p_b = data['conv_p_b']
        self._wp_t = data['wp_t']
        self._wp_scale = float(data['wp_scale'])
        self._bp = data['bp']
        self._conv_v_t = data['conv_v_t']
        self._conv_v_scale = float(data['conv_v_scale'])
        self._conv_v_b = data['conv_v_b']
        self._wv1_t = data['wv1_t']
        self._wv1_scale = float(data['wv1_scale'])
        self._bv1 = data['bv1']
        self._wv2_t = data['wv2_t']
        self._wv2_scale = float(data['wv2_scale'])
        self._bv2 = data['bv2']
        self._res_blocks = []
        for i in range(self.NUM_RES_BLOCKS):
            self._res_blocks.append((
                data[f'res{i}_w1'], float(data[f'res{i}_s1']), data[f'res{i}_b1'],
                data[f'res{i}_w2'], float(data[f'res{i}_s2']), data[f'res{i}_b2'],
            ))


# ---------------------------------------------------------------------------
# QNetwork: action-value network for DQN
# ---------------------------------------------------------------------------


class QNetwork:
    """Q-network: maps board state to action-values for each cell.

    Architecture:
        Input:  3*n*n + 1 flat vector (k/n appended as extra scalar)
        Hidden: Linear(3n²+1, 128) → ReLU → Linear(128, 64) → ReLU
        Output: Linear(64, n²)  (Q-values for each cell)
    """

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        input_size = 3 * n * n + 1
        output_size = n * n
        rng = np.random.default_rng(seed=0)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        self._w1 = he(input_size, 128)
        self._b1 = np.zeros(128, dtype=np.float32)
        self._w2 = he(128, 64)
        self._b2 = np.zeros(64, dtype=np.float32)
        self._w3 = he(64, output_size)
        self._b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Flat input vector of length 3*n*n.

        Returns:
            Q-value array of length n*n.
        """
        x = np.append(x, np.float32(self.k / self.n))
        h1 = relu(x @ self._w1 + self._b1)
        h2 = relu(h1 @ self._w2 + self._b2)
        return h2 @ self._w3 + self._b3

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays in order."""
        return [self._w1, self._b1, self._w2, self._b2, self._w3, self._b3]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set weights from list (same order as get_weights)."""
        self._w1, self._b1, self._w2, self._b2, self._w3, self._b3 = weights

    def copy(self) -> QNetwork:
        """Return a deep copy of this network."""
        net = QNetwork.__new__(QNetwork)
        net.n = self.n
        net.k = self.k
        net._w1 = self._w1.copy()
        net._b1 = self._b1.copy()
        net._w2 = self._w2.copy()
        net._b2 = self._b2.copy()
        net._w3 = self._w3.copy()
        net._b3 = self._b3.copy()
        return net

    def save(self, path: str) -> None:
        """Save weights to a .npz file."""
        np.savez(path, w1=self._w1, b1=self._b1,
                 w2=self._w2, b2=self._b2,
                 w3=self._w3, b3=self._b3)

    def load(self, path: str) -> None:
        """Load weights from a .npz file."""
        data = np.load(path)
        self._w1 = data['w1']
        self._b1 = data['b1']
        self._w2 = data['w2']
        self._b2 = data['b2']
        self._w3 = data['w3']
        self._b3 = data['b3']


# ---------------------------------------------------------------------------
# BitLinear: 1.58-bit linear layer with STE training
# ---------------------------------------------------------------------------

class BitLinear:
    """BitNet 1.58-bit linear layer (no bias).

    Forward pass applies:
    1. Absmean ternary weight quantization: W_q = RoundClip(W/alpha, -1, 1).
    2. Per-token absmax 8-bit activation quantization on the input.
    3. Matrix multiply: out = x_q @ W_q.T * alpha  (effectively, since W_q
       is used as float32 after dequantization).

    Backward uses the Straight-Through Estimator (STE): gradients flow
    through both quantization operations unchanged. Latent float32 weights
    are updated via Adam.

    Training workflow (mini-batch):
    - Call zero_grad() once before the batch.
    - For each example: call accum_grad(d_out, x) → accumulates dW, returns d_x.
    - After all examples: call apply_adam(lr) → one Adam step, resets accumulator.
    - For single-example convenience: backward(d_out, x, lr) = accum_grad + apply_adam.
    """

    def __init__(self, fan_in: int, fan_out: int, rng=None) -> None:
        if rng is None:
            rng = np.random.default_rng(0)
        std = float(np.sqrt(2.0 / fan_in))
        self._w: np.ndarray = rng.standard_normal(
            (fan_in, fan_out)
        ).astype(np.float32) * std
        self._m: np.ndarray = np.zeros((fan_in, fan_out), dtype=np.float32)
        self._v: np.ndarray = np.zeros((fan_in, fan_out), dtype=np.float32)
        self._t: int = 0
        self._dw_accum: np.ndarray = np.zeros((fan_in, fan_out), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Quantized forward pass.

        Args:
            x: Input array of shape (..., fan_in).

        Returns:
            Output array of shape (..., fan_out) in float32.
        """
        w_q, alpha = absmean_quantize_weights(self._w)
        w_fp = w_q.astype(np.float32) * alpha
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        x_q, _ = absmax_quantize_activations(x2d)
        out = x_q @ w_fp
        return out.reshape(*shape[:-1], w_fp.shape[-1]).astype(np.float32)

    def zero_grad(self) -> None:
        """Reset the gradient accumulator to zero."""
        self._dw_accum[:] = 0.0

    def accum_grad(self, d_out: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Accumulate weight gradient and return input gradient (STE).

        Treats both quantization operations as identity for gradient flow:
            dW += x.T @ d_out
            d_x  = d_out @ W_latent.T

        Args:
            d_out: Upstream gradient, shape (..., fan_out).
            x: Corresponding input that was used in forward, shape (..., fan_in).

        Returns:
            Gradient w.r.t. x, same shape as x.
        """
        shape = x.shape
        d2 = d_out.reshape(-1, d_out.shape[-1])   # (N, fan_out)
        x2 = x.reshape(-1, shape[-1])              # (N, fan_in)
        self._dw_accum += x2.T @ d2                # accumulate (fan_in, fan_out)
        d_x = d2 @ self._w.T                       # STE: use latent weights
        return d_x.reshape(shape).astype(np.float32)

    def apply_adam(self, lr: float) -> None:
        """Apply one Adam step using the accumulated gradient, then reset.

        Args:
            lr: Learning rate.
        """
        self._t += 1
        d_w = self._dw_accum
        self._m = 0.9 * self._m + 0.1 * d_w
        self._v = 0.999 * self._v + 0.001 * (d_w ** 2)
        m_hat = self._m / (1.0 - 0.9 ** self._t)
        v_hat = self._v / (1.0 - 0.999 ** self._t)
        self._w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self._dw_accum[:] = 0.0

    def backward(self, d_out: np.ndarray, x: np.ndarray, lr: float) -> np.ndarray:
        """Single-example convenience: accum_grad + apply_adam.

        Args:
            d_out: Upstream gradient, shape (..., fan_out).
            x: Input used in forward, shape (..., fan_in).
            lr: Learning rate.

        Returns:
            Gradient w.r.t. x.
        """
        d_x = self.accum_grad(d_out, x)
        self.apply_adam(lr)
        return d_x

    def copy(self) -> "BitLinear":
        """Return a deep copy of this layer (including Adam state)."""
        other = BitLinear.__new__(BitLinear)
        other._w = self._w.copy()
        other._m = self._m.copy()
        other._v = self._v.copy()
        other._t = self._t
        other._dw_accum = np.zeros_like(self._w)
        return other

    def state_dict(self, prefix: str) -> dict:
        """Serialise to a flat dict of arrays (for np.savez)."""
        return {
            f"{prefix}_w": self._w,
            f"{prefix}_m": self._m,
            f"{prefix}_v": self._v,
            f"{prefix}_t": np.array(self._t, dtype=np.int64),
        }

    def load_state_dict(self, d: dict, prefix: str) -> None:
        """Load from a flat dict produced by state_dict."""
        self._w = d[f"{prefix}_w"].astype(np.float32)
        self._m = d[f"{prefix}_m"].astype(np.float32)
        self._v = d[f"{prefix}_v"].astype(np.float32)
        self._t = int(d[f"{prefix}_t"])
        self._dw_accum = np.zeros_like(self._w)


# ---------------------------------------------------------------------------
# BitMultiHeadAttention: full MHA with BitLinear Q/K/V/O and RoPE
# ---------------------------------------------------------------------------

class BitMultiHeadAttention:
    """Multi-head attention using BitLinear projections and RoPE.

    All board tokens attend to all other board tokens (no causal mask).
    SubLN (rms_norm_bitnet) is applied by the enclosing BitTransformerBlock
    before this module receives its input.
    """

    def __init__(self, d_model: int, n_heads: int, rng) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = BitLinear(d_model, d_model, rng)
        self.wk = BitLinear(d_model, d_model, rng)
        self.wv = BitLinear(d_model, d_model, rng)
        self.wo = BitLinear(d_model, d_model, rng)
        self._cos, self._sin = rope_frequencies(self.d_head)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Saves intermediate tensors for backward.

        Args:
            x: (T, d_model) — input after SubLN.

        Returns:
            (T, d_model) attention output.
        """
        T, d = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.wq.forward(x).reshape(T, H, dh)   # (T, H, dh)
        k = self.wk.forward(x).reshape(T, H, dh)
        v = self.wv.forward(x).reshape(T, H, dh)

        q_rot, k_rot = rope_apply(q, k, self._cos, self._sin)

        scale = float(1.0 / np.sqrt(dh))
        # scores: (H, T, T); attn over source positions s for each query t
        scores = np.einsum("thd,shd->hts", q_rot, k_rot) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9)  # (H, T, T)

        # context: weighted sum of values
        context = np.einsum("hts,shd->thd", attn, v).reshape(T, d)  # (T, d)
        out = self.wo.forward(context)  # (T, d)

        self._saved = (x, q_rot, k_rot, v, attn, context)
        return out.astype(np.float32)

    def backward_accum(self, d_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients for all BitLinears; return d_x.

        Args:
            d_out: Upstream gradient, shape (T, d_model).

        Returns:
            Gradient w.r.t. x (the SubLN-normalised block input), shape (T, d_model).
        """
        x, q_rot, k_rot, v, attn, context = self._saved
        T, d = x.shape
        H, dh = self.n_heads, self.d_head
        scale = float(1.0 / np.sqrt(dh))

        # 1. Wo backward: d_context = gradient w.r.t. context
        d_context = self.wo.accum_grad(d_out, context)          # (T, d)

        # 2. Context = einsum(attn, v) backward
        d_context_h = d_context.reshape(T, H, dh)
        # d_attn[h,t,s] = sum_d d_context[t,h,d] * v[s,h,d]
        d_attn = np.einsum("thd,shd->hts", d_context_h, v)     # (H, T, T)
        # d_v[s,h,d] = sum_t attn[h,t,s] * d_context[t,h,d]
        d_v_thd = np.einsum("hts,thd->shd", attn, d_context_h) # (T, H, dh)

        # 3. Softmax backward: d_scores[h,t,s] = attn*(d_attn - dot(d_attn,attn))
        d_scores = attn * (
            d_attn - (d_attn * attn).sum(axis=-1, keepdims=True)
        ) * scale   # (H, T, T)

        # 4. Scores = einsum(q_rot, k_rot) backward
        # d_q_rot[t,h,d] = sum_s d_scores[h,t,s] * k_rot[s,h,d]
        d_q_rot = np.einsum("hts,shd->thd", d_scores, k_rot)   # (T, H, dh)
        # d_k_rot[s,h,d] = sum_t d_scores[h,t,s] * q_rot[t,h,d]
        d_k_rot = np.einsum("hts,thd->shd", d_scores, q_rot)   # (T, H, dh)

        # 5. RoPE backward (orthogonal inverse)
        d_q, d_k = rope_apply_backward(d_q_rot, d_k_rot, self._cos, self._sin)

        # 6. Wq, Wk, Wv backward: accumulate grads, sum d_x contributions
        d_x = self.wq.accum_grad(d_q.reshape(T, d), x)
        d_x += self.wk.accum_grad(d_k.reshape(T, d), x)
        d_x += self.wv.accum_grad(d_v_thd.reshape(T, d), x)
        return d_x.astype(np.float32)


# ---------------------------------------------------------------------------
# BitFFN: feed-forward sublayer with squared ReLU
# ---------------------------------------------------------------------------

class BitFFN:
    """BitNet FFN: BitLinear(up) → squared_relu → BitLinear(down). No bias.

    Uses squared ReLU (ReLU²) as specified by BitNet b1.58.
    """

    def __init__(self, d_model: int, d_ff: int, rng) -> None:
        self.up = BitLinear(d_model, d_ff, rng)
        self.down = BitLinear(d_ff, d_model, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (T, d_model) → (T, d_model). Saves intermediates."""
        h = self.up.forward(x)       # (T, d_ff)
        h_act = squared_relu(h)      # (T, d_ff)
        out = self.down.forward(h_act)  # (T, d_model)
        self._saved = (x, h, h_act)
        return out.astype(np.float32)

    def backward_accum(self, d_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients; return d_x.

        Args:
            d_out: Upstream gradient, shape (T, d_model).

        Returns:
            Gradient w.r.t. x, shape (T, d_model).
        """
        x, h, h_act = self._saved
        # Backward through down
        d_h_act = self.down.accum_grad(d_out, h_act)   # (T, d_ff)
        # Backward through squared_relu: f'(x) = 2 * relu(x) = 2 * max(0, h)
        d_h = d_h_act * 2.0 * np.maximum(0.0, h)
        # Backward through up
        d_x = self.up.accum_grad(d_h, x)               # (T, d_model)
        return d_x.astype(np.float32)


# ---------------------------------------------------------------------------
# BitTransformerBlock: SubLN + MHA + SubLN + FFN with residual connections
# ---------------------------------------------------------------------------

class BitTransformerBlock:
    """One BitNet Transformer block.

    Structure (SubLN style — RMSNorm before each sublayer):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    No learnable parameters in norm layers.  No biases anywhere.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng) -> None:
        self.attn = BitMultiHeadAttention(d_model, n_heads, rng)
        self.ffn = BitFFN(d_model, d_ff, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (T, d_model) → (T, d_model). Saves pre-sublayer inputs."""
        x_in = x
        x_norm1 = rms_norm_bitnet(x_in)
        attn_out = self.attn.forward(x_norm1)
        x_after_attn = x_in + attn_out

        x_norm2 = rms_norm_bitnet(x_after_attn)
        ffn_out = self.ffn.forward(x_norm2)
        x_out = x_after_attn + ffn_out

        self._saved_fwd = (x_in, x_after_attn)
        return x_out.astype(np.float32)

    def backward_accum(self, d_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients through this block; return d_x_in.

        Args:
            d_out: Upstream gradient w.r.t. block output, shape (T, d_model).

        Returns:
            Gradient w.r.t. block input x_in, shape (T, d_model).
        """
        x_in, x_after_attn = self._saved_fwd

        # -- FFN sublayer (residual: d_out passes through + via FFN) --
        d_x_after_attn = d_out.copy()          # residual branch gradient
        # Backward through FFN (uses saved x_norm2 = rms_norm(x_after_attn))
        x_norm2 = rms_norm_bitnet(x_after_attn)
        d_ffn_norm = self.ffn.backward_accum(d_out)   # grad w.r.t. x_norm2
        # Backward through SubLN(x_after_attn)
        d_x_after_attn += rms_norm_bitnet_backward(x_after_attn, d_ffn_norm)

        # -- Attention sublayer (residual: d_x_after_attn passes through + via attn) --
        d_x_in = d_x_after_attn.copy()         # residual branch gradient
        # Backward through attention (uses saved x_norm1 = rms_norm(x_in))
        x_norm1 = rms_norm_bitnet(x_in)
        d_attn_norm = self.attn.backward_accum(d_x_after_attn)  # grad w.r.t. x_norm1
        # Backward through SubLN(x_in)
        d_x_in += rms_norm_bitnet_backward(x_in, d_attn_norm)

        return d_x_in.astype(np.float32)


# ---------------------------------------------------------------------------
# TernaryBitNetPolicyValueNetwork: small BitNet Transformer (4 layers, d=64)
# ---------------------------------------------------------------------------

class TernaryBitNetPolicyValueNetwork:
    """BitNet 1.58-bit Transformer policy-value network (small variant).

    Treats the board as a sequence of n² position tokens, each described by
    4 features (own pieces, opponent pieces, constant 1.0 plane, k/n scalar).
    A BitLinear embedding projects these to d_model dimensions, followed by
    ``n_layers`` BitTransformerBlocks.

    Architecture:
        Input: 3*n*n flat → reshape (n², 3) → append k/n → (n², 4)
        Embedding: BitLinear(4, 64)
        4 × BitTransformerBlock(d_model=64, n_heads=4, d_ff=128)
        Final RMSNorm (no learnable params)
        Policy head : BitLinear(64, 1) per token → (n²,) logits
        Value head  : mean-pool → BitLinear(64, 1) → tanh → scalar

    Quantization: W1.58A8 — absmean ternary weights, per-token absmax int8
    activations throughout.  No biases anywhere.

    Training: full STE-based Adam on all BitLinear latent float32 weights.
    mini-batch via backward_batch(); single-example via backward().

    Memory (n=3): ~few hundred KB (float32 latent weights + Adam state ×3).
    Compressed forward footprint: ternary weights with int8 activations.
    """

    _N_LAYERS: int = 4
    _D_MODEL:  int = 64
    _N_HEADS:  int = 4
    _D_FF:     int = 128

    def __init__(self, n: int, k: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        rng = np.random.default_rng(seed=42)
        d = self._D_MODEL
        self._embed = BitLinear(4, d, rng)
        self._blocks: list[BitTransformerBlock] = [
            BitTransformerBlock(d, self._N_HEADS, self._D_FF, rng)
            for _ in range(self._N_LAYERS)
        ]
        self._policy_head = BitLinear(d, 1, rng)
        self._value_head  = BitLinear(d, 1, rng)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_input(self, x: np.ndarray) -> np.ndarray:
        """Convert flat (3n²,) board encoding to token matrix (n², 4).

        The first three features per token are the three board-encoding
        channels stacked column-wise; the fourth is the k/n constant.

        Args:
            x: Flat float32 array of length 3*n*n.

        Returns:
            Token matrix of shape (n², 4) in float32.
        """
        n2 = self.n * self.n
        # x shape: (3*n², )  — channels stacked in order [ch0*n², ch1*n², ch2*n²]
        ch = x.reshape(3, n2).T     # (n², 3)
        kn = np.full((n2, 1), self.k / self.n, dtype=np.float32)
        return np.concatenate([ch, kn], axis=-1)   # (n², 4)

    def _all_bitlinears(self) -> list[BitLinear]:
        """Return all BitLinear instances in forward-pass order."""
        bls: list[BitLinear] = [self._embed]
        for blk in self._blocks:
            bls += [
                blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo,
                blk.ffn.up, blk.ffn.down,
            ]
        bls += [self._policy_head, self._value_head]
        return bls

    def _layer_prefixes(self) -> list[str]:
        """Prefixes matching _all_bitlinears() order, used for save/load."""
        keys: list[str] = ["embed"]
        for i in range(self._N_LAYERS):
            for s in ("attn_wq", "attn_wk", "attn_wv", "attn_wo",
                      "ffn_up", "ffn_down"):
                keys.append(f"b{i}_{s}")
        keys += ["policy_head", "value_head"]
        return keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass.

        Args:
            x: Flat input of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits has length n*n and
            value is a scalar in [-1, 1].
        """
        tokens = self._encode_input(x)              # (n², 4)
        h = self._embed.forward(tokens)             # (n², d_model)
        for blk in self._blocks:
            h = blk.forward(h)
        self._h_pre_norm = h                        # save before final norm
        h = rms_norm_bitnet(h)                      # final SubLN
        self._h_final = h                           # save for backward

        policy_logits = self._policy_head.forward(h).squeeze(-1)  # (n²,)
        mean_h = h.mean(axis=0, keepdims=True)      # (1, d_model)
        value_logit = self._value_head.forward(mean_h)[0, 0]
        value = float(np.tanh(value_logit))
        self._value_logit = value_logit
        return policy_logits.astype(np.float32), value

    def _backward_accum(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray
    ) -> None:
        """Run one forward pass and accumulate gradients (no Adam step).

        Called by backward_batch for each example before apply_adam.

        Args:
            x: Flat input of length 3*n*n.
            dp: Gradient w.r.t. policy logits, shape (n²,).
            dv: Gradient w.r.t. value scalar, shape (1,) or scalar.
        """
        # Forward refreshes saved state in all modules and sets _h_pre_norm / _h_final
        policy_logits, value = self.forward(x)
        h_final = self._h_final                     # (n², d_model)
        h_pre_norm = self._h_pre_norm               # (n², d_model) — before final norm
        n2, d = h_final.shape
        tokens = self._encode_input(x)              # (n², 4)

        # -- Output gradients --
        d_policy = dp.reshape(n2, 1)                # (n², 1) grad w.r.t. policy_head output
        dv_scalar = float(np.ravel(dv)[0])
        # tanh backward: d/d(logit) = dv * (1 - tanh²)
        d_value_logit = dv_scalar * (1.0 - value ** 2)

        # -- Policy head backward --
        d_h_policy = self._policy_head.accum_grad(
            d_policy, h_final
        )                                           # (n², d_model)

        # -- Value head backward --
        mean_h = h_final.mean(axis=0, keepdims=True)  # (1, d_model)
        d_mean = np.array([[d_value_logit]], dtype=np.float32)  # (1, 1)
        d_h_value_mean = self._value_head.accum_grad(
            d_mean, mean_h
        )                                           # (1, d_model)
        # Backward through mean-pool: distribute gradient equally to all tokens
        d_h_value = np.repeat(d_h_value_mean, n2, axis=0) / n2  # (n², d_model)

        # Combined gradient w.r.t. h_final (= rms_norm output)
        d_h = d_h_policy + d_h_value               # (n², d_model)

        # -- Backward through final RMSNorm --
        # h_final = rms_norm_bitnet(h_pre_norm); h_pre_norm saved in forward.
        d_h_blocks = rms_norm_bitnet_backward(h_pre_norm, d_h)

        # -- Backward through Transformer blocks (reverse order) --
        d_blk = d_h_blocks
        for blk in reversed(self._blocks):
            d_blk = blk.backward_accum(d_blk)

        # -- Backward through embedding --
        self._embed.accum_grad(d_blk, tokens)

    def backward_batch(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Mini-batch update: accumulate gradients over all examples, one Adam step.

        Args:
            batch: List of (x, dp, dv) tuples.
            lr: Learning rate.
        """
        all_bls = self._all_bitlinears()
        for bl in all_bls:
            bl.zero_grad()
        for x, dp, dv in batch:
            self._backward_accum(x, dp, dv)
        for bl in all_bls:
            bl.apply_adam(lr)

    def backward(
        self, x: np.ndarray, dp: np.ndarray, dv: np.ndarray, lr: float
    ) -> None:
        """Single-example update (delegates to backward_batch).

        Args:
            x: Flat input of length 3*n*n.
            dp: Gradient w.r.t. policy logits.
            dv: Gradient w.r.t. value scalar.
            lr: Learning rate.
        """
        self.backward_batch([(x, dp, dv)], lr)

    def copy(self) -> "TernaryBitNetPolicyValueNetwork":
        """Return a deep copy of this network."""
        other = TernaryBitNetPolicyValueNetwork.__new__(
            TernaryBitNetPolicyValueNetwork
        )
        other.n = self.n
        other.k = self.k
        other._embed = self._embed.copy()
        other._blocks = []
        for blk in self._blocks:
            new_blk = BitTransformerBlock.__new__(BitTransformerBlock)
            new_attn = BitMultiHeadAttention.__new__(BitMultiHeadAttention)
            new_attn.d_model = blk.attn.d_model
            new_attn.n_heads = blk.attn.n_heads
            new_attn.d_head  = blk.attn.d_head
            new_attn.wq = blk.attn.wq.copy()
            new_attn.wk = blk.attn.wk.copy()
            new_attn.wv = blk.attn.wv.copy()
            new_attn.wo = blk.attn.wo.copy()
            new_attn._cos = blk.attn._cos
            new_attn._sin = blk.attn._sin
            new_ffn = BitFFN.__new__(BitFFN)
            new_ffn.up   = blk.ffn.up.copy()
            new_ffn.down = blk.ffn.down.copy()
            new_blk.attn = new_attn
            new_blk.ffn  = new_ffn
            other._blocks.append(new_blk)
        other._policy_head = self._policy_head.copy()
        other._value_head  = self._value_head.copy()
        return other

    def save(self, path: str) -> None:
        """Save all latent weights and Adam states to a .npz file."""
        arrays: dict = {}
        for bl, prefix in zip(self._all_bitlinears(), self._layer_prefixes()):
            arrays.update(bl.state_dict(prefix))
        arrays["meta_n"] = np.array(self.n)
        arrays["meta_k"] = np.array(self.k)
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load weights from a .npz file produced by save()."""
        d = dict(np.load(path))
        for bl, prefix in zip(self._all_bitlinears(), self._layer_prefixes()):
            bl.load_state_dict(d, prefix)


# ---------------------------------------------------------------------------
# LargeTernaryBitNetPolicyValueNetwork: large BitNet Transformer (12 layers, d=256)
# ---------------------------------------------------------------------------

class LargeTernaryBitNetPolicyValueNetwork(TernaryBitNetPolicyValueNetwork):
    """BitNet 1.58-bit Transformer policy-value network (large variant).

    Identical architecture to TernaryBitNetPolicyValueNetwork but with
    more layers and wider hidden dimension, matching AlphaZero scale:
        12 Transformer layers, d_model=256, n_heads=8, d_ff=512.

    All other details (quantization, RoPE, SubLN, squared ReLU, no bias,
    STE training) are inherited from the small variant.

    Memory (n=3): ~few tens of MB (float32 latent weights + Adam state ×3).
    """

    _N_LAYERS: int = 12
    _D_MODEL:  int = 256
    _N_HEADS:  int = 8
    _D_FF:     int = 512

    def copy(self) -> "LargeTernaryBitNetPolicyValueNetwork":
        """Return a deep copy of this network."""
        base = super().copy()
        # Re-wrap as LargeTernaryBitNetPolicyValueNetwork
        base.__class__ = LargeTernaryBitNetPolicyValueNetwork
        return base  # type: ignore[return-value]
