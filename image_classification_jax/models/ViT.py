from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


init_fn = lambda dim: nn.initializers.normal(jnp.sqrt(2 / (5 * dim)))
wang_fn = lambda dim, n_layers: nn.initializers.normal(2 / n_layers / jnp.sqrt(dim))


def _dot_product_attention_core(query, key, value):
    head_dim = query.shape[-1]
    query *= jax.lax.rsqrt(jnp.array(head_dim, dtype=jnp.float32)).astype(query.dtype)
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)
    logits = jnp.tanh(logits / 50) * 50
    probs = jax.nn.softmax(logits.astype(jnp.float32)).astype(logits.dtype)
    encoded = jnp.einsum("BNTS,BSNH->BTNH", probs, value)
    return encoded


def _sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, bfloat16 rounding is catastrophic.
    sinusoid_inp = jnp.einsum(
        "i,j->ij",
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def _apply_rotary_embedding(q, k, cos, sin):
    # come in as (B, T, K, G, H) and (B, T, K, H)
    qlen = q.shape[-4]
    klen = k.shape[-3]

    qcos = jnp.expand_dims(cos[:qlen, :], range(len(q.shape) - 2))
    qsin = jnp.expand_dims(sin[:qlen, :], range(len(q.shape) - 2))
    kcos = jnp.expand_dims(cos[:klen, :], range(len(k.shape) - 2))
    ksin = jnp.expand_dims(sin[:klen, :], range(len(k.shape) - 2))

    qcos = jnp.swapaxes(qcos, -2, -4)
    qsin = jnp.swapaxes(qsin, -2, -4)
    kcos = jnp.swapaxes(kcos, -2, -3)
    ksin = jnp.swapaxes(ksin, -2, -3)

    # done in float32
    out_q = q * qcos + _rotate_half(q) * qsin
    out_k = k * kcos + _rotate_half(k) * ksin

    return out_q.astype(q.dtype), out_k.astype(k.dtype)


class Attention(nn.Module):
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    n_layers: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        N = self.num_heads
        K = self.num_kv_heads
        G = N // K
        H = self.head_dim

        q_params = self.param("q_kernel", init_fn(C), (C, N * H))
        k_params = self.param("k_kernel", init_fn(C), (C, K * H))
        v_params = self.param("v_kernel", init_fn(C), (C, K * H))
        out_params = self.param("out_kernel", wang_fn(N * H, self.n_layers), (N * H, C))

        q = jnp.dot(x, q_params)
        k = jnp.dot(x, k_params)
        v = jnp.dot(x, v_params)

        q = jnp.reshape(q, (B, T, K, G, H))
        k = jnp.reshape(k, (B, T, K, H))
        v = jnp.reshape(v, (B, T, K, H))

        sin, cos = _sine_table(H, T, max_timescale=self.rope_theta)
        q, k = _apply_rotary_embedding(q, k, cos, sin)

        vmapped_fn = jax.vmap(
            _dot_product_attention_core, in_axes=(3, None, None), out_axes=3
        )
        encoded = vmapped_fn(q, k, v)
        encoded = jnp.reshape(encoded, (B, T, N * H))
        out = jnp.dot(encoded, out_params)
        out = nn.LayerNorm(use_bias=False)(out)  # normformer
        return out


class MLP(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", init_fn(C), (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", init_fn(C), (C, self.hidden_dim))
        down_kernel = self.param(
            "down_kernel", wang_fn(self.hidden_dim, self.n_layers), (self.hidden_dim, C)
        )

        gate = jnp.dot(x, gate_kernel)
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
        x = gate * up

        x = nn.LayerNorm(use_bias=False)(x)  # normformer

        down = jnp.dot(x, down_kernel)
        return down


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_dim: int
    rope_theta: float
    n_layers: int

    @nn.compact
    def __call__(self, x):
        attn_layer = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.n_layers,
            self.mesh,
        )
        x += attn_layer(nn.LayerNorm(use_bias=False)(x))
        x += MLP(self.hidden_dim, self.n_layers, self.mesh)(nn.LayerNorm(use_bias=False)(x))
        return x, None


class Transformer(nn.Module):
    n_layers: int = 12
    enc_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    output_dim: int = 1000

    @nn.compact
    def __call__(self, x, is_training: bool):
        B, H, W, C = x.shape
        assert C % self.n_heads == 0
        head_dim = C // self.n_heads

        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=16, p2=16)
        x = nn.Dense(features=self.enc_dim, kernel_init=init_fn(x.shape[-1]), use_bias=False)(x)
        x *= jnp.sqrt(self.enc_dim).astype(x.dtype)

        BlockModule = nn.remat(
            Block,
            prevent_cse=False,
            policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        )
        x, _ = nn.scan(
            BlockModule,
            variable_axes={True: 0},
            split_rngs={True: True},
            length=self.n_layers,
        )(
            self.n_heads,
            self.n_kv_heads,
            head_dim,
            self.hidden_dim,
            10000,
            self.n_layers,
        )(
            x
        )
        x = nn.LayerNorm(use_bias=False)(x)
        logits = nn.Dense(
            features=self.output_dim, kernel_init=init_fn(x.shape[-1]), use_bias=False
        )(x)
        return jnp.tanh(logits / 30) * 30
