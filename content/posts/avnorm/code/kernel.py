import torch
import triton
import triton.language as tl

# Reduced autotune configs for faster compilation (H100 optimized)
_FWD_CONFIGS = [triton.Config({'BLOCK_M': m, 'BLOCK_N': n}, num_warps=w, num_stages=s)  #
                for m in [64, 128]  #
                for n in [64, 128]  #
                for w in [4, 8]  #
                for s in [2, 3]]
_BWD_CONFIGS = _FWD_CONFIGS

_PRE_CONFIGS = [triton.Config({'BLOCK_M': m}, num_warps=w, num_stages=s)  #
                for m in [128, 256, 512]  #
                for w in [4, 8]  #
                for s in [2]]


def _prune_configs(configs, named_args, **kwargs):
    seqlen = named_args['seqlen']
    pruned = []
    for cfg in configs:
        block_m = cfg.kwargs['BLOCK_M']
        block_n = cfg.kwargs.get('BLOCK_N', block_m)
        if block_m <= seqlen and block_n <= seqlen:
            pruned.append(cfg)
    return pruned if pruned else configs[:4]


# =============================================================================
# Non-Varlen Kernels (original implementation)
# =============================================================================

@triton.autotune(configs=_FWD_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _fwd_kernel(Q, K, V, O, M, L, sm_scale, p_value, window_size, stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh,
                stride_kn, stride_kk, stride_vb, stride_vh, stride_vn, stride_vk, stride_ob, stride_oh, stride_om,
                stride_ok, stride_mb, stride_mh, stride_mm, seqlen, headdim, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ):
    bid, hid, mid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = Q + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + bid * stride_kb + hid * stride_kh + offs_k[:, None] * stride_kk
    v_ptrs = V + bid * stride_vb + hid * stride_vh + offs_k[None, :] * stride_vk

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)

    for start_n in range(0, (mid + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn + offs_n[None, :] * stride_kn,
                    mask=(offs_k[:, None] < headdim) & ((start_n + offs_n[None, :]) < seqlen), other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn + offs_n[:, None] * stride_vn,
                    mask=((start_n + offs_n[:, None]) < seqlen) & (offs_k[None, :] < headdim), other=0.0)

        s = tl.dot(q, k) * sm_scale
        # Causal mask: can only attend to previous tokens
        # Sliding window mask: can only attend to tokens within window_size (0 = unlimited)
        # window_size semantics match flash_attn: valid keys in [query_pos - window_size, query_pos]
        causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        if window_size > 0:
            window_mask = offs_m[:, None] - (start_n + offs_n[None, :]) <= window_size
            s = tl.where(causal_mask & window_mask, s, float('-inf'))
        else:
            s = tl.where(causal_mask, s, float('-inf'))

        m_j = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_j)
        l_i = l_i * tl.exp(p_value * (m_i - m_new))

        s_centered = s - m_new[:, None]
        l_i = l_i + tl.sum(tl.exp(p_value * s_centered), axis=1)
        o_i = o_i * tl.exp(m_i - m_new)[:, None] + tl.dot(tl.exp(s_centered).to(v.dtype), v)
        m_i = m_new

    l_i = tl.maximum(l_i, 1e-12)
    o_i = o_i / tl.exp(tl.log(l_i) / p_value)[:, None]

    o_ptrs = O + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, o_i.to(O.dtype.element_ty), mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim))

    m_ptrs = M + bid * stride_mb + hid * stride_mh + offs_m * stride_mm
    l_ptrs = L + bid * stride_mb + hid * stride_mh + offs_m * stride_mm
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)


@triton.autotune(configs=_PRE_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _bwd_preprocess(O, dO, Delta, stride_ob, stride_oh, stride_om, stride_ok, stride_db, stride_dh, stride_dm, seqlen,
                    headdim, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, ):
    bid, hid, mid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    o = tl.load(O + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
    do = tl.load(dO + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                 mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)

    tl.store(Delta + bid * stride_db + hid * stride_dh + offs_m * stride_dm, tl.sum(o * do, axis=1),
             mask=offs_m < seqlen)


@triton.autotune(configs=_BWD_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _bwd_kernel(Q, K, V, M, L, dO, dK, dV, Delta, sm_scale, p_value, window_size, stride_qb, stride_qh, stride_qm, stride_qk,
                stride_kb, stride_kh, stride_kn, stride_kk, stride_vb, stride_vh, stride_vn, stride_vk, stride_dob,
                stride_doh, stride_dom, stride_dok, stride_mb, stride_mh, stride_mm, stride_db, stride_dh, stride_dm,
                seqlen, headdim, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ):
    bid, hid, nid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k = tl.load(K + bid * stride_kb + hid * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
    v = tl.load(V + bid * stride_vb + hid * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    inv_p = 1.0 / p_value

    for start_m in range(nid * BLOCK_N, seqlen, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        curr_m = start_m + offs_m

        q = tl.load(Q + bid * stride_qb + hid * stride_qh + curr_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(curr_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
        do = tl.load(
            dO + bid * stride_dob + hid * stride_doh + curr_m[:, None] * stride_dom + offs_k[None, :] * stride_dok,
            mask=(curr_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
        delta = tl.load(Delta + bid * stride_db + hid * stride_dh + curr_m * stride_dm, mask=curr_m < seqlen, other=0.0)
        m = tl.load(M + bid * stride_mb + hid * stride_mh + curr_m * stride_mm, mask=curr_m < seqlen, other=0.0)
        l = tl.maximum(
            tl.load(L + bid * stride_mb + hid * stride_mh + curr_m * stride_mm, mask=curr_m < seqlen, other=1.0), 1e-12)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        causal_mask = curr_m[:, None] >= offs_n[None, :]
        if window_size > 0:
            window_mask = curr_m[:, None] - offs_n[None, :] <= window_size
            mask = causal_mask & window_mask
        else:
            mask = causal_mask
        s = tl.where(mask, s, float('-inf'))
        s_centered = s - m[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l[:, None]

        dv += tl.dot(tl.trans(attn.to(do.dtype)), do)
        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(mask, ds, 0.0)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    tl.store(dK + bid * stride_kb + hid * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
             dk.to(K.dtype.element_ty), mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim))
    tl.store(dV + bid * stride_vb + hid * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
             dv.to(V.dtype.element_ty), mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim))


@triton.autotune(configs=_BWD_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _bwd_dq_kernel(Q, K, V, M, L, dO, dQ, Delta, sm_scale, p_value, window_size, stride_qb, stride_qh, stride_qm, stride_qk,
                   stride_kb, stride_kh, stride_kn, stride_kk, stride_vb, stride_vh, stride_vn, stride_vk, stride_dob,
                   stride_doh, stride_dom, stride_dok, stride_mb, stride_mh, stride_mm, stride_db, stride_dh, stride_dm,
                   seqlen, headdim, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ):
    bid, hid, mid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(Q + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
    do = tl.load(dO + bid * stride_dob + hid * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok,
                 mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
    delta = tl.load(Delta + bid * stride_db + hid * stride_dh + offs_m * stride_dm, mask=offs_m < seqlen, other=0.0)
    m_full = tl.load(M + bid * stride_mb + hid * stride_mh + offs_m * stride_mm, mask=offs_m < seqlen, other=0.0)
    l_full = tl.maximum(
        tl.load(L + bid * stride_mb + hid * stride_mh + offs_m * stride_mm, mask=offs_m < seqlen, other=1.0), 1e-12)

    dq = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    inv_p = 1.0 / p_value

    for start_n in range(0, (mid + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n

        k = tl.load(K + bid * stride_kb + hid * stride_kh + curr_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(curr_n[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)
        v = tl.load(V + bid * stride_vb + hid * stride_vh + curr_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                    mask=(curr_n[:, None] < seqlen) & (offs_k[None, :] < headdim), other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        if window_size > 0:
            window_mask = offs_m[:, None] - curr_n[None, :] <= window_size
            mask = causal_mask & window_mask
        else:
            mask = causal_mask
        s = tl.where(mask, s, float('-inf'))
        s_centered = s - m_full[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l_full) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l_full[:, None]

        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(mask, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    tl.store(dQ + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
             dq.to(Q.dtype.element_ty), mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim))


# =============================================================================
# Varlen Kernels (packed sequences with cu_seqlens)
# For varlen mode: Q/K/V are (total_tokens, heads, headdim)
# cu_seqlens is (batch+1,) with document boundaries
# =============================================================================

def _prune_configs_varlen(configs, named_args, **kwargs):
    max_seqlen = named_args['max_seqlen']
    pruned = []
    for cfg in configs:
        block_m = cfg.kwargs['BLOCK_M']
        block_n = cfg.kwargs.get('BLOCK_N', block_m)
        if block_m <= max_seqlen and block_n <= max_seqlen:
            pruned.append(cfg)
    return pruned if pruned else configs[:4]


@triton.autotune(configs=_FWD_CONFIGS, key=['max_seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs_varlen})
@triton.jit
def _fwd_kernel_varlen(
    Q, K, V, O, M, L, cu_seqlens,
    sm_scale, p_value, window_size,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_om, stride_oh, stride_ok,
    stride_mh, stride_mm,
    total_tokens, num_heads, max_seqlen, headdim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs: (batch_idx, head_idx, block_m_idx)
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)

    # Get document boundaries from cu_seqlens
    bos = tl.load(cu_seqlens + bid)
    eos = tl.load(cu_seqlens + bid + 1)
    doc_len = eos - bos

    # Early exit if this block is beyond the document length
    if mid * BLOCK_M >= doc_len:
        return

    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Global positions (within packed tensor)
    global_m = bos + offs_m

    # Pointers - Q/K/V are (total_tokens, heads, headdim)
    q_ptrs = Q + global_m[:, None] * stride_qm + hid * stride_qh + offs_k[None, :] * stride_qk
    k_base = K + bos * stride_km + hid * stride_kh
    v_base = V + bos * stride_vm + hid * stride_vh

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load query (local positions within document)
    q_mask = (offs_m[:, None] < doc_len) & (offs_k[None, :] < headdim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Iterate over key/value blocks (only up to causal boundary)
    for start_n in range(0, (mid + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n

        # Load K and V (local positions within document)
        k_ptrs = k_base + curr_n[None, :] * stride_km + offs_k[:, None] * stride_kk
        v_ptrs = v_base + curr_n[:, None] * stride_vm + offs_k[None, :] * stride_vk

        k_mask = (offs_k[:, None] < headdim) & (curr_n[None, :] < doc_len)
        v_mask = (curr_n[:, None] < doc_len) & (offs_k[None, :] < headdim)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        s = tl.dot(q, k) * sm_scale

        # Causal mask (local positions)
        causal_mask = offs_m[:, None] >= curr_n[None, :]

        # Document boundary mask (both query and key must be within document)
        doc_mask = (offs_m[:, None] < doc_len) & (curr_n[None, :] < doc_len)

        if window_size > 0:
            window_mask = offs_m[:, None] - curr_n[None, :] <= window_size
            valid_mask = causal_mask & window_mask & doc_mask
        else:
            valid_mask = causal_mask & doc_mask

        s = tl.where(valid_mask, s, float('-inf'))

        m_j = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_j)
        l_i = l_i * tl.exp(p_value * (m_i - m_new))

        s_centered = s - m_new[:, None]
        l_i = l_i + tl.sum(tl.exp(p_value * s_centered), axis=1)
        o_i = o_i * tl.exp(m_i - m_new)[:, None] + tl.dot(tl.exp(s_centered).to(v.dtype), v)
        m_i = m_new

    l_i = tl.maximum(l_i, 1e-12)
    o_i = o_i / tl.exp(tl.log(l_i) / p_value)[:, None]

    # Store output
    o_ptrs = O + global_m[:, None] * stride_om + hid * stride_oh + offs_k[None, :] * stride_ok
    o_mask = (offs_m[:, None] < doc_len) & (offs_k[None, :] < headdim)
    tl.store(o_ptrs, o_i.to(O.dtype.element_ty), mask=o_mask)

    # Store M and L for backward pass
    m_ptrs = M + hid * stride_mh + global_m * stride_mm
    l_ptrs = L + hid * stride_mh + global_m * stride_mm
    ml_mask = offs_m < doc_len
    tl.store(m_ptrs, m_i, mask=ml_mask)
    tl.store(l_ptrs, l_i, mask=ml_mask)


@triton.autotune(configs=_PRE_CONFIGS, key=['max_seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs_varlen})
@triton.jit
def _bwd_preprocess_varlen(
    O, dO, Delta, cu_seqlens,
    stride_om, stride_oh, stride_ok,
    stride_dh, stride_dm,
    total_tokens, num_heads, max_seqlen, headdim,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)

    bos = tl.load(cu_seqlens + bid)
    eos = tl.load(cu_seqlens + bid + 1)
    doc_len = eos - bos

    if mid * BLOCK_M >= doc_len:
        return

    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    global_m = bos + offs_m

    mask = (offs_m[:, None] < doc_len) & (offs_k[None, :] < headdim)

    o = tl.load(O + global_m[:, None] * stride_om + hid * stride_oh + offs_k[None, :] * stride_ok,
                mask=mask, other=0.0)
    do = tl.load(dO + global_m[:, None] * stride_om + hid * stride_oh + offs_k[None, :] * stride_ok,
                 mask=mask, other=0.0)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + hid * stride_dh + global_m * stride_dm, delta, mask=offs_m < doc_len)


@triton.autotune(configs=_BWD_CONFIGS, key=['max_seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs_varlen})
@triton.jit
def _bwd_kernel_varlen(
    Q, K, V, M, L, dO, dK, dV, Delta, cu_seqlens,
    sm_scale, p_value, window_size,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_mh, stride_mm,
    stride_dh, stride_dm,
    total_tokens, num_heads, max_seqlen, headdim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.program_id(2)

    bos = tl.load(cu_seqlens + bid)
    eos = tl.load(cu_seqlens + bid + 1)
    doc_len = eos - bos

    if nid * BLOCK_N >= doc_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_n = bos + offs_n

    # Load K and V for this block
    k_mask = (offs_n[:, None] < doc_len) & (offs_k[None, :] < headdim)
    v_mask = k_mask

    k = tl.load(K + global_n[:, None] * stride_km + hid * stride_kh + offs_k[None, :] * stride_kk,
                mask=k_mask, other=0.0)
    v = tl.load(V + global_n[:, None] * stride_vm + hid * stride_vh + offs_k[None, :] * stride_vk,
                mask=v_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    inv_p = 1.0 / p_value

    # Iterate over query blocks that can attend to this key block
    for start_m in range(nid * BLOCK_N, doc_len, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        curr_m = start_m + offs_m
        global_m = bos + curr_m

        q_mask = (curr_m[:, None] < doc_len) & (offs_k[None, :] < headdim)
        q = tl.load(Q + global_m[:, None] * stride_qm + hid * stride_qh + offs_k[None, :] * stride_qk,
                    mask=q_mask, other=0.0)
        do = tl.load(dO + global_m[:, None] * stride_dom + hid * stride_doh + offs_k[None, :] * stride_dok,
                     mask=q_mask, other=0.0)

        delta = tl.load(Delta + hid * stride_dh + global_m * stride_dm, mask=curr_m < doc_len, other=0.0)
        m = tl.load(M + hid * stride_mh + global_m * stride_mm, mask=curr_m < doc_len, other=0.0)
        l = tl.maximum(tl.load(L + hid * stride_mh + global_m * stride_mm, mask=curr_m < doc_len, other=1.0), 1e-12)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        causal_mask = curr_m[:, None] >= offs_n[None, :]
        doc_mask = (curr_m[:, None] < doc_len) & (offs_n[None, :] < doc_len)

        if window_size > 0:
            window_mask = curr_m[:, None] - offs_n[None, :] <= window_size
            mask = causal_mask & window_mask & doc_mask
        else:
            mask = causal_mask & doc_mask

        s = tl.where(mask, s, float('-inf'))
        s_centered = s - m[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l[:, None]

        dv += tl.dot(tl.trans(attn.to(do.dtype)), do)
        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(mask, ds, 0.0)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    # Store dK and dV
    dk_mask = (offs_n[:, None] < doc_len) & (offs_k[None, :] < headdim)
    tl.store(dK + global_n[:, None] * stride_km + hid * stride_kh + offs_k[None, :] * stride_kk,
             dk.to(K.dtype.element_ty), mask=dk_mask)
    tl.store(dV + global_n[:, None] * stride_vm + hid * stride_vh + offs_k[None, :] * stride_vk,
             dv.to(V.dtype.element_ty), mask=dk_mask)


@triton.autotune(configs=_BWD_CONFIGS, key=['max_seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs_varlen})
@triton.jit
def _bwd_dq_kernel_varlen(
    Q, K, V, M, L, dO, dQ, Delta, cu_seqlens,
    sm_scale, p_value, window_size,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_mh, stride_mm,
    stride_dh, stride_dm,
    total_tokens, num_heads, max_seqlen, headdim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)

    bos = tl.load(cu_seqlens + bid)
    eos = tl.load(cu_seqlens + bid + 1)
    doc_len = eos - bos

    if mid * BLOCK_M >= doc_len:
        return

    offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_m = bos + offs_m

    q_mask = (offs_m[:, None] < doc_len) & (offs_k[None, :] < headdim)
    q = tl.load(Q + global_m[:, None] * stride_qm + hid * stride_qh + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    do = tl.load(dO + global_m[:, None] * stride_dom + hid * stride_doh + offs_k[None, :] * stride_dok,
                 mask=q_mask, other=0.0)

    delta = tl.load(Delta + hid * stride_dh + global_m * stride_dm, mask=offs_m < doc_len, other=0.0)
    m_full = tl.load(M + hid * stride_mh + global_m * stride_mm, mask=offs_m < doc_len, other=0.0)
    l_full = tl.maximum(tl.load(L + hid * stride_mh + global_m * stride_mm, mask=offs_m < doc_len, other=1.0), 1e-12)

    dq = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    inv_p = 1.0 / p_value

    # Iterate over key/value blocks up to causal boundary
    for start_n in range(0, (mid + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n
        global_n = bos + curr_n

        kv_mask = (curr_n[:, None] < doc_len) & (offs_k[None, :] < headdim)
        k = tl.load(K + global_n[:, None] * stride_km + hid * stride_kh + offs_k[None, :] * stride_kk,
                    mask=kv_mask, other=0.0)
        v = tl.load(V + global_n[:, None] * stride_vm + hid * stride_vh + offs_k[None, :] * stride_vk,
                    mask=kv_mask, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        causal_mask = offs_m[:, None] >= curr_n[None, :]
        doc_mask = (offs_m[:, None] < doc_len) & (curr_n[None, :] < doc_len)

        if window_size > 0:
            window_mask = offs_m[:, None] - curr_n[None, :] <= window_size
            mask = causal_mask & window_mask & doc_mask
        else:
            mask = causal_mask & doc_mask

        s = tl.where(mask, s, float('-inf'))
        s_centered = s - m_full[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l_full) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l_full[:, None]

        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(mask, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    dq_mask = (offs_m[:, None] < doc_len) & (offs_k[None, :] < headdim)
    tl.store(dQ + global_m[:, None] * stride_qm + hid * stride_qh + offs_k[None, :] * stride_qk,
             dq.to(Q.dtype.element_ty), mask=dq_mask)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# -----------------------------------------------------------------------------
# torch.library.triton_op registration for torch.compile compatibility with fullgraph=True
# This is the recommended way to use triton kernels with torch.compile.
# See: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

from torch import Tensor
from torch.library import triton_op, wrap_triton

# Forward op using triton_op decorator (non-varlen)
@triton_op("nanogpt::l2_attention_fwd", mutates_args={})
def l2_attention_fwd_op(
    q: Tensor, k: Tensor, v: Tensor, p: float, sm_scale: float, window_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Forward pass of L2 attention using triton kernel."""
    batch, heads, seqlen, headdim = q.shape
    assert headdim in [16, 32, 48, 64, 128], f"headdim must be in [16, 32, 48, 64, 128], got {headdim}"

    o = torch.empty_like(q)
    m = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)
    l = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)
    BLOCK_K = _next_power_of_2(headdim)

    # Use wrap_triton for proper torch.compile integration
    grid = lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M']))
    wrap_triton(_fwd_kernel)[grid](
        q, k, v, o, m, l, sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        seqlen, headdim, BLOCK_K=BLOCK_K,
    )
    return o, m, l


# Backward op using triton_op decorator (non-varlen)
@triton_op("nanogpt::l2_attention_bwd", mutates_args={})
def l2_attention_bwd_op(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, m: Tensor, l: Tensor,
    do: Tensor, p: float, sm_scale: float, window_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward pass of L2 attention using triton kernels."""
    batch, heads, seqlen, headdim = q.shape
    BLOCK_K = _next_power_of_2(headdim)

    do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)

    # Preprocess kernel
    grid_pre = lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M']))
    wrap_triton(_bwd_preprocess)[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        seqlen, headdim, BLOCK_K=BLOCK_K,
    )

    # dK/dV kernel
    grid_kv = lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_N']))
    wrap_triton(_bwd_kernel)[grid_kv](
        q, k, v, m, l, do, dk, dv, delta, sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        seqlen, headdim, BLOCK_K=BLOCK_K,
    )

    # dQ kernel
    grid_q = lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M']))
    wrap_triton(_bwd_dq_kernel)[grid_q](
        q, k, v, m, l, do, dq, delta, sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        seqlen, headdim, BLOCK_K=BLOCK_K,
    )
    return dq, dk, dv


# =============================================================================
# Varlen forward/backward ops
# =============================================================================

@triton_op("nanogpt::l2_attention_fwd_varlen", mutates_args={})
def l2_attention_fwd_varlen_op(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor,
    max_seqlen: int, p: float, sm_scale: float, window_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Forward pass of varlen L2 attention using triton kernel.

    Args:
        q, k, v: (total_tokens, heads, headdim)
        cu_seqlens: (batch+1,) cumulative sequence lengths
        max_seqlen: maximum sequence length in the batch
    """
    total_tokens, heads, headdim = q.shape
    batch = cu_seqlens.shape[0] - 1
    assert headdim in [16, 32, 48, 64, 128], f"headdim must be in [16, 32, 48, 64, 128], got {headdim}"

    o = torch.empty_like(q)
    m = torch.empty(heads, total_tokens, device=q.device, dtype=torch.float32)
    l = torch.empty(heads, total_tokens, device=q.device, dtype=torch.float32)
    BLOCK_K = _next_power_of_2(headdim)

    grid = lambda META: (batch, heads, triton.cdiv(max_seqlen, META['BLOCK_M']))
    wrap_triton(_fwd_kernel_varlen)[grid](
        q, k, v, o, m, l, cu_seqlens,
        sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        m.stride(0), m.stride(1),
        total_tokens, heads, max_seqlen, headdim,
        BLOCK_K=BLOCK_K,
    )
    return o, m, l


@triton_op("nanogpt::l2_attention_bwd_varlen", mutates_args={})
def l2_attention_bwd_varlen_op(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, m: Tensor, l: Tensor,
    do: Tensor, cu_seqlens: Tensor, max_seqlen: int,
    p: float, sm_scale: float, window_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward pass of varlen L2 attention using triton kernels."""
    total_tokens, heads, headdim = q.shape
    batch = cu_seqlens.shape[0] - 1
    BLOCK_K = _next_power_of_2(headdim)

    do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(heads, total_tokens, device=q.device, dtype=torch.float32)

    # Preprocess kernel
    grid_pre = lambda META: (batch, heads, triton.cdiv(max_seqlen, META['BLOCK_M']))
    wrap_triton(_bwd_preprocess_varlen)[grid_pre](
        o, do, delta, cu_seqlens,
        o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1),
        total_tokens, heads, max_seqlen, headdim,
        BLOCK_K=BLOCK_K,
    )

    # dK/dV kernel
    grid_kv = lambda META: (batch, heads, triton.cdiv(max_seqlen, META['BLOCK_N']))
    wrap_triton(_bwd_kernel_varlen)[grid_kv](
        q, k, v, m, l, do, dk, dv, delta, cu_seqlens,
        sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        m.stride(0), m.stride(1),
        delta.stride(0), delta.stride(1),
        total_tokens, heads, max_seqlen, headdim,
        BLOCK_K=BLOCK_K,
    )

    # dQ kernel
    grid_q = lambda META: (batch, heads, triton.cdiv(max_seqlen, META['BLOCK_M']))
    wrap_triton(_bwd_dq_kernel_varlen)[grid_q](
        q, k, v, m, l, do, dq, delta, cu_seqlens,
        sm_scale, p, window_size,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        m.stride(0), m.stride(1),
        delta.stride(0), delta.stride(1),
        total_tokens, heads, max_seqlen, headdim,
        BLOCK_K=BLOCK_K,
    )
    return dq, dk, dv


# =============================================================================
# Autograd registration
# =============================================================================

# Register autograd for the forward op (non-varlen)
def _l2_attention_backward(ctx, grad_o: Tensor, grad_m: Tensor, grad_l: Tensor):
    q, k, v, o, m, l = ctx.saved_tensors
    p, sm_scale, window_size = ctx.scales
    dq, dk, dv = torch.ops.nanogpt.l2_attention_bwd(
        q, k, v, o, m, l, grad_o, p, sm_scale, window_size
    )
    return dq, dk, dv, None, None, None


def _l2_attention_setup_context(ctx, inputs, output):
    q, k, v, p, sm_scale, window_size = inputs
    o, m, l = output
    ctx.save_for_backward(q, k, v, o, m, l)
    ctx.scales = (p, sm_scale, window_size)


l2_attention_fwd_op.register_autograd(_l2_attention_backward, setup_context=_l2_attention_setup_context)


# Register autograd for varlen forward op
def _l2_attention_varlen_backward(ctx, grad_o: Tensor, grad_m: Tensor, grad_l: Tensor):
    q, k, v, o, m, l, cu_seqlens = ctx.saved_tensors
    max_seqlen, p, sm_scale, window_size = ctx.params
    dq, dk, dv = torch.ops.nanogpt.l2_attention_bwd_varlen(
        q, k, v, o, m, l, grad_o, cu_seqlens, max_seqlen, p, sm_scale, window_size
    )
    return dq, dk, dv, None, None, None, None, None


def _l2_attention_varlen_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen, p, sm_scale, window_size = inputs
    o, m, l = output
    ctx.save_for_backward(q, k, v, o, m, l, cu_seqlens)
    ctx.params = (max_seqlen, p, sm_scale, window_size)


l2_attention_fwd_varlen_op.register_autograd(_l2_attention_varlen_backward, setup_context=_l2_attention_varlen_setup_context)


# =============================================================================
# Main API function
# =============================================================================

def triton_p_softmax_attention(
    q: Tensor, k: Tensor, v: Tensor, p: float = 1.0, sm_scale: float = None,
    window_size: int = 0, cu_seqlens: Tensor = None, max_seqlen: int = None
) -> Tensor:
    """
    p-softmax attention with optional sliding window and varlen support.
    Compatible with torch.compile fullgraph=True.

    Args:
        q, k, v: Query, key, value tensors
            - Non-varlen: shape (batch, heads, seqlen, headdim)
            - Varlen: shape (total_tokens, heads, headdim)
        p: p-softmax parameter (p=1 is standard softmax, p=2 is L2 normalization)
        sm_scale: Attention scale factor (default: headdim ** -0.5)
        window_size: Sliding window size (0 = unlimited/full causal attention)
        cu_seqlens: Cumulative sequence lengths for varlen mode (shape: batch+1, dtype: int32)
            When provided, enables varlen mode for packed sequences.
        max_seqlen: Maximum sequence length in the batch (required for varlen mode).
            Must be provided when cu_seqlens is not None.

    Returns:
        Output tensor with same shape as q
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    if cu_seqlens is None:
        # Non-varlen path: (batch, heads, seqlen, headdim)
        o, _, _ = torch.ops.nanogpt.l2_attention_fwd(q, k, v, p, sm_scale, window_size)
    else:
        # Varlen path: (total_tokens, heads, headdim)
        assert max_seqlen is not None, "max_seqlen must be provided for varlen mode"
        o, _, _ = torch.ops.nanogpt.l2_attention_fwd_varlen(
            q, k, v, cu_seqlens, max_seqlen, p, sm_scale, window_size
        )
    return o


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Testing non-varlen path")
    print("=" * 60)

    B, H, S, D, P = 2, 8, 256, 64, 2.0

    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16, requires_grad=True)

    out = triton_p_softmax_attention(q, k, v, p=P)
    out.sum().backward()

    q2, k2, v2 = [x.detach().clone().requires_grad_(True) for x in [q, k, v]]
    scale = D ** -0.5
    scores = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(S, S, device='cuda', dtype=torch.bool))
    scores = scores.masked_fill(~mask, float('-inf'))
    scores_centered = scores - scores.max(dim=-1, keepdim=True).values
    attn = torch.exp(scores_centered) / torch.pow(torch.exp(P * scores_centered).sum(dim=-1, keepdim=True), 1 / P)
    out_ref = torch.matmul(attn, v2)
    out_ref.sum().backward()

    print(f"fwd diff: {(out - out_ref).abs().max().item():.6f}")
    print(f"dq diff:  {(q.grad - q2.grad).abs().max().item():.6f}")
    print(f"dk diff:  {(k.grad - k2.grad).abs().max().item():.6f}")
    print(f"dv diff:  {(v.grad - v2.grad).abs().max().item():.6f}")

    print()
    print("=" * 60)
    print("Testing varlen path")
    print("=" * 60)

    # Create packed sequences with different lengths
    torch.manual_seed(42)
    seq_lens = [64, 128, 96]  # 3 documents with different lengths
    total_tokens = sum(seq_lens)
    H, D, P = 8, 64, 2.0

    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
                               device='cuda', dtype=torch.int32)

    # Varlen format: (total_tokens, heads, headdim)
    q_var = torch.randn(total_tokens, H, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k_var = torch.randn(total_tokens, H, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v_var = torch.randn(total_tokens, H, D, device='cuda', dtype=torch.float16, requires_grad=True)

    out_var = triton_p_softmax_attention(q_var, k_var, v_var, p=P, cu_seqlens=cu_seqlens, max_seqlen=max(seq_lens))
    out_var.sum().backward()

    # Reference: compute attention for each document separately
    q_var2 = q_var.detach().clone().requires_grad_(True)
    k_var2 = k_var.detach().clone().requires_grad_(True)
    v_var2 = v_var.detach().clone().requires_grad_(True)

    out_ref_list = []
    scale = D ** -0.5

    for i, seq_len in enumerate(seq_lens):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()

        # Extract this document's Q, K, V: (seq_len, heads, headdim)
        q_doc = q_var2[start:end]  # (seq_len, H, D)
        k_doc = k_var2[start:end]
        v_doc = v_var2[start:end]

        # Compute attention: need to transpose for matmul
        # scores: (H, seq_len, seq_len)
        scores = torch.einsum('qhd,khd->hqk', q_doc.float(), k_doc.float()) * scale

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device='cuda', dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # p-softmax
        scores_centered = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.exp(scores_centered) / torch.pow(
            torch.exp(P * scores_centered).sum(dim=-1, keepdim=True), 1 / P
        )

        # Output: (H, seq_len, D) -> (seq_len, H, D)
        out_doc = torch.einsum('hqk,khd->qhd', attn, v_doc.float())
        out_ref_list.append(out_doc.to(torch.float16))

    out_ref_var = torch.cat(out_ref_list, dim=0)
    out_ref_var.sum().backward()

    print(f"varlen fwd diff: {(out_var - out_ref_var).abs().max().item():.6f}")
    print(f"varlen dq diff:  {(q_var.grad - q_var2.grad).abs().max().item():.6f}")
    print(f"varlen dk diff:  {(k_var.grad - k_var2.grad).abs().max().item():.6f}")
    print(f"varlen dv diff:  {(v_var.grad - v_var2.grad).abs().max().item():.6f}")

    print()
    print("=" * 60)
    print("Testing that documents don't attend across boundaries")
    print("=" * 60)

    # Test that document boundary masking works correctly
    # Create 2 documents and verify no cross-document attention
    torch.manual_seed(123)
    seq_lens_test = [32, 32]
    total_test = sum(seq_lens_test)

    cu_seqlens_test = torch.tensor([0, 32, 64], device='cuda', dtype=torch.int32)

    # Make Q of doc2 very similar to K of doc1 - if boundary masking fails,
    # doc2 queries would attend to doc1 keys
    q_test = torch.randn(total_test, H, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k_test = torch.randn(total_test, H, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v_test = torch.randn(total_test, H, D, device='cuda', dtype=torch.float16, requires_grad=True)

    out_test = triton_p_softmax_attention(q_test, k_test, v_test, p=P, cu_seqlens=cu_seqlens_test, max_seqlen=32)

    # Compute reference for doc2 only using its own K, V
    q_doc2 = q_test[32:64].detach()
    k_doc2 = k_test[32:64].detach()
    v_doc2 = v_test[32:64].detach()

    scores_doc2 = torch.einsum('qhd,khd->hqk', q_doc2.float(), k_doc2.float()) * scale
    causal_mask_doc2 = torch.tril(torch.ones(32, 32, device='cuda', dtype=torch.bool))
    scores_doc2 = scores_doc2.masked_fill(~causal_mask_doc2, float('-inf'))
    scores_centered_doc2 = scores_doc2 - scores_doc2.max(dim=-1, keepdim=True).values
    attn_doc2 = torch.exp(scores_centered_doc2) / torch.pow(
        torch.exp(P * scores_centered_doc2).sum(dim=-1, keepdim=True), 1 / P
    )
    out_doc2_ref = torch.einsum('hqk,khd->qhd', attn_doc2, v_doc2.float()).to(torch.float16)

    doc2_diff = (out_test[32:64] - out_doc2_ref).abs().max().item()
    print(f"Doc2 isolation test (should be small): {doc2_diff:.6f}")

    if doc2_diff < 0.01:
        print("Document boundary masking is working correctly!")
    else:
        print("WARNING: Document boundary masking may not be working correctly!")

    print()
    print("All tests completed!")
