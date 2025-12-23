import torch
import triton
import triton.language as tl

_FWD_CONFIGS = [triton.Config({'BLOCK_M': m, 'BLOCK_N': n}, num_warps=w, num_stages=s)  #
                for m in [32, 64, 128, 256]  #
                for n in [32, 64, 128, 256]  #
                for w in [2, 4, 8, 16]  #
                for s in [1, 2, 3, 4]]
_BWD_CONFIGS = _FWD_CONFIGS

_PRE_CONFIGS = [triton.Config({'BLOCK_M': m}, num_warps=w, num_stages=s)  #
                for m in [64, 128, 256, 512, 1024]  #
                for w in [4, 8, 16]  #
                for s in [1, 2]]


def _prune_configs(configs, named_args, **kwargs):
    seqlen = named_args['seqlen']
    pruned = []
    for cfg in configs:
        block_m = cfg.kwargs['BLOCK_M']
        block_n = cfg.kwargs.get('BLOCK_N', block_m)
        if block_m <= seqlen and block_n <= seqlen:
            pruned.append(cfg)
    return pruned if pruned else configs[:4]


@triton.autotune(configs=_FWD_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _fwd_kernel(Q, K, V, O, M, L, sm_scale, p_value, stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh,
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
        s = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), s, float('-inf'))

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
def _bwd_kernel(Q, K, V, M, L, dO, dK, dV, Delta, sm_scale, p_value, stride_qb, stride_qh, stride_qm, stride_qk,
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
        s = tl.where(causal_mask, s, float('-inf'))
        s_centered = s - m[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l[:, None]

        dv += tl.dot(tl.trans(attn.to(do.dtype)), do)
        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(causal_mask, ds, 0.0)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    tl.store(dK + bid * stride_kb + hid * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
             dk.to(K.dtype.element_ty), mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim))
    tl.store(dV + bid * stride_vb + hid * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
             dv.to(V.dtype.element_ty), mask=(offs_n[:, None] < seqlen) & (offs_k[None, :] < headdim))


@triton.autotune(configs=_BWD_CONFIGS, key=['seqlen', 'headdim'],
                 prune_configs_by={'early_config_prune': _prune_configs})
@triton.jit
def _bwd_dq_kernel(Q, K, V, M, L, dO, dQ, Delta, sm_scale, p_value, stride_qb, stride_qh, stride_qm, stride_qk,
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
        s = tl.where(causal_mask, s, float('-inf'))
        s_centered = s - m_full[:, None]

        attn = tl.exp(s_centered) / tl.exp(tl.log(l_full) * inv_p)[:, None]
        attn_p = tl.exp(p_value * s_centered) / l_full[:, None]

        ds = attn * tl.dot(do, tl.trans(v)) - attn_p * delta[:, None]
        ds = tl.where(causal_mask, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    tl.store(dQ + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
             dq.to(Q.dtype.element_ty), mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < headdim))


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class TritonPSoftmaxAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, p):
        batch, heads, seqlen, headdim = q.shape
        assert headdim in [16, 32, 48, 64, 128], f"headdim must be in [16, 32, 48, 64, 128], got {headdim}"

        o = torch.empty_like(q)
        m = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)
        l = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)
        BLOCK_K = _next_power_of_2(headdim)
        sm_scale = headdim ** -0.5

        _fwd_kernel[(lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M'])))](q, k, v, o, m, l, sm_scale, p,
                                                                                         q.stride(0), q.stride(1),
                                                                                         q.stride(2), q.stride(3),
                                                                                         k.stride(0), k.stride(1),
                                                                                         k.stride(2), k.stride(3),
                                                                                         v.stride(0), v.stride(1),
                                                                                         v.stride(2), v.stride(3),
                                                                                         o.stride(0), o.stride(1),
                                                                                         o.stride(2), o.stride(3),
                                                                                         m.stride(0), m.stride(1),
                                                                                         m.stride(2), seqlen, headdim,
                                                                                         BLOCK_K=BLOCK_K, )

        ctx.save_for_backward(q, k, v, o, m, l)
        ctx.p, ctx.sm_scale, ctx.BLOCK_K = p, sm_scale, BLOCK_K
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, l = ctx.saved_tensors
        batch, heads, seqlen, headdim = q.shape
        BLOCK_K = ctx.BLOCK_K

        do = do.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        delta = torch.empty(batch, heads, seqlen, device=q.device, dtype=torch.float32)

        _bwd_preprocess[(lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M'])))](o, do, delta, o.stride(0),
                                                                                             o.stride(1), o.stride(2),
                                                                                             o.stride(3),
                                                                                             delta.stride(0),
                                                                                             delta.stride(1),
                                                                                             delta.stride(2), seqlen,
                                                                                             headdim, BLOCK_K=BLOCK_K, )

        _bwd_kernel[(lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_N'])))](q, k, v, m, l, do, dk, dv,
                                                                                         delta, ctx.sm_scale, ctx.p,
                                                                                         q.stride(0), q.stride(1),
                                                                                         q.stride(2), q.stride(3),
                                                                                         k.stride(0), k.stride(1),
                                                                                         k.stride(2), k.stride(3),
                                                                                         v.stride(0), v.stride(1),
                                                                                         v.stride(2), v.stride(3),
                                                                                         do.stride(0), do.stride(1),
                                                                                         do.stride(2), do.stride(3),
                                                                                         m.stride(0), m.stride(1),
                                                                                         m.stride(2), delta.stride(0),
                                                                                         delta.stride(1),
                                                                                         delta.stride(2), seqlen,
                                                                                         headdim, BLOCK_K=BLOCK_K, )

        _bwd_dq_kernel[(lambda META: (batch, heads, triton.cdiv(seqlen, META['BLOCK_M'])))](q, k, v, m, l, do, dq,
                                                                                            delta, ctx.sm_scale, ctx.p,
                                                                                            q.stride(0), q.stride(1),
                                                                                            q.stride(2), q.stride(3),
                                                                                            k.stride(0), k.stride(1),
                                                                                            k.stride(2), k.stride(3),
                                                                                            v.stride(0), v.stride(1),
                                                                                            v.stride(2), v.stride(3),
                                                                                            do.stride(0), do.stride(1),
                                                                                            do.stride(2), do.stride(3),
                                                                                            m.stride(0), m.stride(1),
                                                                                            m.stride(2),
                                                                                            delta.stride(0),
                                                                                            delta.stride(1),
                                                                                            delta.stride(2), seqlen,
                                                                                            headdim, BLOCK_K=BLOCK_K, )
        return dq, dk, dv, None


def triton_p_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    return TritonPSoftmaxAttention.apply(q, k, v, p)


if __name__ == "__main__":
    torch.manual_seed(42)
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
