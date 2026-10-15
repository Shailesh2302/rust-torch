//! Scaled dot-product + causal multi-head attention, RoPE.

use crate::activation::silu;
use crate::nn::linear::Linear;
use crate::nn::module::{Module, TrainFlag};
use crate::tensor::Tensor;

/// Rotary Position Embedding cache.
#[derive(Clone, Debug)]
pub struct RopeCache {
    pub cos: Vec<f32>, // [T, head_dim/2]
    pub sin: Vec<f32>,
    pub t: usize,
    pub head_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq: usize, head_dim: usize, theta: f32) -> Self {
        assert!(head_dim % 2 == 0);
        let half = head_dim / 2;
        let mut cos = vec![0.0; max_seq * half];
        let mut sin = vec![0.0; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let freq = 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32);
                let a = pos as f32 * freq;
                cos[pos * half + i] = a.cos();
                sin[pos * half + i] = a.sin();
            }
        }
        Self { cos, sin, t: max_seq, head_dim }
    }

    /// Apply RoPE to x shaped [B, nh, T, hs] -> same shape (differentiable).
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4);
        let (b, nh, t, hs) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(hs, self.head_dim);
        let half = hs / 2;
        let xd = x.data_vec();
        let mut out = vec![0.0; xd.len()];
        // x layout contiguous [B, nh, T, hs]
        for bi in 0..b {
            for h in 0..nh {
                for pos in 0..t {
                    for i in 0..half {
                        let base = ((bi * nh + h) * t + pos) * hs;
                        let x0 = xd[base + i];
                        let x1 = xd[base + half + i];
                        let c = self.cos[pos * half + i];
                        let s = self.sin[pos * half + i];
                        out[base + i] = x0 * c - x1 * s;
                        out[base + half + i] = x0 * s + x1 * c;
                    }
                }
            }
        }
        // wrap as differentiable-ish op: linear combo of inputs -> build via elementwise?
        // For simplicity route grad straight-through (rotation is orthogonal; exact backward
        // would rotate grad by -angle). Implement exact backward manually here by constructing
        // graph through available ops is complex, so we create an Identity-linked tensor then
        // patch data. Grad flows approximately (rotated). For training correctness with RoPE
        // we implement proper backward by custom handling in transformer (recompute).
        // Here: create tensor linked to x with correct forward data.
        let rg = x.requires_grad();
        if rg && !crate::tensor::is_no_grad() {
            // Build output that depends on x via mul/add so autograd works exactly:
            // out = x0*c - x1*s etc. We can express with narrow/cat/mul/add.
            // Simpler exact approach: out = x * C + x_roll * S with precomputed C,S tensors.
            let mut c_data = vec![0.0; xd.len()];
            let mut s_data = vec![0.0; xd.len()];
            // C = [c, c], S pattern = [-s for first half, +s for second]? out0 = x0*c + x1*(-s); out1 = x0*s + x1*c
            // Represent as: out = x*Cmat + x_swap*Smat where x_swap swaps halves with sign.
            for bi in 0..b {
                for h in 0..nh {
                    for pos in 0..t {
                        for i in 0..half {
                            let base = ((bi * nh + h) * t + pos) * hs;
                            let c = self.cos[pos * half + i];
                            let s = self.sin[pos * half + i];
                            c_data[base + i] = c;
                            c_data[base + half + i] = c;
                            s_data[base + i] = -s;
                            s_data[base + half + i] = s;
                        }
                    }
                }
            }
            // x_swap: swap halves of x: [x1..., x0...]
            let xs = swap_halves(x);
            let c_t = Tensor::new(c_data, shape.clone(), false);
            let s_t = Tensor::new(s_data, shape.clone(), false);
            // out = x*C + x_swap*S_weighted? careful: for first half need x1*(-s): xs first half = x1, times -s? s_data first half = -s, but xs first half=x1 so x1*(-s) good.
            // second half: xs second half = x0, times +s => x0*s good.
            let t1 = x.mul(&c_t);
            let t2 = xs.mul(&s_t);
            return t1.add(&t2);
        }
        Tensor::new(out, shape, false)
    }
}

fn swap_halves(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let hs = *shape.last().unwrap();
    let half = hs / 2;
    let d = x.data_vec();
    let rows = d.len() / hs;
    let mut out = vec![0.0; d.len()];
    for r in 0..rows {
        out[r * hs..r * hs + half].copy_from_slice(&d[r * hs + half..r * hs + hs]);
        out[r * hs + half..r * hs + hs].copy_from_slice(&d[r * hs..r * hs + half]);
    }
    // keep graph: build via narrow+cat so grad flows
    if x.requires_grad() && !crate::tensor::is_no_grad() {
        let a = x.narrow(-1, 0, half);
        let b = x.narrow(-1, half, half);
        return Tensor::cat(&[b, a], -1);
    }
    Tensor::new(out, shape, false)
}

/// Causal multi-head self-attention.
pub struct CausalSelfAttention {
    pub qkv: Option<Linear>, // fused GPT-style (3*embd) — used when fused=true
    pub q: Option<Linear>,
    pub k: Option<Linear>,
    pub v: Option<Linear>,
    pub proj: Linear,
    pub n_head: usize,
    pub n_embd: usize,
    pub head_dim: usize,
    pub dropout_p: f32,
    pub use_rope: bool,
    pub rope: Option<RopeCache>,
    flag: TrainFlag,
}

impl CausalSelfAttention {
    pub fn new_gpt(n_embd: usize, n_head: usize, dropout: f32, bias: bool) -> Self {
        assert_eq!(n_embd % n_head, 0);
        Self {
            qkv: Some(Linear::new(n_embd, 3 * n_embd, bias)),
            q: None, k: None, v: None,
            proj: Linear::new(n_embd, n_embd, bias),
            n_head, n_embd,
            head_dim: n_embd / n_head,
            dropout_p: dropout,
            use_rope: false, rope: None,
            flag: TrainFlag::new(true),
        }
    }

    pub fn new_llama(n_embd: usize, n_head: usize, dropout: f32, max_seq: usize) -> Self {
        assert_eq!(n_embd % n_head, 0);
        let hd = n_embd / n_head;
        Self {
            qkv: None,
            q: Some(Linear::new(n_embd, n_embd, false)),
            k: Some(Linear::new(n_embd, n_embd, false)),
            v: Some(Linear::new(n_embd, n_embd, false)),
            proj: Linear::new(n_embd, n_embd, false),
            n_head, n_embd, head_dim: hd,
            dropout_p: dropout,
            use_rope: true,
            rope: Some(RopeCache::new(max_seq, hd, 10000.0)),
            flag: TrainFlag::new(true),
        }
    }

    fn split_heads(&self, x: &Tensor, b: usize, t: usize) -> Tensor {
        // x: [B*T, C] or [B, T, C] -> [B, nh, T, hs]
        let c = self.n_embd;
        let x3 = if x.ndim() == 2 { x.reshape(&[b, t, c]) } else { x.reshape(&[b, t, c]) };
        let x4 = x3.reshape(&[b, t, self.n_head, self.head_dim]); // [B,T,nh,hs]
        x4.permute(&[0, 2, 1, 3]) // [B,nh,T,hs]
    }

    fn merge_heads(&self, x: &Tensor, b: usize, t: usize) -> Tensor {
        // [B,nh,T,hs] -> [B,T,C]
        let x2 = x.permute(&[0, 2, 1, 3]); // [B,T,nh,hs]
        x2.reshape(&[b, t, self.n_embd])
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [B, T, C]
        let s = x.shape();
        assert_eq!(s.len(), 3);
        let (b, t, c) = (s[0], s[1], s[2]);
        assert_eq!(c, self.n_embd);

        let (qh, kh, vh) = if let Some(qkv) = &self.qkv {
            let fused = qkv.forward(x); // [B,T,3C]
            let f2 = fused.reshape(&[b, t, 3, c]);
            // split last-but-one dim via narrow on dim 2
            let q = f2.narrow(2, 0, 1).squeeze(2); // [B,T,C]
            let k = f2.narrow(2, 1, 1).squeeze(2);
            let v = f2.narrow(2, 2, 1).squeeze(2);
            (self.split_heads(&q, b, t), self.split_heads(&k, b, t), self.split_heads(&v, b, t))
        } else {
            let q = self.q.as_ref().unwrap().forward(x);
            let k = self.k.as_ref().unwrap().forward(x);
            let v = self.v.as_ref().unwrap().forward(x);
            let mut qh = self.split_heads(&q, b, t);
            let mut kh = self.split_heads(&k, b, t);
            let vh = self.split_heads(&v, b, t);
            if self.use_rope {
                let rope = self.rope.as_ref().unwrap();
                qh = rope.apply(&qh);
                kh = rope.apply(&kh);
            }
            (qh, kh, vh)
        };

        // scores = Q @ K^T / sqrt(hd)  -> [B,nh,T,T]
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let kt = kh.transpose(); // last-two swap -> [B,nh,hs? wait kh [B,nh,T,hs] -> [B,nh,hs,T]] — correct via transpose()
        let mut scores = qh.matmul(&kt).mul_scalar(scale);

        // causal mask: upper triangular -> -inf
        scores = causal_mask_add(&scores);

        let attn = scores.softmax(-1).dropout(self.dropout_p, self.flag.get());
        let y = attn.matmul(&vh); // [B,nh,T,hs]
        let merged = self.merge_heads(&y, b, t); // [B,T,C]
        let merged2d = merged.reshape(&[b * t, c]);
        let out2d = self.proj.forward(&merged2d);
        out2d.reshape(&[b, t, c])
    }
}

impl Module for CausalSelfAttention {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = vec![];
        if let Some(qkv) = &self.qkv { v.extend(qkv.parameters()); }
        if let Some(q) = &self.q { v.extend(q.parameters()); }
        if let Some(k) = &self.k { v.extend(k.parameters()); }
        if let Some(vv) = &self.v { v.extend(vv.parameters()); }
        v.extend(self.proj.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.flag.set(t);
        if let Some(qkv) = &self.qkv { qkv.set_training(t); }
        if let Some(q) = &self.q { q.set_training(t); }
        if let Some(k) = &self.k { k.set_training(t); }
        if let Some(v) = &self.v { v.set_training(t); }
        self.proj.set_training(t);
    }
    fn is_training(&self) -> bool { self.flag.get() }
}

fn causal_mask_add(scores: &Tensor) -> Tensor {
    // scores [B,nh,T,T]: set j>i to -1e9 (no grad needed for mask)
    let s = scores.shape();
    let t = s[s.len() - 2];
    let t2 = s[s.len() - 1];
    assert_eq!(t, t2);
    let mut d = scores.data_vec();
    let rows_total = d.len() / (t * t);
    for r in 0..rows_total {
        for i in 0..t {
            for j in (i + 1)..t {
                d[r * t * t + i * t + j] = -1e9;
            }
        }
    }
    // keep graph via add of zero-with-mask? Simplest: rebuild tensor linked by add:
    // masked = scores + mask_tensor where mask is 0 / -1e9+scores diff... Instead do:
    // diff = new - old (no grad), then masked = scores + diff_tensor (diff no grad) so grad flows identity.
    let diff: Vec<f32> = d.iter().zip(scores.data_vec().iter()).map(|(a, b)| a - b).collect();
    let mask = Tensor::new(diff, s.clone(), false);
    scores.add(&mask)
}

/// Gated MLP variants live in mlp.rs; re-export SiLU-gated forward helper here if needed.
#[allow(dead_code)]
pub fn silu_gate(x: &Tensor) -> Tensor {
    silu(x)
}
