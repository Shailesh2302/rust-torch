//! Transformer blocks + GPT / LLaMA models + generation.

use crate::nn::attention::CausalSelfAttention;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::mlp::{Mlp, SwiGluMlp};
use crate::nn::module::{Module, TrainFlag};
use crate::tensor::{no_grad, Tensor};

// ---------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------

pub enum NormKind {
    LayerNorm(crate::nn::norm::LayerNorm),
    Rms(crate::nn::norm::RmsNorm),
}

impl NormKind {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            NormKind::LayerNorm(n) => n.forward(x),
            NormKind::Rms(n) => n.forward(x),
        }
    }
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            NormKind::LayerNorm(n) => n.parameters(),
            NormKind::Rms(n) => n.parameters(),
        }
    }
    fn set_training(&self, t: bool) {
        match self {
            NormKind::LayerNorm(n) => n.set_training(t),
            NormKind::Rms(n) => n.set_training(t),
        }
    }
}

pub enum MlpKind {
    Gpt(Mlp),
    SwiGlu(SwiGluMlp),
}

impl MlpKind {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            MlpKind::Gpt(m) => m.forward(x),
            MlpKind::SwiGlu(m) => m.forward(x),
        }
    }
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            MlpKind::Gpt(m) => m.parameters(),
            MlpKind::SwiGlu(m) => m.parameters(),
        }
    }
    fn set_training(&self, t: bool) {
        match self {
            MlpKind::Gpt(m) => m.set_training(t),
            MlpKind::SwiGlu(m) => m.set_training(t),
        }
    }
}

pub struct TransformerBlock {
    pub ln1: NormKind,
    pub attn: CausalSelfAttention,
    pub ln2: NormKind,
    pub mlp: MlpKind,
}

impl TransformerBlock {
    pub fn new_gpt(n_embd: usize, n_head: usize, dropout: f32, bias: bool) -> Self {
        Self {
            ln1: NormKind::LayerNorm(crate::nn::norm::LayerNorm::new(n_embd, 1e-5)),
            attn: CausalSelfAttention::new_gpt(n_embd, n_head, dropout, bias),
            ln2: NormKind::LayerNorm(crate::nn::norm::LayerNorm::new(n_embd, 1e-5)),
            mlp: MlpKind::Gpt(Mlp::new_gpt(n_embd, dropout, bias)),
        }
    }

    pub fn new_llama(n_embd: usize, n_head: usize, hidden: usize, max_seq: usize) -> Self {
        Self {
            ln1: NormKind::Rms(crate::nn::norm::RmsNorm::new(n_embd, 1e-6)),
            attn: CausalSelfAttention::new_llama(n_embd, n_head, 0.0, max_seq),
            ln2: NormKind::Rms(crate::nn::norm::RmsNorm::new(n_embd, 1e-6)),
            mlp: MlpKind::SwiGlu(SwiGluMlp::new(n_embd, hidden)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let a = self.attn.forward(&self.ln1.forward(x));
        let x = x.add(&a);
        let m = self.mlp.forward(&self.ln2.forward(&x));
        x.add(&m)
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.ln1.parameters();
        v.extend(self.attn.parameters());
        v.extend(self.ln2.parameters());
        v.extend(self.mlp.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.ln1.set_training(t);
        self.attn.set_training(t);
        self.ln2.set_training(t);
        self.mlp.set_training(t);
    }
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub bias: bool,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self { vocab_size: 65, block_size: 32, n_layer: 2, n_head: 2, n_embd: 32, dropout: 0.0, bias: true }
    }
}

pub struct GPT {
    pub cfg: GPTConfig,
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: crate::nn::norm::LayerNorm,
    pub lm_head: Linear,
    flag: TrainFlag,
}

impl GPT {
    pub fn new(cfg: GPTConfig) -> Self {
        let tok_emb = Embedding::new(cfg.vocab_size, cfg.n_embd);
        let pos_emb = Embedding::new(cfg.block_size, cfg.n_embd);
        let blocks = (0..cfg.n_layer).map(|_| TransformerBlock::new_gpt(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.bias)).collect();
        let ln_f = crate::nn::norm::LayerNorm::new(cfg.n_embd, 1e-5);
        let lm_head = Linear::new(cfg.n_embd, cfg.vocab_size, false);
        Self { cfg, tok_emb, pos_emb, blocks, ln_f, lm_head, flag: TrainFlag::new(true) }
    }

    /// `idx`: token ids len B*T, returns logits [B, T, V].
    pub fn forward(&self, idx: &[usize], b: usize, t: usize) -> Tensor {
        assert_eq!(idx.len(), b * t);
        assert!(t <= self.cfg.block_size, "sequence length {} > block_size {}", t, self.cfg.block_size);
        let tok = self.tok_emb.forward(idx, &[b, t]); // [B,T,C]
        let pos_ids: Vec<usize> = (0..t).collect();
        // pos emb [T,C] -> broadcast over B
        let pos = self.pos_emb.forward(&pos_ids, &[t]); // [T,C]
        // broadcast add: tok [B,T,C] + pos [T,C] — our broadcast aligns trailing dims,
        // [B,T,C] vs [T,C] broadcasts leading B. Good.
        let mut x = tok.add(&pos);
        x = x.dropout(self.cfg.dropout, self.flag.get());
        for blk in &self.blocks {
            x = blk.forward(&x);
        }
        x = self.ln_f.forward(&x);
        self.lm_head.forward(&x) // [B,T,V]
    }

    /// Cross-entropy loss over logits [B,T,V] and targets [B*T].
    pub fn loss(&self, logits: &Tensor, targets: &[usize]) -> Tensor {
        let s = logits.shape();
        let (b, t, v) = (s[0], s[1], s[2]);
        assert_eq!(targets.len(), b * t);
        let flat = logits.reshape(&[b * t, v]);
        flat.cross_entropy(targets)
    }

    /// Greedy / sampled generation. Returns full id sequence (prompt + new tokens).
    pub fn generate(&self, mut idx: Vec<usize>, max_new: usize, temperature: f32, top_k: Option<usize>) -> Vec<usize> {
        self.eval();
        no_grad(|| {
            for _ in 0..max_new {
                let t = idx.len().min(self.cfg.block_size);
                let start = idx.len() - t;
                let window = idx[start..].to_vec();
                let logits = self.forward(&window, 1, t); // [1,T,V]
                let last = logits.narrow(1, t - 1, 1).squeeze(1); // [1,V]
                let next = crate::sampling::sample_next(&last, temperature, top_k, None);
                idx.push(next);
            }
        });
        self.train();
        idx
    }

    pub fn num_params(&self) -> usize {
        crate::utils::count_params(&self.parameters())
    }
}

impl Module for GPT {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.tok_emb.parameters();
        v.extend(self.pos_emb.parameters());
        for b in &self.blocks {
            v.extend(b.parameters());
        }
        v.extend(self.ln_f.parameters());
        v.extend(self.lm_head.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.flag.set(t);
        for b in &self.blocks {
            b.set_training(t);
        }
        self.ln_f.set_training(t);
    }
    fn is_training(&self) -> bool { self.flag.get() }
}

// ---------------------------------------------------------------------------
// LLaMA-style (RMSNorm + RoPE + SwiGLU, no learned pos emb)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub hidden: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self { vocab_size: 32000, block_size: 128, n_layer: 2, n_head: 4, n_embd: 64, hidden: 128 }
    }
}

pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub tok_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub norm: crate::nn::norm::RmsNorm,
    pub lm_head: Linear,
    flag: TrainFlag,
}

impl LlamaModel {
    pub fn new(cfg: LlamaConfig) -> Self {
        let tok_emb = Embedding::new(cfg.vocab_size, cfg.n_embd);
        let blocks = (0..cfg.n_layer).map(|_| TransformerBlock::new_llama(cfg.n_embd, cfg.n_head, cfg.hidden, cfg.block_size)).collect();
        let norm = crate::nn::norm::RmsNorm::new(cfg.n_embd, 1e-6);
        let lm_head = Linear::new(cfg.n_embd, cfg.vocab_size, false);
        Self { cfg, tok_emb, blocks, norm, lm_head, flag: TrainFlag::new(true) }
    }

    pub fn forward(&self, idx: &[usize], b: usize, t: usize) -> Tensor {
        assert!(t <= self.cfg.block_size);
        let mut x = self.tok_emb.forward(idx, &[b, t]);
        for blk in &self.blocks {
            x = blk.forward(&x);
        }
        x = self.norm.forward(&x);
        self.lm_head.forward(&x)
    }

    pub fn loss(&self, logits: &Tensor, targets: &[usize]) -> Tensor {
        let s = logits.shape();
        let flat = logits.reshape(&[s[0] * s[1], s[2]]);
        flat.cross_entropy(targets)
    }
}

impl Module for LlamaModel {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.tok_emb.parameters();
        for b in &self.blocks {
            v.extend(b.parameters());
        }
        v.extend(self.norm.parameters());
        v.extend(self.lm_head.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.flag.set(t);
        for b in &self.blocks {
            b.set_training(t);
        }
    }
    fn is_training(&self) -> bool { self.flag.get() }
}
