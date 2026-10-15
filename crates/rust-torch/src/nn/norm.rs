//! LayerNorm / RMSNorm / Dropout (differentiable via Tensor ops).

use crate::nn::module::{Module, TrainFlag};
use crate::tensor::Tensor;

pub struct LayerNorm {
    pub weight: Tensor, // [dim]
    pub bias: Tensor,   // [dim]
    pub eps: f32,
    pub dim: usize,
    flag: TrainFlag,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::new(vec![1.0; dim], vec![dim], true),
            bias: Tensor::new(vec![0.0; dim], vec![dim], true),
            eps,
            dim,
            flag: TrainFlag::new(true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // normalize over last dim (fully differentiable composition)
        let mean = x.mean_dim(-1, true);
        let diff = x.sub(&mean);
        let var = diff.mul(&diff).mean_dim(-1, true);
        let eps_t = Tensor::full(&var.shape(), self.eps, false);
        // norm = (x - mean) / sqrt(var + eps)
        let denom = var.add(&eps_t).sqrt();
        let norm = diff.div(&denom);
        // scale + shift with broadcasting over last dim
        norm.mul(&self.weight).add(&self.bias)
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    fn set_training(&self, t: bool) { self.flag.set(t); }
    fn is_training(&self) -> bool { self.flag.get() }
}

pub struct RmsNorm {
    pub weight: Tensor, // [dim]
    pub eps: f32,
    pub dim: usize,
    flag: TrainFlag,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: Tensor::new(vec![1.0; dim], vec![dim], true), eps, dim, flag: TrainFlag::new(true) }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let sq = x.mul(x);
        let m = sq.mean_dim(-1, true);
        let eps_t = Tensor::full(&m.shape(), self.eps, false);
        let rms = m.add(&eps_t).sqrt();
        let normed = x.div(&rms);
        normed.mul(&self.weight)
    }
}

impl Module for RmsNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
    fn set_training(&self, t: bool) { self.flag.set(t); }
    fn is_training(&self) -> bool { self.flag.get() }
}

pub struct Dropout {
    pub p: f32,
    flag: TrainFlag,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p, flag: TrainFlag::new(true) }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.dropout(self.p, self.flag.get())
    }
}

impl Module for Dropout {
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn set_training(&self, t: bool) { self.flag.set(t); }
    fn is_training(&self) -> bool { self.flag.get() }
}
