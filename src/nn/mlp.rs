//! MLP blocks: GPT-style (GELU) and LLaMA-style SwiGLU.

use crate::activation::{gelu, silu};
use crate::nn::linear::Linear;
use crate::nn::module::{Module, TrainFlag};
use crate::tensor::Tensor;

pub struct Mlp {
    pub fc: Linear,
    pub proj: Linear,
    pub dropout_p: f32,
    flag: TrainFlag,
}

impl Mlp {
    pub fn new_gpt(n_embd: usize, dropout: f32, bias: bool) -> Self {
        Self {
            fc: Linear::new(n_embd, 4 * n_embd, bias),
            proj: Linear::new(4 * n_embd, n_embd, bias),
            dropout_p: dropout,
            flag: TrainFlag::new(true),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc.forward(x);
        let h = gelu(&h);
        let h = self.proj.forward(&h);
        h.dropout(self.dropout_p, self.flag.get())
    }
}

impl Module for Mlp {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.fc.parameters();
        v.extend(self.proj.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.flag.set(t);
        self.fc.set_training(t);
        self.proj.set_training(t);
    }
    fn is_training(&self) -> bool { self.flag.get() }
}

/// SwiGLU MLP (LLaMA): down(silu(gate) * up)
pub struct SwiGluMlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    hidden: usize,
}

impl SwiGluMlp {
    pub fn new(n_embd: usize, hidden: usize) -> Self {
        Self {
            gate: Linear::new(n_embd, hidden, false),
            up: Linear::new(n_embd, hidden, false),
            down: Linear::new(hidden, n_embd, false),
            hidden,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let g = self.gate.forward(x);
        let u = self.up.forward(x);
        let h = silu(&g).mul(&u);
        self.down.forward(&h)
    }
}

impl Module for SwiGluMlp {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.gate.parameters();
        v.extend(self.up.parameters());
        v.extend(self.down.parameters());
        v
    }
    fn set_training(&self, t: bool) {
        self.gate.set_training(t);
        self.up.set_training(t);
        self.down.set_training(t);
    }
}
