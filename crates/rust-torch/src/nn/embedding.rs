//! Embedding layer.

use crate::nn::module::Module;
use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: Tensor, // [vocab, dim]
    pub vocab_size: usize,
    pub dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let data = Tensor::randn(&[vocab_size, dim]).data_vec().into_iter().map(|x| x * 0.02).collect();
        let weight = Tensor::new(data, vec![vocab_size, dim], true);
        Self { weight, vocab_size, dim }
    }

    /// `indices`: flat ids, `index_shape`: e.g. `[B, T]` -> output `[B, T, dim]`.
    pub fn forward(&self, indices: &[usize], index_shape: &[usize]) -> Tensor {
        Tensor::embedding(&self.weight, indices, index_shape)
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
