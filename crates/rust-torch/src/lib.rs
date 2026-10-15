//! # rusTorch — PyTorch-like deep learning + LLM library in pure Rust
//!
//! Build and train transformers / LLMs with a familiar API:
//!
//! ```rust
//! use rust_torch::{Tensor, nn::{GPT, GPTConfig, Module}, optim::AdamW};
//!
//! let cfg = GPTConfig { vocab_size: 65, block_size: 16, n_layer: 2, n_head: 2, n_embd: 32, dropout: 0.0, bias: true };
//! let model = GPT::new(cfg);
//! let ids = vec![1usize; 1*16];
//! let logits = model.forward(&ids, 1, 16); // [1, 16, 65]
//! assert_eq!(logits.shape(), vec![1, 16, 65]);
//! ```
//!
//! Modules: [`tensor`], [`nn`], [`optim`], [`loss`], [`data`], [`tokenizer`], [`sampling`], [`activation`], [`utils`].

pub mod tensor;
pub mod activation;
pub mod nn;
pub mod optim;
pub mod loss;
pub mod data;
pub mod tokenizer;
pub mod sampling;
pub mod utils;

// convenient re-exports (torch-like top level)
pub use tensor::{Tensor, no_grad, is_no_grad};
pub use nn::{Module, Linear, Embedding, LayerNorm, RmsNorm, Dropout, GPT, GPTConfig, LlamaModel, LlamaConfig};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module;

    #[test]
    fn tensor_add_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = Tensor::new(vec![10.0, 20.0], vec![2], false);
        let c = a.add(&b);
        assert_eq!(c.data_vec(), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn autograd_simple() {
        let x = Tensor::new(vec![2.0, 3.0], vec![2], true);
        let y = x.mul(&x).sum(); // x^2 sum = 13
        assert!((y.item() - 13.0).abs() < 1e-5);
        y.backward();
        assert!((x.grad_vec()[0] - 4.0).abs() < 1e-4);
        assert!((x.grad_vec()[1] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn matmul_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);
        let c = a.matmul(&b);
        assert_eq!(c.data_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3], false);
        let s = a.softmax(-1);
        let d = s.data_vec();
        assert!((d[0] + d[1] + d[2] - 1.0).abs() < 1e-5);
        assert!((d[3] + d[4] + d[5] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_forward_shape() {
        let lin = Linear::new(4, 8, true);
        let x = Tensor::randn(&[2, 3, 4]);
        let y = lin.forward(&x);
        assert_eq!(y.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn gpt_forward_shape() {
        let cfg = GPTConfig { vocab_size: 16, block_size: 8, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0.0, bias: true };
        let m = GPT::new(cfg);
        let ids = vec![0usize; 2 * 8];
        let logits = m.forward(&ids, 2, 8);
        assert_eq!(logits.shape(), vec![2, 8, 16]);
    }

    #[test]
    fn gpt_loss_decreases_one_step() {
        use crate::optim::AdamW;
        let cfg = GPTConfig { vocab_size: 8, block_size: 4, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0.0, bias: true };
        let m = GPT::new(cfg);
        let mut opt = AdamW::new(m.parameters(), 1e-2);
        let ids: Vec<usize> = vec![1, 2, 3, 4, 2, 3, 4, 5];
        let tgt: Vec<usize> = vec![2, 3, 4, 5, 3, 4, 5, 6];
        let l0 = m.loss(&m.forward(&ids, 2, 4), &tgt).item();
        opt.zero_grad();
        let loss = m.loss(&m.forward(&ids, 2, 4), &tgt);
        loss.backward();
        opt.step();
        let l1 = m.loss(&crate::tensor::no_grad(|| m.forward(&ids, 2, 4)), &tgt).item();
        assert!(l1 <= l0 + 1e-3, "loss should not explode: {} -> {}", l0, l1);
    }

    #[test]
    fn cross_entropy_known_value() {
        // uniform logits -> loss = ln(V)
        let logits = Tensor::zeros(&[2, 4], true);
        let loss = logits.cross_entropy(&[0, 3]);
        assert!((loss.item() - (4.0f32).ln()).abs() < 1e-4);
        loss.backward();
    }
}
