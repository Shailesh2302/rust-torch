//! `Module` trait — PyTorch-like base for all layers.

use crate::tensor::Tensor;
use std::cell::Cell;
use std::rc::Rc;

/// Minimal PyTorch-like Module.
pub trait Module {
    /// All learnable tensors (shared handles — mutating data/grad affects the module).
    fn parameters(&self) -> Vec<Tensor>;
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.parameters().into_iter().enumerate().map(|(i, p)| (format!("param_{}", i), p)).collect()
    }
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
    fn train(&self) {
        self.set_training(true);
    }
    fn eval(&self) {
        self.set_training(false);
    }
    fn is_training(&self) -> bool {
        true
    }
    fn set_training(&self, _training: bool) {}
}

/// Shared training flag helper.
#[derive(Clone, Debug)]
pub struct TrainFlag {
    inner: Rc<Cell<bool>>,
}

impl TrainFlag {
    pub fn new(training: bool) -> Self {
        Self { inner: Rc::new(Cell::new(training)) }
    }
    pub fn get(&self) -> bool {
        self.inner.get()
    }
    pub fn set(&self, v: bool) {
        self.inner.set(v);
    }
}
