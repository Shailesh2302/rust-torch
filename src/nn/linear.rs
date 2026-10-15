//! Linear layer: `y = x @ Wᵀ + b`.

use crate::nn::module::Module;
use crate::nn::module::TrainFlag;
use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor, // [out, in]
    pub bias: Option<Tensor>, // [out]
    pub in_features: usize,
    pub out_features: usize,
    flag: TrainFlag,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let data = Tensor::kaiming_uniform(in_features, out_features);
        let weight = Tensor::new(data, vec![out_features, in_features], true);
        let bias = if bias {
            Some(Tensor::new(vec![0.0; out_features], vec![out_features], true))
        } else {
            None
        };
        Self { weight, bias, in_features, out_features, flag: TrainFlag::new(true) }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., in] @ W.T [in, out] -> [..., out]
        let in_shape = x.shape();
        assert!(*in_shape.last().unwrap() == self.in_features, "Linear in_features mismatch: got {:?} expected {}", in_shape, self.in_features);
        let wt = self.weight.transpose(); // [in, out]
        // flatten leading dims
        let outer: usize = if in_shape.len() == 1 { 1 } else { in_shape[..in_shape.len()-1].iter().product() };
        let x2d = x.reshape(&[outer, self.in_features]);
        let mut y2d = x2d.matmul(&wt); // [outer, out]
        if let Some(b) = &self.bias {
            // broadcast bias [out] over rows: manual add via broadcasting
            // y2d shape [outer, out], bias [out] -> broadcast ok
            y2d = y2d.add(b);
        }
        let mut out_shape = if in_shape.len() == 1 { vec![self.out_features] } else { in_shape[..in_shape.len()-1].to_vec() };
        out_shape.push(self.out_features);
        if out_shape != y2d.shape() {
            y2d.reshape(&out_shape)
        } else {
            y2d
        }
    }

    pub fn no_bias(mut self) -> Self {
        self.bias = None;
        self
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut v = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            v.push(b.clone());
        }
        v
    }
    fn set_training(&self, t: bool) { self.flag.set(t); }
    fn is_training(&self) -> bool { self.flag.get() }
}
