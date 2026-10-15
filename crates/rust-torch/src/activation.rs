//! Activation functions (differentiable, built from Tensor ops).

use crate::tensor::Tensor;

pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

pub fn gelu(x: &Tensor) -> Tensor {
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let x3 = x.mul(x).mul(x).mul_scalar(0.044715);
    let inner = x.add(&x3).mul_scalar(c);
    let t = inner.tanh();
    let one = Tensor::ones(&t.shape(), false);
    // need broadcast ones + t: t shape [...], ones same shape
    let gate = t.add(&one).mul_scalar(0.5);
    x.mul(&gate)
}

pub fn gelu_exact(_x: &Tensor) -> Tensor {
    // alias — exact erf version approximated by tanh version (no std erf in core)
    gelu(_x)
}

pub fn silu(x: &Tensor) -> Tensor {
    // x * sigmoid(x), sigmoid = 1 / (1 + exp(-x))
    let neg = x.neg();
    let e = neg.exp();
    let ones = Tensor::ones(&e.shape(), false);
    let denom = ones.add(&e);
    let one = Tensor::ones(&x.shape(), false);
    let sig = one.div(&denom);
    x.mul(&sig)
}
pub fn swish(x: &Tensor) -> Tensor {
    silu(x)
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    let neg = x.neg();
    let e = neg.exp();
    let ones = Tensor::ones(&e.shape(), false);
    let denom = ones.add(&e);
    let one = Tensor::ones(&x.shape(), false);
    one.div(&denom)
}

pub fn tanh_act(x: &Tensor) -> Tensor {
    x.tanh()
}

pub fn softmax(x: &Tensor, dim: isize) -> Tensor {
    x.softmax(dim)
}

pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    x.log_softmax(dim)
}
