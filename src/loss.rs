//! Loss functions.

use crate::tensor::Tensor;

pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.mse(target)
}

pub fn cross_entropy(logits_2d: &Tensor, targets: &[usize]) -> Tensor {
    logits_2d.cross_entropy(targets)
}

/// Language-model loss: logits [B,T,V] + targets [B*T] -> scalar.
pub fn lm_loss(logits_3d: &Tensor, targets: &[usize]) -> Tensor {
    let s = logits_3d.shape();
    assert_eq!(s.len(), 3);
    let flat = logits_3d.reshape(&[s[0] * s[1], s[2]]);
    flat.cross_entropy(targets)
}

/// Binary cross-entropy with logits (stable), mean reduction.
pub fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Tensor {
    // stable: max(l,0) - l*t + log(1 + exp(-|l|))
    let relu = logits.relu();
    let term1 = relu.sub(&logits.mul(targets));
    let abs_l = logits.relu().add(&logits.neg().relu());
    let stable = abs_l.neg().exp().add_scalar(1.0).log();
    term1.add(&stable).mean()
}
