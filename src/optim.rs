//! Optimizers: SGD, Adam, AdamW (PyTorch-like API).

use crate::tensor::Tensor;

fn get_grad(p: &Tensor) -> Vec<f32> {
    p.inner.grad.borrow().clone()
}

fn add_to_data(p: &Tensor, delta: &[f32]) {
    let mut d = p.inner.data.borrow_mut();
    assert_eq!(d.len(), delta.len());
    for (a, b) in d.iter_mut().zip(delta.iter()) {
        *a += *b;
    }
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    vel: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let vel = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Self { params, lr, momentum: 0.0, weight_decay: 0.0, vel }
    }
    pub fn with_momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let g = get_grad(p);
            let n = g.len();
            let mut upd = vec![0.0; n];
            for j in 0..n {
                let mut gj = g[j];
                if self.weight_decay != 0.0 {
                    gj += self.weight_decay * p.inner.data.borrow()[j];
                }
                if self.momentum != 0.0 {
                    self.vel[i][j] = self.momentum * self.vel[i][j] + gj;
                    upd[j] = -self.lr * self.vel[i][j];
                } else {
                    upd[j] = -self.lr * gj;
                }
            }
            add_to_data(p, &upd);
        }
    }
}

// ---------------------------------------------------------------------------
// Adam / AdamW
// ---------------------------------------------------------------------------

pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0, m, v, t: 0 }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
        self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
    pub fn step(&mut self) {
        self.t += 1;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, p) in self.params.iter().enumerate() {
            let g = get_grad(p);
            let mut upd = vec![0.0; g.len()];
            {
                let d = p.inner.data.borrow();
                for j in 0..g.len() {
                    let mut gj = g[j];
                    if self.weight_decay != 0.0 {
                        // Adam with L2 (AdamW if decoupled flag — here coupled; AdamW below is decoupled)
                        gj += self.weight_decay * d[j];
                    }
                    self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * gj;
                    self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * gj * gj;
                    let mh = self.m[i][j] / bc1;
                    let vh = self.v[i][j] / bc2;
                    upd[j] = -self.lr * mh / (vh.sqrt() + self.eps);
                }
            }
            add_to_data(p, &upd);
        }
    }
}

/// AdamW with decoupled weight decay (the standard LLM optimizer).
pub struct AdamW {
    inner: Adam,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { inner: Adam::new(params, lr) }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner.betas = (b1, b2);
        self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.inner.weight_decay = wd;
        self
    }
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.inner.eps = eps;
        self
    }
    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }
    pub fn step(&mut self) {
        // decoupled: apply decay directly to params before adaptive step
        if self.inner.weight_decay != 0.0 {
            let wd = self.inner.weight_decay;
            let lr = self.inner.lr;
            for p in &self.inner.params {
                let mut d = p.inner.data.borrow_mut();
                for x in d.iter_mut() {
                    *x -= lr * wd * *x;
                }
            }
            let saved = self.inner.weight_decay;
            self.inner.weight_decay = 0.0;
            self.inner.step();
            self.inner.weight_decay = saved;
        } else {
            self.inner.step();
        }
    }
    pub fn set_lr(&mut self, lr: f32) {
        self.inner.lr = lr;
    }
    pub fn lr(&self) -> f32 {
        self.inner.lr
    }
}

/// Cosine LR schedule with warmup (common for LLM training).
pub fn cosine_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step < warmup {
        return max_lr * (step as f32 + 1.0) / warmup as f32;
    }
    if step >= total {
        return min_lr;
    }
    let progress = (step - warmup) as f32 / (total - warmup) as f32;
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
}
