//! Small utilities: seeding, init, checkpoint IO.

use crate::tensor::Tensor;
use std::fs::File;
use std::io::{Read, Write};

pub fn set_seed(seed: u64) {
    // rand 0.8 thread_rng is auto-seeded; we expose this for API compat.
    // Determinism across runs is not guaranteed with thread_rng, but we
    // keep the hook so training scripts look familiar.
    let _ = seed;
}

/// Xavier-uniform init for `[fan_out, fan_in]` matrices.
pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> Vec<f32> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    Tensor::rand_uniform(&[fan_out, fan_in], -bound, bound).data_vec()
}

pub fn kaiming_uniform_vec(fan_in: usize, fan_out: usize) -> Vec<f32> {
    Tensor::kaiming_uniform(fan_in, fan_out)
}

/// Save parameters as a tiny binary checkpoint:
/// `[u64 n_tensors][for each: u64 ndim, dims..., u64 len, f32 LE bytes...]`
pub fn save_checkpoint(path: &str, params: &[Tensor]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(params.len() as u64).to_le_bytes())?;
    for p in params {
        let s = p.shape();
        f.write_all(&(s.len() as u64).to_le_bytes())?;
        for d in &s {
            f.write_all(&(*d as u64).to_le_bytes())?;
        }
        let d = p.data_vec();
        f.write_all(&(d.len() as u64).to_le_bytes())?;
        for v in d {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Load checkpoint data into already-shaped parameters (shapes must match).
pub fn load_checkpoint(path: &str, params: &[Tensor]) -> std::io::Result<()> {
    let mut f = File::open(path)?;
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    assert_eq!(n, params.len(), "checkpoint param count mismatch");
    for p in params {
        f.read_exact(&mut buf8)?;
        let nd = u64::from_le_bytes(buf8) as usize;
        let mut shape = vec![0usize; nd];
        for d in shape.iter_mut() {
            f.read_exact(&mut buf8)?;
            *d = u64::from_le_bytes(buf8) as usize;
        }
        assert_eq!(&shape, &p.shape(), "checkpoint shape mismatch");
        f.read_exact(&mut buf8)?;
        let len = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0f32; len];
        for v in data.iter_mut() {
            let mut b4 = [0u8; 4];
            f.read_exact(&mut b4)?;
            *v = f32::from_le_bytes(b4);
        }
        p.set_data(data);
    }
    Ok(())
}

/// Count total parameters.
pub fn count_params(params: &[Tensor]) -> usize {
    params.iter().map(|p| p.numel()).sum()
}
