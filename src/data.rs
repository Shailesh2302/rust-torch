//! Data utilities: text datasets + batch sampling for LM training.

use rand::Rng;
use rand::seq::SliceRandom;

/// A flat token stream cut into `[B, T]` input/target batches (next-token shift).
#[derive(Clone, Debug)]
pub struct TextDataset {
    pub tokens: Vec<usize>,
    pub block_size: usize,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, block_size: usize) -> Self {
        assert!(tokens.len() > block_size + 1);
        Self { tokens, block_size }
    }

    pub fn len(&self) -> usize {
        self.tokens.len() - self.block_size
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.len() <= self.block_size
    }

    /// Get one example: inputs `[T]`, targets `[T]` (shifted by one).
    pub fn get(&self, start: usize) -> (Vec<usize>, Vec<usize>) {
        let x = self.tokens[start..start + self.block_size].to_vec();
        let y = self.tokens[start + 1..start + 1 + self.block_size].to_vec();
        (x, y)
    }

    /// Random batch: `(inputs [B*T], targets [B*T], B, T)`.
    pub fn random_batch(&self, batch_size: usize) -> (Vec<usize>, Vec<usize>, usize, usize) {
        let mut rng = rand::thread_rng();
        let max_start = self.tokens.len() - self.block_size - 1;
        let mut xs = Vec::with_capacity(batch_size * self.block_size);
        let mut ys = Vec::with_capacity(batch_size * self.block_size);
        for _ in 0..batch_size {
            let s = rng.gen_range(0..=max_start);
            let (x, y) = self.get(s);
            xs.extend(x);
            ys.extend(y);
        }
        (xs, ys, batch_size, self.block_size)
    }

    pub fn split_train_val(tokens: Vec<usize>, block_size: usize, val_frac: f32) -> (Self, Self) {
        let n_val = ((tokens.len() as f32) * val_frac) as usize;
        let n_train = tokens.len() - n_val;
        let train = tokens[..n_train].to_vec();
        let val = tokens[n_train..].to_vec();
        (Self::new(train, block_size), Self::new(val, block_size))
    }
}

/// Simple shuffling batch iterator over start positions.
pub struct BatchIter {
    starts: Vec<usize>,
    pos: usize,
    pub batch_size: usize,
}

impl BatchIter {
    pub fn new(dataset_len: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut starts: Vec<usize> = (0..dataset_len).collect();
        if shuffle {
            starts.shuffle(&mut rand::thread_rng());
        }
        Self { starts, pos: 0, batch_size }
    }
    pub fn next_batch(&mut self, ds: &TextDataset) -> Option<(Vec<usize>, Vec<usize>, usize, usize)> {
        if self.pos >= self.starts.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.starts.len());
        let mut xs = vec![];
        let mut ys = vec![];
        for &s in &self.starts[self.pos..end] {
            let s = s % ds.len().max(1);
            let (x, y) = ds.get(s);
            xs.extend(x);
            ys.extend(y);
        }
        let b = end - self.pos;
        self.pos = end;
        Some((xs, ys, b, ds.block_size))
    }
    pub fn reset(&mut self, shuffle: bool) {
        self.pos = 0;
        if shuffle {
            self.starts.shuffle(&mut rand::thread_rng());
        }
    }
}
