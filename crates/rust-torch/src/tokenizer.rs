//! Tokenizers: char-level + whitespace word-level (+ save/load).

use std::collections::{HashMap, HashSet};
use std::fs;

pub trait Tokenizer: Clone {
    fn encode(&self, s: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

/// Character-level tokenizer built from a corpus.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    pub stoi: HashMap<char, usize>,
    pub itos: Vec<char>,
}

impl CharTokenizer {
    pub fn build(corpus: &str) -> Self {
        let mut set: HashSet<char> = HashSet::new();
        for c in corpus.chars() {
            set.insert(c);
        }
        let mut itos: Vec<char> = set.into_iter().collect();
        itos.sort();
        let stoi: HashMap<char, usize> = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { stoi, itos }
    }

    pub fn encode_str(&self, s: &str) -> Vec<usize> {
        s.chars().map(|c| *self.stoi.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode_ids(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.itos[i % self.itos.len()]).collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let s: String = self.itos.iter().collect();
        fs::write(path, s)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        let itos: Vec<char> = s.chars().collect();
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Ok(Self { stoi, itos })
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, s: &str) -> Vec<usize> {
        self.encode_str(s)
    }
    fn decode(&self, ids: &[usize]) -> String {
        self.decode_ids(ids)
    }
    fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}

/// Whitespace word-level tokenizer.
#[derive(Clone, Debug)]
pub struct WordTokenizer {
    pub stoi: HashMap<String, usize>,
    pub itos: Vec<String>,
    pub unk: usize,
}

impl WordTokenizer {
    pub fn build(corpus: &str) -> Self {
        let mut set = HashSet::new();
        for w in corpus.split_whitespace() {
            set.insert(w.to_string());
        }
        let mut itos: Vec<String> = set.into_iter().collect();
        itos.sort();
        itos.insert(0, "<unk>".to_string());
        let stoi = itos.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        Self { stoi, itos, unk: 0 }
    }
}

impl Tokenizer for WordTokenizer {
    fn encode(&self, s: &str) -> Vec<usize> {
        s.split_whitespace().map(|w| *self.stoi.get(w).unwrap_or(&self.unk)).collect()
    }
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.itos[i % self.itos.len()].clone()).collect::<Vec<_>>().join(" ")
    }
    fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}
