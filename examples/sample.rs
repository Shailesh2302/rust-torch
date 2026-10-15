//! Inference demo: build a tiny GPT and sample from it.
//! Run: `cargo run --example sample`

use rust_torch::nn::{GPT, GPTConfig};
use rust_torch::tokenizer::{CharTokenizer, Tokenizer};

fn main() {
    let corpus = "abcdefghijklmnopqrstuvwxyz .,!?";
    let tok = CharTokenizer::build(corpus);
    let cfg = GPTConfig {
        vocab_size: tok.vocab_size(),
        block_size: 16,
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
        bias: true,
    };
    let model = GPT::new(cfg);
    println!("params: {}", model.num_params());

    for temp in [0.0, 0.7] {
        let prompt = "hello";
        let ids = tok.encode(prompt);
        let out = model.generate(ids, 40, temp, Some(10));
        println!("temp {temp}: {}", tok.decode(&out));
    }
}
