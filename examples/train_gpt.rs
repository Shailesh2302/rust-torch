//! Train a tiny char-level GPT on a toy corpus with rusTorch.
//! Run (release recommended): `cargo run --release --example train_gpt`

use rust_torch::data::TextDataset;
use rust_torch::nn::{GPT, GPTConfig, Module};
use rust_torch::optim::{cosine_lr, AdamW};
use rust_torch::tokenizer::{CharTokenizer, Tokenizer};

fn main() {
    let corpus = "hello world! this is rustorch, a tiny pytorch in rust. \
        we build llms from scratch with tensors, autograd, attention, and sampling. \
        hello rust, hello tensors, hello transformers! ".repeat(20);

    let tok = CharTokenizer::build(&corpus);
    println!("vocab size: {}", tok.vocab_size());
    let ids = tok.encode(&corpus);
    println!("tokens: {}", ids.len());

    let block_size = 16usize;
    let (train, val) = TextDataset::split_train_val(ids, block_size, 0.1);

    let cfg = GPTConfig {
        vocab_size: tok.vocab_size(),
        block_size,
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
        bias: true,
    };
    let model = GPT::new(cfg.clone());
    println!("params: {}", model.num_params());

    let mut opt = AdamW::new(model.parameters(), 3e-3);
    opt = opt.with_betas(0.9, 0.999).with_weight_decay(0.01);

    let steps = 120usize;
    let batch_size = 4usize;
    for step in 0..steps {
        let lr = cosine_lr(step, 20, steps, 3e-3, 3e-4);
        opt.set_lr(lr);
        let (xs, ys, b, t) = train.random_batch(batch_size);
        opt.zero_grad();
        let logits = model.forward(&xs, b, t);
        let loss = model.loss(&logits, &ys);
        let lv = loss.item();
        loss.backward();
        // grad clip
        for p in model.parameters() {
            p.clip_grad(-1.0, 1.0);
        }
        opt.step();
        if step % 20 == 0 {
            let (vxs, vys, vb, vt) = val.random_batch(batch_size);
            let vloss = rust_torch::tensor::no_grad(|| model.loss(&model.forward(&vxs, vb, vt), &vys).item());
            println!("step {step:4} lr {lr:.5} train {lv:.4} val {vloss:.4}");
        }
    }

    // sample
    let prompt = "hello ";
    let ids = tok.encode(prompt);
    let out = model.generate(ids, 80, 0.8, Some(20));
    println!("--- sample ---\n{}", tok.decode(&out));

    rust_torch::utils::save_checkpoint("/tmp/rustorch_gpt.bin", &model.parameters()).unwrap();
    println!("saved to /tmp/rustorch_gpt.bin");
}
