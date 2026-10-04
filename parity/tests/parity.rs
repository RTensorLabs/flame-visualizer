//! Parity between flame-engine and FLAME itself, on the real TinyStories-5M checkpoint.
//!
//! FLAME is used strictly read-only. Set FLAME_DIR to point at a FLAME checkout
//! (default: ../../FLAME-main/FLAME relative to this crate).

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use flame::model::LanguageModel;
use flame_engine::{Config, Model, Tokenizer, TraceOptions};
use std::path::PathBuf;

fn flame_dir() -> PathBuf {
    std::env::var("FLAME_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../FLAME-main/FLAME"))
}
fn weights() -> PathBuf {
    flame_dir().join("runs/tinystories-5m/best/model.safetensors")
}
fn tok_path() -> PathBuf {
    flame_dir().join("data/tokenizers/tinystories-4k.json")
}

fn flame_cfg() -> flame::config::ModelConfig {
    flame::config::RunConfig::load(flame_dir().join("runs/tinystories-5m/config.toml")).unwrap().model
}

fn engine() -> (Model, Tokenizer) {
    let c = flame_cfg();
    let cfg = Config {
        vocab_size: c.vocab_size,
        d_model: c.d_model,
        n_layers: c.n_layers,
        n_heads: c.n_heads,
        mlp_hidden: c.mlp_hidden(),
        max_seq_len: c.max_seq_len,
        rope_theta: c.rope_theta,
        norm_eps: c.norm_eps,
        tie_embeddings: c.tie_embeddings,
    };
    let bytes = std::fs::read(weights()).unwrap();
    let tok = Tokenizer::from_json(&std::fs::read_to_string(tok_path()).unwrap()).unwrap();
    (Model::from_safetensors(&bytes, cfg).unwrap(), tok)
}

fn flame_model() -> flame::model::Transformer {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights()], DType::F32, &Device::Cpu).unwrap() };
    flame::model::Transformer::new(&flame_cfg(), vb).unwrap()
}

fn flame_logits(m: &flame::model::Transformer, ids: &[u32]) -> Vec<Vec<f32>> {
    let input = Tensor::new(ids, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let logits = m.forward(&input).unwrap().logits;
    (0..ids.len()).map(|i| logits.i((0, i)).unwrap().to_vec1::<f32>().unwrap()).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

const PROMPTS: &[&str] = &[
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy named Tom went to the park with his mom. He saw a big red ball",
    "The cat sat on the",
    "Sue was sad. Her toy was broken!",
    "\"Can I play too?\" asked the dog.",
];

#[test]
fn tokenizer_matches_flame_on_corpus() {
    let json = std::fs::read_to_string(tok_path()).unwrap();
    let ours = Tokenizer::from_json(&json).unwrap();
    let theirs = flame::tokenizer::Tokenizer::load(tok_path()).unwrap();
    assert_eq!(ours.vocab_size(), theirs.vocab_size());
    for id in 0..ours.vocab_size() as u32 {
        assert_eq!(ours.decode(&[id]), theirs.decode(&[id]), "decode id {id}");
    }
    let mut corpus: Vec<String> = PROMPTS.iter().map(|s| s.to_string()).collect();
    corpus.extend(["  double  spaces\n\n  indented", "Grüße, ζ! 🚀 café 12345678", "tab\tsep\r\nCRLF", ""].map(String::from));
    let raw = flame_dir().join("data/raw/tinystories-train.txt");
    if let Ok(f) = std::fs::File::open(&raw) {
        use std::io::Read;
        let mut buf = vec![0u8; 2_000_000];
        let n = f.take(2_000_000).read(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf[..n]).into_owned();
        corpus.extend(text.split("<|endoftext|>").map(String::from));
    }
    let mut n_tokens = 0;
    for s in &corpus {
        let (a, b) = (ours.encode(s), theirs.encode(s));
        assert_eq!(a, b, "encode mismatch on {s:?}");
        n_tokens += a.len();
    }
    eprintln!("tokenizer parity on {} texts / {n_tokens} tokens", corpus.len());
}

#[test]
fn logits_match_flame_at_every_position() {
    let (ours, tok) = engine();
    let theirs = flame_model();
    let mut worst = 0f32;
    for p in PROMPTS {
        let ids = tok.encode(p);
        let reference = flame_logits(&theirs, &ids);
        for len in 1..=ids.len() {
            let got = ours.logits_last(&ids[..len]).unwrap();
            let err = max_abs(&got, &reference[len - 1]);
            worst = worst.max(err);
            assert!(err < 1e-3, "prompt {p:?} pos {len}: max |Δlogit| = {err}");
        }
    }
    eprintln!("worst |Δlogit| = {worst:e}");
}

#[test]
fn trace_logits_equal_fast_path() {
    let (ours, tok) = engine();
    let ids = tok.encode(PROMPTS[1]);
    let t = ours.trace(&ids, &TraceOptions::default()).unwrap();
    let (_, l) = t.get("logits_last").unwrap();
    assert_eq!(l, ours.logits_last(&ids).unwrap().as_slice());
}

#[test]
fn greedy_generation_matches_flame() {
    let (ours, tok) = engine();
    let theirs = flame_model();
    let prompt = tok.encode("Once upon a time");
    let opts = flame::generate::SampleOptions { max_new_tokens: 40, temperature: 0.0, top_k: 0, seed: 0, stop_token: None };
    let reference = flame::generate::generate(&theirs, &prompt, &opts, &Device::Cpu).unwrap();
    let mut ids = prompt.clone();
    let mut got = Vec::new();
    for _ in 0..40 {
        let l = ours.logits_last(&ids).unwrap();
        let next = l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        ids.push(next);
        got.push(next);
    }
    eprintln!("greedy: {:?}", tok.decode(&got));
    assert_eq!(got, reference);
}

#[test]
fn trace_is_internally_consistent() {
    let (ours, tok) = engine();
    let cfg = ours.config().clone();
    let ids = tok.encode(PROMPTS[0]);
    let t = ours.trace(&ids, &TraceOptions::default()).unwrap();
    let n = ids.len();
    for l in 0..cfg.n_layers {
        let (_, probs) = t.get(&format!("L{l}.probs")).unwrap();
        for h in 0..cfg.n_heads {
            for i in 0..n {
                let row = &probs[(h * n + i) * n..(h * n + i + 1) * n];
                let s: f32 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "probs row sum {s}");
                assert!(row[i + 1..].iter().all(|&v| v == 0.0), "future attention must be 0");
            }
        }
        let (_, contrib) = t.get(&format!("L{l}.head_contrib")).unwrap();
        let (_, attn) = t.get(&format!("L{l}.attn_out")).unwrap();
        let d = cfg.d_model;
        for k in 0..n * d {
            let s: f32 = (0..cfg.n_heads).map(|h| contrib[h * n * d + k]).sum();
            assert!((s - attn[k]).abs() < 1e-4, "Σ head_contrib ≠ attn_out");
        }
        if l + 1 < cfg.n_layers {
            assert_eq!(t.get(&format!("L{l}.x_out")).unwrap().1, t.get(&format!("L{}.x_in", l + 1)).unwrap().1);
        }
    }
}

#[test]
fn causal_prefix_is_unaffected_by_later_tokens() {
    let (ours, tok) = engine();
    let a = tok.encode("The cat sat on the mat and");
    let mut b = a.clone();
    *b.last_mut().unwrap() = tok.encode(" dog")[0];
    let opts = TraceOptions { lens_top_k: 0, head_contrib: false };
    let (ta, tb) = (ours.trace(&a, &opts).unwrap(), ours.trace(&b, &opts).unwrap());
    let n = a.len();
    let d = ours.config().d_model;
    let (_, xa) = ta.get("L5.x_out").unwrap();
    let (_, xb) = tb.get("L5.x_out").unwrap();
    assert_eq!(&xa[..(n - 1) * d], &xb[..(n - 1) * d]);
}

#[test]
fn timing_report() {
    let (ours, tok) = engine();
    let text = "Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big dog in the park. The dog was very friendly and wanted to play with her too. Lily";
    let ids: Vec<u32> = tok.encode(text).into_iter().take(64).collect();
    let t0 = std::time::Instant::now();
    let tr = ours.trace(&ids, &TraceOptions::default()).unwrap();
    eprintln!("native trace T={} : {:?}, arena {} floats", ids.len(), t0.elapsed(), tr.data.len());
}
