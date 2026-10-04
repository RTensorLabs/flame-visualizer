//! Model hyper-parameters, read from the exported `config.json`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// 0 means FLAME's LLaMA default (see [`llama_mlp_hidden`]).
    #[serde(default)]
    pub mlp_hidden: usize,
    pub max_seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_true")]
    pub tie_embeddings: bool,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}

/// FLAME's MLP sizing rule: ⌈8·d/3⌉ rounded up to a multiple of 64.
pub fn llama_mlp_hidden(d_model: usize) -> usize {
    let h = (8 * d_model).div_ceil(3);
    h.div_ceil(64) * 64
}

impl Config {
    pub fn from_json(s: &str) -> crate::Result<Self> {
        let c: Config = serde_json::from_str(s).map_err(|e| crate::Error(format!("config.json: {e}")))?;
        c.validate()?;
        Ok(c)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn hidden(&self) -> usize {
        if self.mlp_hidden > 0 {
            self.mlp_hidden
        } else {
            llama_mlp_hidden(self.d_model)
        }
    }

    pub fn validate(&self) -> crate::Result<()> {
        if self.vocab_size == 0 || self.d_model == 0 || self.n_layers == 0 || self.n_heads == 0 {
            crate::bail!("config: sizes must be non-zero");
        }
        if self.d_model % self.n_heads != 0 || self.head_dim() % 2 != 0 {
            crate::bail!("config: d_model must split into an even head_dim");
        }
        Ok(())
    }
}
