//! Standalone inference engine for FLAME checkpoints.
//!
//! Re-implements FLAME's LLaMA-style decoder and byte-level BPE tokenizer with no dependency on
//! the FLAME crates, so it compiles to WebAssembly and can record every intermediate value of a
//! forward pass for visualisation. Numerical agreement with FLAME itself is checked by the
//! `parity` crate in this workspace.

pub mod config;
pub mod model;
pub mod safetensors;
pub mod tokenizer;
pub mod trace;

#[cfg(target_arch = "wasm32")]
mod wasm;

pub use config::Config;
pub use model::Model;
pub use tokenizer::Tokenizer;
pub use trace::{Trace, TraceOptions};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::Error(format!($($arg)*))) };
}
pub(crate) use bail;
