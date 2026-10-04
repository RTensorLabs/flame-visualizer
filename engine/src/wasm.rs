//! JavaScript bindings (wasm-bindgen). Used from a Web Worker by the frontend.

use crate::{Config, Model, Tokenizer, TraceOptions};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn js(e: crate::Error) -> JsError {
    JsError::new(&e.0)
}

#[wasm_bindgen]
pub struct Engine {
    model: Model,
    tok: Tokenizer,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8], config_json: &str, tokenizer_json: &str) -> Result<Engine, JsError> {
        let cfg = Config::from_json(config_json).map_err(js)?;
        let tok = Tokenizer::from_json(tokenizer_json).map_err(js)?;
        if tok.vocab_size() > cfg.vocab_size {
            return Err(JsError::new("tokenizer vocabulary is larger than the model's"));
        }
        let model = Model::from_safetensors(weights, cfg).map_err(js)?;
        Ok(Engine { model, tok })
    }

    pub fn config_json(&self) -> String {
        serde_json::to_string(self.model.config()).unwrap()
    }

    pub fn hidden(&self) -> usize {
        self.model.config().hidden()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    /// Raw bytes of a single token, for display (a token can be part of a UTF-8 character).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.tok.token_bytes(id)
    }

    pub fn logits_last(&self, ids: &[u32]) -> Result<Vec<f32>, JsError> {
        self.model.logits_last(ids).map_err(js)
    }

    pub fn trace(&self, ids: &[u32], lens_top_k: usize, head_contrib: bool) -> Result<TraceResult, JsError> {
        let t = self.model.trace(ids, &TraceOptions { lens_top_k, head_contrib }).map_err(js)?;
        Ok(TraceResult { manifest: t.manifest_json(), data: t.data })
    }
}

#[wasm_bindgen]
pub struct TraceResult {
    manifest: String,
    data: Vec<f32>,
}

#[wasm_bindgen]
impl TraceResult {
    pub fn manifest(&self) -> String {
        self.manifest.clone()
    }

    /// Copies the arena into a fresh `Float32Array` (transferable to the main thread).
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
}
