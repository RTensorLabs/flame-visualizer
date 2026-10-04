//! Activation trace: every recorded tensor lives in one contiguous f32 arena, described by a
//! manifest of `{name, offset, shape}`. The arena crosses into JavaScript as a single
//! `Float32Array` with no per-tensor allocations.

use serde::Serialize;

#[derive(Debug, Clone)]
pub struct TraceOptions {
    /// Top-k entries kept per position for the logit lens (0 disables it).
    pub lens_top_k: usize,
    /// Record each head's contribution to the attention output (H × T × D per layer).
    pub head_contrib: bool,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self { lens_top_k: 5, head_contrib: true }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    pub name: String,
    pub offset: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub ids: Vec<u32>,
    pub entries: Vec<Entry>,
}

pub struct Trace {
    pub data: Vec<f32>,
    pub manifest: Manifest,
}

impl Trace {
    pub fn get(&self, name: &str) -> Option<(&[usize], &[f32])> {
        let e = self.manifest.entries.iter().find(|e| e.name == name)?;
        let n: usize = e.shape.iter().product();
        Some((&e.shape, &self.data[e.offset..e.offset + n]))
    }

    pub fn manifest_json(&self) -> String {
        serde_json::to_string(&self.manifest).expect("manifest serialises")
    }
}

pub struct Recorder {
    pub opts: TraceOptions,
    data: Vec<f32>,
    entries: Vec<Entry>,
}

impl Recorder {
    pub fn new(t: usize, opts: TraceOptions) -> Self {
        Self { opts, data: Vec::with_capacity(t * 60_000), entries: Vec::new() }
    }

    pub fn put(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
        debug_assert_eq!(shape.iter().product::<usize>(), vals.len(), "{name}");
        self.entries.push(Entry { name: name.to_string(), offset: self.data.len(), shape: shape.to_vec() });
        self.data.extend_from_slice(vals);
    }

    pub fn finish(self, ids: &[u32]) -> Trace {
        Trace { data: self.data, manifest: Manifest { ids: ids.to_vec(), entries: self.entries } }
    }
}
