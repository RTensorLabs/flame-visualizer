//! The FLAME decoder, written out with plain loops so every intermediate is easy to record:
//!
//! ```text
//! x = tok_emb[ids]
//! N × [ x += Wo·Attn(RoPE(Wq·n), RoPE(Wk·n), Wv·n)  where n = RMSNorm(x)
//!       x += Wdown·(silu(Wgate·m) ⊙ Wup·m)          where m = RMSNorm(x) ]
//! logits = RMSNorm(x) · tok_embᵀ   (tied) or · lm_headᵀ
//! ```
//!
//! Linear weights are stored as in FLAME / PyTorch: (out, in), row-major.

use crate::config::Config;
use crate::safetensors::{self, Tensor};
use crate::trace::{Recorder, Trace, TraceOptions};

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    mlp_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

pub struct Model {
    cfg: Config,
    emb: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    /// RoPE tables, (max_seq_len, head_dim), "rotate-half" layout.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Model {
    pub fn from_safetensors(bytes: &[u8], cfg: Config) -> crate::Result<Self> {
        let mut t = safetensors::load(bytes)?;
        let (d, f, v) = (cfg.d_model, cfg.hidden(), cfg.vocab_size);
        let mut take = |name: &str, shape: &[usize]| -> crate::Result<Vec<f32>> {
            let Some(Tensor { shape: s, data }) = t.remove(name) else { crate::bail!("missing tensor {name}") };
            if s != shape {
                crate::bail!("tensor {name}: expected shape {shape:?}, found {s:?}");
            }
            Ok(data)
        };
        let emb = take("tok_emb.weight", &[v, d])?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(Layer {
                attn_norm: take(&p("attn_norm.weight"), &[d])?,
                wq: take(&p("attn.q_proj.weight"), &[d, d])?,
                wk: take(&p("attn.k_proj.weight"), &[d, d])?,
                wv: take(&p("attn.v_proj.weight"), &[d, d])?,
                wo: take(&p("attn.o_proj.weight"), &[d, d])?,
                mlp_norm: take(&p("mlp_norm.weight"), &[d])?,
                gate: take(&p("mlp.gate.weight"), &[f, d])?,
                up: take(&p("mlp.up.weight"), &[f, d])?,
                down: take(&p("mlp.down.weight"), &[d, f])?,
            });
        }
        let norm = take("norm.weight", &[d])?;
        let lm_head = if cfg.tie_embeddings { None } else { Some(take("lm_head.weight", &[v, d])?) };

        // Same construction as FLAME: inv_freq in f64 → f32, angle = pos·inv_freq in f32.
        let hd = cfg.head_dim();
        let half = hd / 2;
        let inv: Vec<f32> = (0..half).map(|i| cfg.rope_theta.powf(-((2 * i) as f64) / hd as f64) as f32).collect();
        let mut cos = vec![0f32; cfg.max_seq_len * hd];
        let mut sin = vec![0f32; cfg.max_seq_len * hd];
        for p in 0..cfg.max_seq_len {
            for j in 0..hd {
                let a = p as f32 * inv[j % half];
                cos[p * hd + j] = a.cos();
                sin[p * hd + j] = a.sin();
            }
        }
        Ok(Self { cfg, emb, layers, norm, lm_head, cos, sin })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Next-token logits for the last position only (fast path, nothing recorded).
    pub fn logits_last(&self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let x = self.run(ids, None)?;
        let d = self.cfg.d_model;
        let last = &x[(ids.len() - 1) * d..];
        let (h, _) = rmsnorm(last, &self.norm, self.cfg.norm_eps);
        Ok(self.head(&h))
    }

    /// Full forward pass recording every intermediate value.
    pub fn trace(&self, ids: &[u32], opts: &TraceOptions) -> crate::Result<Trace> {
        let mut rec = Recorder::new(ids.len(), opts.clone());
        let x = self.run(ids, Some(&mut rec))?;
        let (t, d) = (ids.len(), self.cfg.d_model);
        let mut fin = vec![0f32; t * d];
        for i in 0..t {
            fin[i * d..(i + 1) * d].copy_from_slice(&rmsnorm(&x[i * d..(i + 1) * d], &self.norm, self.cfg.norm_eps).0);
        }
        let logits = self.head(&fin[(t - 1) * d..]);
        rec.put("final_norm", &[t, d], &fin);
        rec.put("logits_last", &[self.cfg.vocab_size], &logits);
        Ok(rec.finish(ids))
    }

    fn head(&self, h: &[f32]) -> Vec<f32> {
        let w = self.lm_head.as_deref().unwrap_or(&self.emb);
        w.chunks_exact(self.cfg.d_model).map(|row| dot(h, row)).collect()
    }

    fn check(&self, ids: &[u32]) -> crate::Result<()> {
        if ids.is_empty() {
            crate::bail!("empty input");
        }
        if ids.len() > self.cfg.max_seq_len {
            crate::bail!("sequence length {} exceeds max_seq_len {}", ids.len(), self.cfg.max_seq_len);
        }
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= self.cfg.vocab_size) {
            crate::bail!("token id {bad} out of vocabulary");
        }
        Ok(())
    }

    /// Runs all blocks and returns the final residual stream (T, D).
    fn run(&self, ids: &[u32], mut rec: Option<&mut Recorder>) -> crate::Result<Vec<f32>> {
        self.check(ids)?;
        let c = &self.cfg;
        let (t, d, nh, hd, f) = (ids.len(), c.d_model, c.n_heads, c.head_dim(), c.hidden());
        let eps = c.norm_eps;
        let scale = (1.0 / (hd as f64).sqrt()) as f32;

        let mut x = Vec::with_capacity(t * d);
        for &id in ids {
            x.extend_from_slice(&self.emb[id as usize * d..(id as usize + 1) * d]);
        }
        if let Some(r) = rec.as_deref_mut() {
            r.put("embed", &[t, d], &x);
        }

        for (l, ly) in self.layers.iter().enumerate() {
            // ---- attention ----
            let (n, rms_in) = rmsnorm_rows(&x, &ly.attn_norm, d, eps);
            let q_lin = linear(&n, &ly.wq, d, d);
            let k_lin = linear(&n, &ly.wk, d, d);
            let v_lin = linear(&n, &ly.wv, d, d);
            let (q_pre, k_pre, v) = (split_heads(&q_lin, t, nh, hd), split_heads(&k_lin, t, nh, hd), split_heads(&v_lin, t, nh, hd));
            let q = self.rope(&q_pre, t, nh, hd);
            let k = self.rope(&k_pre, t, nh, hd);

            let mut scores = vec![f32::NAN; nh * t * t];
            let mut probs = vec![0f32; nh * t * t];
            let mut head_out = vec![0f32; nh * t * hd];
            for h in 0..nh {
                for i in 0..t {
                    let qi = &q[(h * t + i) * hd..(h * t + i + 1) * hd];
                    let row = &mut scores[(h * t + i) * t..(h * t + i + 1) * t];
                    for j in 0..=i {
                        row[j] = dot(qi, &k[(h * t + j) * hd..(h * t + j + 1) * hd]) * scale;
                    }
                    // softmax over the visible prefix; masked entries (score − 1e9) underflow to 0
                    let m = row[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let p = &mut probs[(h * t + i) * t..(h * t + i + 1) * t];
                    let mut sum = 0f32;
                    for j in 0..=i {
                        p[j] = (row[j] - m).exp();
                        sum += p[j];
                    }
                    for pj in p[..=i].iter_mut() {
                        *pj /= sum;
                    }
                    let o = &mut head_out[(h * t + i) * hd..(h * t + i + 1) * hd];
                    for j in 0..=i {
                        let vj = &v[(h * t + j) * hd..(h * t + j + 1) * hd];
                        let a = p[j];
                        for e in 0..hd {
                            o[e] += a * vj[e];
                        }
                    }
                }
            }
            let merged = merge_heads(&head_out, t, nh, hd);
            let attn_out = linear(&merged, &ly.wo, d, d);
            let x_mid: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

            // ---- MLP ----
            let (m, rms_mid) = rmsnorm_rows(&x_mid, &ly.mlp_norm, d, eps);
            let gate_pre = linear(&m, &ly.gate, d, f);
            let up = linear(&m, &ly.up, d, f);
            let gate: Vec<f32> = gate_pre.iter().map(|&g| silu(g)).collect();
            let hidden: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| a * b).collect();
            let mlp_out = linear(&hidden, &ly.down, f, d);
            let x_out: Vec<f32> = x_mid.iter().zip(&mlp_out).map(|(a, b)| a + b).collect();

            if let Some(r) = rec.as_deref_mut() {
                let p = |s: &str| format!("L{l}.{s}");
                r.put(&p("x_in"), &[t, d], &x);
                r.put(&p("rms_in"), &[t], &rms_in);
                r.put(&p("attn_norm"), &[t, d], &n);
                r.put(&p("q_pre"), &[nh, t, hd], &q_pre);
                r.put(&p("k_pre"), &[nh, t, hd], &k_pre);
                r.put(&p("q"), &[nh, t, hd], &q);
                r.put(&p("k"), &[nh, t, hd], &k);
                r.put(&p("v"), &[nh, t, hd], &v);
                r.put(&p("scores"), &[nh, t, t], &scores);
                r.put(&p("probs"), &[nh, t, t], &probs);
                r.put(&p("head_out"), &[nh, t, hd], &head_out);
                if r.opts.head_contrib {
                    r.put(&p("head_contrib"), &[nh, t, d], &head_contrib(&head_out, &ly.wo, t, nh, hd, d));
                }
                r.put(&p("attn_out"), &[t, d], &attn_out);
                r.put(&p("x_mid"), &[t, d], &x_mid);
                r.put(&p("rms_mid"), &[t], &rms_mid);
                r.put(&p("mlp_norm"), &[t, d], &m);
                r.put(&p("gate_pre"), &[t, f], &gate_pre);
                r.put(&p("gate"), &[t, f], &gate);
                r.put(&p("up"), &[t, f], &up);
                r.put(&p("hidden"), &[t, f], &hidden);
                r.put(&p("mlp_out"), &[t, d], &mlp_out);
                r.put(&p("x_out"), &[t, d], &x_out);
                if r.opts.lens_top_k > 0 {
                    let (ids_k, p_k) = self.lens(&x_out, t, r.opts.lens_top_k);
                    let k = r.opts.lens_top_k;
                    r.put(&p("lens_ids"), &[t, k], &ids_k);
                    r.put(&p("lens_p"), &[t, k], &p_k);
                }
            }
            x = x_out;
        }
        Ok(x)
    }

    /// Logit lens: decode each position's residual stream with the final norm + LM head.
    fn lens(&self, x: &[f32], t: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.d_model;
        let (mut ids, mut ps) = (Vec::with_capacity(t * k), Vec::with_capacity(t * k));
        for i in 0..t {
            let (h, _) = rmsnorm(&x[i * d..(i + 1) * d], &self.norm, self.cfg.norm_eps);
            let p = softmax(&self.head(&h));
            for (id, pv) in top_k(&p, k) {
                ids.push(id as f32);
                ps.push(pv);
            }
        }
        (ids, ps)
    }

    /// Rotate-half RoPE on (H, T, Hd): out = x·cos + cat(−x₂, x₁)·sin.
    fn rope(&self, x: &[f32], t: usize, nh: usize, hd: usize) -> Vec<f32> {
        let half = hd / 2;
        let mut out = vec![0f32; x.len()];
        for h in 0..nh {
            for p in 0..t {
                let base = (h * t + p) * hd;
                let (cs, sn) = (&self.cos[p * hd..(p + 1) * hd], &self.sin[p * hd..(p + 1) * hd]);
                for j in 0..hd {
                    let rot = if j < half { -x[base + j + half] } else { x[base + j - half] };
                    out[base + j] = x[base + j] * cs[j] + rot * sn[j];
                }
            }
        }
        out
    }
}

/// Head h's contribution to the attention output: its slice of the merged vector through Wo.
fn head_contrib(head_out: &[f32], wo: &[f32], t: usize, nh: usize, hd: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; nh * t * d];
    for h in 0..nh {
        for i in 0..t {
            let y = &head_out[(h * t + i) * hd..(h * t + i + 1) * hd];
            let o = &mut out[(h * t + i) * d..(h * t + i + 1) * d];
            for (r, ov) in o.iter_mut().enumerate() {
                *ov = dot(y, &wo[r * d + h * hd..r * d + (h + 1) * hd]);
            }
        }
    }
    out
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// y = x·Wᵀ for x (T, in) and W (out, in).
fn linear(x: &[f32], w: &[f32], din: usize, dout: usize) -> Vec<f32> {
    let t = x.len() / din;
    let mut y = vec![0f32; t * dout];
    for i in 0..t {
        let xi = &x[i * din..(i + 1) * din];
        for (o, row) in w.chunks_exact(din).enumerate() {
            y[i * dout + o] = dot(xi, row);
        }
    }
    y
}

/// RMSNorm of one vector; returns (normalised · g, rms divisor).
fn rmsnorm(x: &[f32], g: &[f32], eps: f64) -> (Vec<f32>, f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = (ms + eps as f32).sqrt();
    (x.iter().zip(g).map(|(v, w)| v / r * w).collect(), r)
}

fn rmsnorm_rows(x: &[f32], g: &[f32], d: usize, eps: f64) -> (Vec<f32>, Vec<f32>) {
    let mut out = Vec::with_capacity(x.len());
    let mut rs = Vec::with_capacity(x.len() / d);
    for row in x.chunks_exact(d) {
        let (n, r) = rmsnorm(row, g, eps);
        out.extend(n);
        rs.push(r);
    }
    (out, rs)
}

fn silu(x: f32) -> f32 {
    x / ((-x).exp() + 1.0)
}

/// (T, H·Hd) → (H, T, Hd)
fn split_heads(x: &[f32], t: usize, nh: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for i in 0..t {
        for h in 0..nh {
            out[(h * t + i) * hd..(h * t + i + 1) * hd].copy_from_slice(&x[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd]);
        }
    }
    out
}

/// (H, T, Hd) → (T, H·Hd)
fn merge_heads(x: &[f32], t: usize, nh: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for h in 0..nh {
        for i in 0..t {
            out[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd].copy_from_slice(&x[(h * t + i) * hd..(h * t + i + 1) * hd]);
        }
    }
    out
}

pub(crate) fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

pub(crate) fn top_k(p: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let k = k.min(p.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| p[b].total_cmp(&p[a]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    idx.into_iter().map(|i| (i, p[i])).collect()
}

