//! Minimal safetensors reader (F32 / F16 / BF16 → f32). The format is an 8-byte little-endian
//! header length, a JSON header mapping names to `{dtype, shape, data_offsets}`, then raw data.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load(bytes: &[u8]) -> crate::Result<HashMap<String, Tensor>> {
    if bytes.len() < 8 {
        crate::bail!("safetensors: file too small");
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if 8 + n > bytes.len() {
        crate::bail!("safetensors: bad header length");
    }
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| crate::Error(format!("safetensors header: {e}")))?;
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    for (name, v) in header {
        if name == "__metadata__" {
            continue;
        }
        let e: Entry = serde_json::from_value(v).map_err(|e| crate::Error(format!("safetensors {name}: {e}")))?;
        let [s, t] = e.data_offsets;
        if s > t || t > data.len() {
            crate::bail!("safetensors {name}: offsets out of range");
        }
        let raw = &data[s..t];
        let numel: usize = e.shape.iter().product();
        let vals: Vec<f32> = match e.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            d => crate::bail!("safetensors {name}: unsupported dtype {d}"),
        };
        if vals.len() != numel {
            crate::bail!("safetensors {name}: expected {numel} values, got {}", vals.len());
        }
        out.insert(name, Tensor { shape: e.shape, data: vals });
    }
    Ok(out)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_known_values() {
        assert_eq!(super::f16_to_f32(0x3c00), 1.0);
        assert_eq!(super::f16_to_f32(0xc000), -2.0);
        assert_eq!(super::f16_to_f32(0x0000), 0.0);
        assert!((super::f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
