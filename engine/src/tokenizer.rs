//! Byte-level BPE tokenizer, compatible with FLAME's `{"kind":"bpe","merges":[...]}` JSON.
//!
//! Id layout (same as FLAME): 0..=255 raw bytes, 256 `<|endoftext|>`, 257 `<|pad|>`,
//! 258.. merges in learned order. Pre-tokenisation and merge order follow FLAME exactly;
//! `parity` tests compare against FLAME on a corpus of strings.

use serde::Deserialize;
use std::collections::HashMap;

pub const EOS_ID: u32 = 256;
pub const PAD_ID: u32 = 257;
pub const FIRST_MERGE_ID: u32 = 258;
pub const EOS_TEXT: &str = "<|endoftext|>";

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Json {
    Byte,
    Bpe { merges: Vec<(u32, u32)> },
}

pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    token_bytes: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn from_json(s: &str) -> crate::Result<Self> {
        let j: Json = serde_json::from_str(s).map_err(|e| crate::Error(format!("tokenizer json: {e}")))?;
        let merges = match j {
            Json::Byte => Vec::new(),
            Json::Bpe { merges } => merges,
        };
        let ranks = merges.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        let mut token_bytes: Vec<Vec<u8>> = Vec::with_capacity(merges.len());
        for &(a, b) in &merges {
            let mut v = bytes_of(&token_bytes, a);
            v.extend(bytes_of(&token_bytes, b));
            token_bytes.push(v);
        }
        Ok(Self { merges, ranks, token_bytes })
    }

    pub fn vocab_size(&self) -> usize {
        FIRST_MERGE_ID as usize + self.merges.len()
    }

    /// Raw bytes of one token (EOS gives its marker text, PAD gives nothing).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match id {
            EOS_ID => EOS_TEXT.as_bytes().to_vec(),
            _ => bytes_of(&self.token_bytes, id),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        for w in pretokenize(text) {
            if let Some(ids) = cache.get(w) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.encode_word(w.as_bytes());
            out.extend_from_slice(&ids);
            cache.insert(w, ids);
        }
        out
    }

    /// Repeatedly applies the earliest-learned merge present in the word.
    fn encode_word(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
        loop {
            let mut best: Option<u32> = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|b| r < b) {
                        best = Some(r);
                    }
                }
            }
            let Some(rank) = best else { break };
            let pair = self.merges[rank as usize];
            let new_id = FIRST_MERGE_ID + rank;
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        ids
    }
}

fn bytes_of(table: &[Vec<u8>], id: u32) -> Vec<u8> {
    match id {
        0..=255 => vec![id as u8],
        EOS_ID | PAD_ID => vec![],
        _ => table.get((id - FIRST_MERGE_ID) as usize).cloned().unwrap_or_default(),
    }
}

/// Splits text into "words" before BPE so merges never cross word boundaries:
/// word runs, digit runs (≤ 3), punctuation runs, each optionally preceded by one space;
/// newlines plus the following indentation form one chunk.
pub fn pretokenize(text: &str) -> Vec<&str> {
    #[derive(PartialEq, Clone, Copy)]
    enum Class {
        Word,
        Digit,
        Punct,
        Space,
        Newline,
    }
    fn class(c: char) -> Class {
        if c == '\n' || c == '\r' {
            Class::Newline
        } else if c.is_whitespace() {
            Class::Space
        } else if c.is_ascii_digit() {
            Class::Digit
        } else if c.is_alphanumeric() || c == '_' {
            Class::Word
        } else {
            Class::Punct
        }
    }

    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut i = 0;
    while i < n {
        let start = i;
        let mut c = class(chars[i].1);
        if chars[i].1 == ' ' && i + 1 < n && !matches!(class(chars[i + 1].1), Class::Space | Class::Newline) {
            i += 1;
            c = class(chars[i].1);
        }
        match c {
            Class::Word => {
                while i < n && matches!(class(chars[i].1), Class::Word | Class::Digit) {
                    i += 1;
                }
            }
            Class::Digit => {
                let s = i;
                while i < n && class(chars[i].1) == Class::Digit && i - s < 3 {
                    i += 1;
                }
            }
            Class::Punct => {
                while i < n && class(chars[i].1) == Class::Punct {
                    i += 1;
                }
            }
            Class::Newline => {
                while i < n && class(chars[i].1) == Class::Newline {
                    i += 1;
                }
                while i < n && (chars[i].1 == ' ' || chars[i].1 == '\t') {
                    i += 1;
                }
            }
            Class::Space => {
                while i < n && class(chars[i].1) == Class::Space {
                    if chars[i].1 == ' '
                        && i + 1 < n
                        && !matches!(class(chars[i + 1].1), Class::Space | Class::Newline)
                        && i > start
                    {
                        break;
                    }
                    i += 1;
                }
            }
        }
        if i == start {
            i += 1;
        }
        out.push(&text[byte_at(start)..byte_at(i)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_is_lossless() {
        for t in ["Once upon a time,  there was   a café. 12345!", "fn main() {\n    x = 1;\n}\n", "  lead", ""] {
            assert_eq!(pretokenize(t).concat(), t);
        }
    }

    #[test]
    fn byte_tokenizer_roundtrip() {
        let tok = Tokenizer::from_json(r#"{"kind":"byte"}"#).unwrap();
        let s = "Grüße 🚀";
        assert_eq!(tok.decode(&tok.encode(s)), s);
    }
}
