#!/usr/bin/env python3
"""Package a FLAME run for the Hugging Face Hub (reads the FLAME checkout, never writes to it).

usage: tools/export_hf.py --run <FLAME>/runs/tinystories-5m --tokenizer <FLAME>/data/tokenizers/tinystories-4k.json \
                          --out hf_export --repo <user>/<model-repo> [--space <user>/<space>]
"""
import argparse, hashlib, json, math, shutil, tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def llama_mlp_hidden(d: int) -> int:
    return math.ceil(math.ceil(8 * d / 3) / 64) * 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", required=True, help="model repo id, e.g. user/flame-tinystories-5m")
    ap.add_argument("--space", default="", help="visualizer Space id, linked from the card")
    a = ap.parse_args()

    run_cfg = tomllib.loads((a.run / "config.toml").read_text())
    m, tr = run_cfg["model"], run_cfg["train"]
    hidden = m.get("mlp_hidden") or llama_mlp_hidden(m["d_model"])
    config = {
        "architecture": "flame-llama",
        "format_version": 1,
        "vocab_size": m["vocab_size"],
        "d_model": m["d_model"],
        "n_layers": m["n_layers"],
        "n_heads": m["n_heads"],
        "mlp_hidden": hidden,
        "max_seq_len": m["max_seq_len"],
        "rope_theta": m.get("rope_theta", 10000.0),
        "norm_eps": m.get("norm_eps", 1e-5),
        "tie_embeddings": m.get("tie_embeddings", True),
    }
    d, L, V = config["d_model"], config["n_layers"], config["vocab_size"]
    params = V * d * (1 if config["tie_embeddings"] else 2) + L * (4 * d * d + 3 * d * hidden + 2 * d) + d

    metrics = [json.loads(l) for l in (a.run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    evals = [x for x in metrics if "val_loss" in x]
    last_train = [x for x in metrics if "loss" in x][-1]
    best = min(evals, key=lambda x: x["val_loss"]) if evals else None

    a.out.mkdir(parents=True, exist_ok=True)
    weights = a.run / "best" / "model.safetensors"
    shutil.copyfile(weights, a.out / "model.safetensors")
    shutil.copyfile(a.tokenizer, a.out / "flame_tokenizer.json")
    shutil.copyfile(a.run / "metrics.jsonl", a.out / "metrics.jsonl")
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    card = (HERE.parent / "hf" / "MODEL_CARD.md").read_text()
    fills = {
        "REPO": a.repo,
        "SPACE_LINK": f"https://huggingface.co/spaces/{a.space}" if a.space else "(coming soon)",
        "PARAMS": f"{params / 1e6:.2f}M",
        "D": d, "L": L, "H": config["n_heads"], "HD": d // config["n_heads"], "F": hidden, "V": V,
        "CTX": config["max_seq_len"],
        "STEPS": last_train["step"],
        "TOKENS": f"{last_train['tokens'] / 1e6:.1f}M",
        "HOURS": f"{last_train.get('train_hours', 0):.1f}",
        "TRAIN_LOSS": f"{last_train['loss']:.3f}",
        "VAL_LOSS": f"{best['val_loss']:.3f}" if best else "n/a",
        "VAL_STEP": best["step"] if best else "n/a",
        "LR": tr["lr"], "MIN_LR": tr["min_lr"], "WARMUP": tr["warmup_steps"],
        "BATCH": tr["micro_batch_size"] * tr.get("grad_accum_steps", 1), "SEQ": tr["seq_len"],
        "SHA256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    for k, v in fills.items():
        card = card.replace("{{" + k + "}}", str(v))
    (a.out / "README.md").write_text(card)
    print(f"wrote {a.out}/ : params={fills['PARAMS']} val_loss={fills['VAL_LOSS']}")


if __name__ == "__main__":
    main()
