"""Pre-compute umT5-xxl text embeddings for the Wan backbones of WAMModel.

Reads each LeRobot dataset's meta/tasks.jsonl, normalizes every task string the way WAMProcessor does
(formalize_language: lower + strip punctuation; ``_normalize`` below), encodes each unique normalized
string through the umT5 path the extractor uses (``fred.model.modules.wan_t5.load_wan_t5``, seq_len 512) and
saves one .pt payload read by ``TextEmbedCache`` (fred/model/modules/wan.py):

    {
      "embeddings": {normalized_str: Tensor [seq_len, 4096] bf16 on CPU},
      "masks":      {normalized_str: Tensor [seq_len] bool on CPU},
      ...meta
    }

The extractor looks entries up by the normalized prompt and feeds the raw umT5 output to the DiT (Wan's
text projection is inside the DiT, so nothing is projected here). Point the run at the file with
``model.text_embed_cache`` (launcher ``TEXT_EMBED_CACHE``); with ``model.text_embed_require_hit`` (the
default) training runs cache-only and never loads the encoder.

Hard requirement: 100% of tasks.jsonl strings encode + a re-load self-check passes.

``--append``: add prompts to an EXISTING cache in place (``--output-path``) without a rebuild, encoded with
the cache's own model_id / tokenizer_model_id / seq_len / dtype. Default prompt set: the empty prompt '' — caption dropout
(``data.text_dropout``) drops captions to '' and a cache-only run raises a miss without that key. A .bak
copy is written first unless ``--no-backup``.

Run on a GPU — the umT5-xxl encoder crawls on CPU.

    python scripts/precompute_text_embeds.py --data-root <lerobot dir> [...] --output-path cache.pt
    python scripts/precompute_text_embeds.py --append --output-path cache.pt [--prompts '' ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from fred.model.modules.wan_t5 import WAN_T5_MODEL_ID, WAN_T5_TOKENIZER_ID, load_wan_t5
from fred.model.wam.processor import normalize_language


def _normalize(s: str, formalize_language: bool) -> str:
    return normalize_language(s) if formalize_language else s


def _read_tasks(data_root: Path) -> list[str]:
    tasks_path = data_root / "meta" / "tasks.jsonl"
    assert tasks_path.exists(), f"no meta/tasks.jsonl under {data_root}"
    out = []
    with tasks_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("task") is not None:
                    out.append(row["task"])
    print(f"[i] read {len(out)} task strings from {tasks_path}")
    return out


def _encode(text_encoder, tokenizer, norms: list[str], device: torch.device, dtype: torch.dtype):
    """One umT5 batch through the extractor's path: [(emb [S, 4096], mask [S] bool)] on CPU, one per prompt."""
    ids, mask = tokenizer(norms, return_mask=True)   # [b,S], [b,S]
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        emb = text_encoder(ids, mask)                # [b,S,4096]
    return [(emb[k].to(dtype=dtype).cpu().contiguous(), mask[k].bool().cpu().contiguous()) for k in range(len(norms))]


def _warn_no_gpu(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        print(
            f"WARNING: umT5 encoder WITHOUT GPU (device={device}, cuda_available={torch.cuda.is_available()}). "
            f"The umT5-xxl forward crawls on CPU; allocate a GPU or fetch an already-built cache (deterministic). "
            f"Proceeding, but expect it to be very slow.",
            flush=True,
        )


def append(args) -> None:
    cache_path = Path(args.output_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict) and "embeddings" in payload and "masks" in payload, f"not a WAM umT5 text cache: {cache_path}"
    assert payload["embeddings"], f"{cache_path} has no entries to infer the dtype from"
    formalize = bool(payload["formalize_language"])
    seq_len = int(payload["seq_len"])
    emb_dim = int(payload["embedding_dim"])
    cache_dtype = next(iter(payload["embeddings"].values())).dtype   # the cache stays single-dtype
    assert not str(payload["model_id"]).endswith("-Diffusers"), (
        f"{cache_path} was built from {payload['model_id']}; rebuild it")

    to_add = []
    for raw in args.prompts + [t for dr in args.data_root for t in _read_tasks(Path(dr))]:
        norm = _normalize(raw, formalize_language=formalize)
        if norm in payload["embeddings"]:
            print(f"[i] '{norm}' already present — skip")
        elif norm not in to_add:
            to_add.append(norm)
    if not to_add:
        print("[ok] nothing to add (all prompts already cached)")
        return
    print(f"[i] cache={cache_path} entries={len(payload['embeddings'])} seq_len={seq_len} emb_dim={emb_dim}; "
          f"adding {len(to_add)} -> {to_add!r}")

    device = torch.device(args.device)
    _warn_no_gpu(device)
    text_encoder, tokenizer = load_wan_t5(payload["model_id"], payload.get("tokenizer_model_id", WAN_T5_TOKENIZER_ID),
                                          torch_dtype=cache_dtype, device=device,
                                          max_sequence_length=int(payload["max_sequence_length"]))
    for norm in to_add:
        (e, m), = _encode(text_encoder, tokenizer, [norm], device, cache_dtype)
        assert tuple(e.shape) == (seq_len, emb_dim), f"shape {tuple(e.shape)} != ({seq_len},{emb_dim})"
        payload["embeddings"][norm] = e
        payload["masks"][norm] = m
    ss = payload.get("source_stats")
    if isinstance(ss, dict):
        ss["n_appended_prompts"] = ss.get("n_appended_prompts", 0) + len(to_add)
        ss["includes_empty_prompt"] = "" in payload["embeddings"]

    if not args.no_backup:
        bak = cache_path.with_suffix(cache_path.suffix + ".bak")
        shutil.copy2(cache_path, bak)
        print(f"[i] backup -> {bak}")
    torch.save(payload, cache_path)

    chk = torch.load(cache_path, map_location="cpu", weights_only=False)
    for norm in to_add:
        assert norm in chk["embeddings"] and norm in chk["masks"], f"'{norm}' missing after reload"
    print(f"[ok] appended {len(to_add)} entrie(s); cache now {len(chk['embeddings'])} entries. "
          f"'' present={'' in chk['embeddings']}")


def build(args) -> None:
    data_roots = [Path(p) for p in args.data_root]
    data_root = data_roots[0] if len(data_roots) == 1 else data_roots   # single root: payload unchanged
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_tasks: list[str] = []
    for _dr in data_roots:
        raw_tasks.extend(_read_tasks(_dr))
    if len(data_roots) > 1:
        print(f"[i] {len(data_roots)} data roots → {len(raw_tasks)} raw task strings in total")
    # normalized -> first-seen raw (dedupe on the normalized key = the extractor's key)
    unique: dict[str, str] = {}
    for s in raw_tasks:
        norm = _normalize(s, formalize_language=args.formalize_language)
        if norm not in unique:
            unique[norm] = s
    if args.include_empty_prompt and "" not in unique:
        unique[""] = ""
        print("[i] --include-empty-prompt: added '' (empty prompt) for caption dropout")
    print(f"[i] {len(unique)} unique normalized strings (formalize_language={args.formalize_language})")

    device = torch.device(args.device)
    _warn_no_gpu(device)
    dtype = torch.bfloat16

    print(f"[i] loading umT5 encoder + tokenizer (model_id={args.model_id}, tokenizer={args.tokenizer_model_id})")
    text_encoder, tokenizer = load_wan_t5(args.model_id, args.tokenizer_model_id, torch_dtype=dtype, device=device,
                                          max_sequence_length=int(args.max_sequence_length))

    embeddings: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    # resume from .partial.pt if an earlier (preempted) run left one
    partial_path = output_path.with_suffix(".partial.pt")
    if partial_path.exists():
        _pp = torch.load(partial_path, map_location="cpu", weights_only=False)
        embeddings.update(_pp["embeddings"])
        masks.update(_pp["masks"])
        print(f"[i] RESUME: {len(embeddings)} entries from {partial_path}")
    items = [(n, r) for n, r in unique.items() if n not in embeddings]  # (norm, raw) not yet encoded
    print(f"[i] encoding {len(items)} strings via umT5 (seq_len={args.max_sequence_length}, dtype=bf16)")
    emb_dim = None
    seq_len = int(args.max_sequence_length)   # logical seq_len, recorded in the payload even when trimmed
    bs = max(1, int(args.batch_size))
    for i0 in range(0, len(items), bs):
        chunk = items[i0:i0 + bs]
        encoded = _encode(text_encoder, tokenizer, [n for n, _ in chunk], device, dtype)
        for (norm, raw), (e, m) in zip(chunk, encoded):
            if emb_dim is None:
                emb_dim = int(e.shape[1])
                print(f"      per-prompt embed shape ({int(e.shape[0])}, {emb_dim}), mask [{int(m.shape[0])}]")
            elif int(e.shape[1]) != emb_dim or int(e.shape[0]) != seq_len:
                raise RuntimeError(f"shape drift at {norm!r}: {tuple(e.shape)} != ({seq_len},{emb_dim})")
            if args.trim_to_mask:
                L = max(1, int(m.sum().item()))             # the umT5 mask is a contiguous prefix
                # e[:L] is a view of the full 512-row storage and torch.save would store all of it; clone() copies
                e = e[:L].clone()
                m = m[:L].clone()
            embeddings[norm] = e
            masks[norm] = m
        done_n = min(i0 + bs, len(items))
        if (i0 // bs) % 10 == 0 or done_n == len(items):
            print(f"      [{done_n}/{len(items)}] last_raw={chunk[-1][1]!r}", flush=True)
        if (i0 // bs) % 100 == 99 or done_n == len(items):
            _tmp = partial_path.with_suffix(".tmp")
            torch.save({"embeddings": embeddings, "masks": masks}, _tmp)
            os.replace(_tmp, partial_path)   # atomic, so a kill never leaves a torn partial file

    payload = {
        "version": 1,
        "model": "Wan2.2-TI2V-5B-umT5",
        "backbone_type": "wan22",
        "model_id": args.model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "formalize_language": bool(args.formalize_language),
        "max_sequence_length": int(args.max_sequence_length),
        "projected": False,  # Wan's text projection is inside the DiT; the raw umT5 output is cached
        "embeddings": embeddings,
        "masks": masks,
        "embedding_dim": int(emb_dim),
        "seq_len": int(seq_len),
        "dtype": "torch.bfloat16",
        "trimmed": bool(args.trim_to_mask),
        "source_stats": {"data_root": str(data_root), "n_raw_tasks": len(raw_tasks), "n_unique": len(unique)},
    }
    torch.save(payload, output_path)
    if partial_path.exists():
        partial_path.unlink()

    chk = torch.load(output_path, map_location="cpu", weights_only=False)
    assert set(chk["embeddings"].keys()) == set(embeddings.keys()), "embedding key set drift on reload"
    assert set(chk["masks"].keys()) == set(embeddings.keys()), "mask keys != embedding keys"
    first = next(iter(chk["embeddings"].values()))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"[ok] wrote {output_path} ({size_mb:.1f} MiB): {len(embeddings)} entries, "
        f"embed {tuple(first.shape)} {first.dtype}, masks present, self-check PASS"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", action="append", default=[],
                    help="LeRobot dataset root (repeatable; all roots' tasks.jsonl are merged)")
    ap.add_argument("--output-path", required=True, help="cache file to write (with --append: the cache to extend)")
    ap.add_argument("--model-id", default=WAN_T5_MODEL_ID, help="repo holding models_t5_umt5-xxl-enc-bf16.pth")
    ap.add_argument("--tokenizer-model-id", default=WAN_T5_TOKENIZER_ID, help="repo holding google/umt5-xxl/")
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1, help="umT5 encode batch (>=1) for large prompt sets (DROID: 123k)")
    ap.add_argument("--trim-to-mask", action=argparse.BooleanOptionalAction, default=False,
                    help="store each entry cut to its mask length L as [L,4096] (the full 512 rows are ~517GB for DROID); "
                         "the loader zero-pads back to seq_len with mask False (what the model does to pad positions anyway).")
    ap.add_argument("--formalize-language", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-empty-prompt", action=argparse.BooleanOptionalAction, default=False,
                    help="Also encode the empty prompt '' into the cache (needed for data.text_dropout > 0).")
    ap.add_argument("--append", action="store_true", help="add --prompts (and any --data-root tasks) to the existing --output-path cache")
    ap.add_argument("--prompts", nargs="*", default=[""], help="--append: raw prompt strings to add (default: the empty prompt '')")
    ap.add_argument("--no-backup", action="store_true", help="--append: skip the .bak copy")
    args = ap.parse_args()
    if args.append:
        append(args)
    else:
        assert args.data_root, "--data-root is required (at least one)"
        build(args)


if __name__ == "__main__":
    main()
