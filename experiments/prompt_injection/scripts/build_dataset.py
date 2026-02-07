#!/usr/bin/env python3
"""Build prompt-injection datasets for benchmarking.

This script:
1) Loads trusted Hugging Face datasets and an optional local dataset.
2) Keeps benchmark and local corpora separate.
3) Normalizes to a common schema: prompt, label, source.
4) Removes exact duplicates.
5) Removes near-duplicates using SimHash bucketing.
6) Creates one stratified split per corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.model_selection import train_test_split

WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def normalize_text(text: str) -> str:
    text = text.replace("\u200b", " ")
    text = WS_RE.sub(" ", text.strip().lower())
    return text


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    try:
        return 1 if float(value) >= 0.5 else 0
    except (TypeError, ValueError):
        as_str = str(value).strip().lower()
        if as_str in {"1", "true", "yes", "attack", "adversarial"}:
            return 1
        return 0


def sample_with_stratification(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    if df["label"].nunique() < 2:
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        stratify=df["label"],
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def load_deepset(seed: int, max_rows: int | None) -> pd.DataFrame:
    ds = load_dataset("deepset/prompt-injections")
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for rec in ds[split]:
            rows.append(
                {
                    "prompt": rec["text"],
                    "label": normalize_label(rec["label"]),
                    "source": "deepset/prompt-injections",
                    "source_split": split,
                }
            )
    return sample_with_stratification(pd.DataFrame(rows), max_rows, seed)


def load_xtram(seed: int, max_rows: int | None) -> pd.DataFrame:
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for rec in ds[split]:
            rows.append(
                {
                    "prompt": rec["text"],
                    "label": normalize_label(rec["label"]),
                    "source": "xTRam1/safe-guard-prompt-injection",
                    "source_split": split,
                }
            )
    return sample_with_stratification(pd.DataFrame(rows), max_rows, seed)


def load_jbb(seed: int, max_rows: int | None) -> pd.DataFrame:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    rows: list[dict[str, Any]] = []
    for rec in ds["harmful"]:
        rows.append(
            {
                "prompt": rec["Goal"],
                "label": 1,
                "source": "JailbreakBench/JBB-Behaviors",
                "source_split": "harmful",
            }
        )
    for rec in ds["benign"]:
        rows.append(
            {
                "prompt": rec["Goal"],
                "label": 0,
                "source": "JailbreakBench/JBB-Behaviors",
                "source_split": "benign",
            }
        )
    return sample_with_stratification(pd.DataFrame(rows), max_rows, seed)


def load_local_paytm(local_path: Path, seed: int, max_rows: int | None) -> pd.DataFrame:
    ds = load_from_disk(str(local_path))
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for rec in ds[split]:
            prompt = rec.get("user") or ""
            if not prompt:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "label": normalize_label(rec.get("true_label")),
                    "source": "local/paytm_50k",
                    "source_split": split,
                }
            )
    return sample_with_stratification(pd.DataFrame(rows), max_rows, seed)


def text_shingles(text: str, n: int = 3) -> list[str]:
    clean = NON_ALNUM_RE.sub(" ", text)
    clean = WS_RE.sub(" ", clean).strip()
    if not clean:
        return [""]
    if len(clean) <= n:
        return [clean]
    return [clean[i : i + n] for i in range(0, len(clean) - n + 1)]


def simhash64(text: str) -> int:
    vector = [0] * 64
    for shingle in text_shingles(text, n=3):
        h = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, value in enumerate(vector):
        if value > 0:
            out |= 1 << bit
    return out


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def remove_near_duplicates(df: pd.DataFrame, hamming_threshold: int) -> tuple[pd.DataFrame, int, int]:
    kept_indices: list[int] = []
    # Bucket by high bits for efficient candidate filtering.
    buckets: dict[int, list[tuple[int, int, int]]] = {}
    removed = 0
    conflicting_labels = 0

    for idx, row in df.iterrows():
        sim = simhash64(row["norm_prompt"])
        label = int(row["label"])
        bucket = sim >> 48
        duplicate_found = False

        for kept_idx, kept_sim, kept_label in buckets.get(bucket, []):
            if hamming_distance(sim, kept_sim) <= hamming_threshold:
                duplicate_found = True
                removed += 1
                if kept_label != label:
                    conflicting_labels += 1
                break

        if not duplicate_found:
            kept_indices.append(idx)
            buckets.setdefault(bucket, []).append((idx, sim, label))

    return df.loc[kept_indices].reset_index(drop=True), removed, conflicting_labels


def describe_labels(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["label"].value_counts(dropna=False).to_dict()
    return {
        "total": int(len(df)),
        "label_0": int(counts.get(0, 0)),
        "label_1": int(counts.get(1, 0)),
        "attack_rate": float((counts.get(1, 0) / len(df)) if len(df) else 0.0),
    }


def load_env_local() -> None:
    """Load optional secrets/config from .env.local without hardcoding user paths.

    We check:
    - repo root: <repo>/.env.local
    - workspace root (one level above repo): <workspace>/.env.local
    """

    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    workspace_root = here.parents[4]
    for candidate in (repo_root / ".env.local", workspace_root / ".env.local"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break


def default_local_paytm_path() -> Path:
    # Keep a sensible auto-detected default for your workspace layout.
    here = Path(__file__).resolve()
    workspace_root = here.parents[4]  # .../vjl
    return workspace_root / "projects" / "proto" / "abhi-pi" / "data" / "read-only" / "paytm_50k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "data" / "datasets"),
    )
    parser.add_argument(
        "--local-paytm-path",
        type=Path,
        default=default_local_paytm_path(),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--near-dup-hamming-threshold", type=int, default=3)
    parser.add_argument("--balance-holdout", action="store_true", default=True)
    parser.add_argument("--no-balance-holdout", action="store_true")
    parser.add_argument("--local-holdout-from-test-only", action="store_true", default=True)
    parser.add_argument("--no-local-holdout-from-test-only", action="store_true")

    parser.add_argument("--max-deepset", type=int, default=0)
    parser.add_argument("--max-xtram", type=int, default=10000)
    parser.add_argument("--max-jbb", type=int, default=0)
    parser.add_argument("--max-local-paytm", type=int, default=10000)
    return parser.parse_args()


def to_limit(v: int) -> int | None:
    return None if v <= 0 else v


def balanced_binary_sample(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    label_counts = df["label"].value_counts().to_dict()
    c0 = int(label_counts.get(0, 0))
    c1 = int(label_counts.get(1, 0))
    per_class = min(c0, c1, max(target_n // 2, 1))
    if per_class <= 0:
        raise ValueError("Cannot create balanced sample: one class has zero rows.")

    neg = df[df["label"] == 0].sample(n=per_class, random_state=seed)
    pos = df[df["label"] == 1].sample(n=per_class, random_state=seed + 1)
    out = pd.concat([neg, pos], ignore_index=False)
    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if target_n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if df["label"].nunique() < 2:
        return df.sample(n=target_n, random_state=seed).reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=target_n,
        stratify=df["label"],
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def build_one_corpus(
    corpus_name: str,
    raw_df: pd.DataFrame,
    output_root: Path,
    holdout_ratio: float,
    seed: int,
    hamming_threshold: int,
    balance_holdout: bool,
    holdout_from_source_split: set[str] | None = None,
) -> dict[str, Any]:
    out_dir = output_root / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = raw_df.dropna(subset=["prompt"]).copy()
    merged["prompt"] = merged["prompt"].astype(str)
    merged["norm_prompt"] = merged["prompt"].map(normalize_text)
    merged = merged[merged["norm_prompt"] != ""].reset_index(drop=True)
    merged["label"] = merged["label"].map(normalize_label).astype(int)
    merged["_row_id"] = range(len(merged))

    pre_exact = len(merged)
    merged = merged.drop_duplicates(subset=["norm_prompt"], keep="first").reset_index(drop=True)
    exact_removed = pre_exact - len(merged)

    merged, near_removed, conflicting_labels = remove_near_duplicates(
        merged, hamming_threshold=hamming_threshold
    )

    target_holdout_n = max(2, int(round(len(merged) * holdout_ratio)))

    if holdout_from_source_split:
        holdout_pool = merged[merged["source_split"].isin(holdout_from_source_split)].copy()
        dev_pool = merged[~merged["source_split"].isin(holdout_from_source_split)].copy()
    else:
        holdout_pool = merged.copy()
        dev_pool = None

    if holdout_pool.empty:
        raise ValueError(f"{corpus_name}: holdout pool is empty")

    target_holdout_n = min(target_holdout_n, len(holdout_pool))
    if balance_holdout:
        holdout_df = balanced_binary_sample(holdout_pool, target_holdout_n, seed)
    else:
        holdout_df = stratified_sample(holdout_pool, target_holdout_n, seed)

    if dev_pool is not None:
        dev_df = dev_pool.reset_index(drop=True)
    else:
        holdout_ids = set(holdout_df["_row_id"].tolist())
        dev_df = merged[~merged["_row_id"].isin(holdout_ids)].reset_index(drop=True)

    merged_out = merged.drop(columns=["norm_prompt", "_row_id"])
    dev_out = dev_df.drop(columns=["norm_prompt", "_row_id"])
    holdout_out = holdout_df.drop(columns=["norm_prompt", "_row_id"])

    merged_out.to_parquet(out_dir / "all.parquet", index=False)
    dev_out.to_parquet(out_dir / "dev.parquet", index=False)
    holdout_out.to_parquet(out_dir / "holdout.parquet", index=False)

    merged_out.to_csv(out_dir / "all.csv", index=False)
    dev_out.to_csv(out_dir / "dev.csv", index=False)
    holdout_out.to_csv(out_dir / "holdout.csv", index=False)

    source_counts = (
        merged_out.groupby(["source", "label"]).size().reset_index(name="count").sort_values(["source", "label"])
    )
    source_counts.to_csv(out_dir / "source_label_counts.csv", index=False)

    manifest = {
        "corpus": corpus_name,
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "target_holdout_size": target_holdout_n,
        "near_dup_hamming_threshold": hamming_threshold,
        "balance_holdout": balance_holdout,
        "holdout_from_source_split": sorted(list(holdout_from_source_split)) if holdout_from_source_split else None,
        "source_input_counts": {
            str(source): int(count)
            for source, count in raw_df.groupby("source").size().to_dict().items()
        },
        "dedup": {
            "exact_removed": int(exact_removed),
            "near_removed": int(near_removed),
            "near_removed_conflicting_labels": int(conflicting_labels),
        },
        "all_stats": describe_labels(merged_out),
        "dev_stats": describe_labels(dev_out),
        "holdout_stats": describe_labels(holdout_out),
        "holdout_source_split_counts": {
            str(k): int(v) for k, v in holdout_out["source_split"].value_counts().to_dict().items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    load_env_local()
    args = parse_args()
    balance_holdout = bool(args.balance_holdout and not args.no_balance_holdout)
    local_holdout_from_test_only = bool(
        args.local_holdout_from_test_only and not args.no_local_holdout_from_test_only
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    deepset_df = load_deepset(args.seed, to_limit(args.max_deepset))
    xtram_df = load_xtram(args.seed, to_limit(args.max_xtram))
    jbb_df = load_jbb(args.seed, to_limit(args.max_jbb))
    benchmark_raw = pd.concat([deepset_df, xtram_df, jbb_df], ignore_index=True)

    summary: dict[str, Any] = {
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "near_dup_hamming_threshold": args.near_dup_hamming_threshold,
        "balance_holdout": balance_holdout,
        "local_holdout_from_test_only": local_holdout_from_test_only,
        "limits": {
            "max_deepset": to_limit(args.max_deepset),
            "max_xtram": to_limit(args.max_xtram),
            "max_jbb": to_limit(args.max_jbb),
            "max_local_paytm": to_limit(args.max_local_paytm),
        },
        "corpora": {},
    }

    summary["corpora"]["benchmark_pool"] = build_one_corpus(
        corpus_name="benchmark_pool",
        raw_df=benchmark_raw,
        output_root=args.output_dir,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        hamming_threshold=args.near_dup_hamming_threshold,
        balance_holdout=balance_holdout,
    )

    if args.local_paytm_path.exists():
        local_df = load_local_paytm(args.local_paytm_path, args.seed, to_limit(args.max_local_paytm))
        summary["corpora"]["local_paytm"] = build_one_corpus(
            corpus_name="local_paytm",
            raw_df=local_df,
            output_root=args.output_dir,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
            hamming_threshold=args.near_dup_hamming_threshold,
            balance_holdout=balance_holdout,
            holdout_from_source_split={"test"} if local_holdout_from_test_only else None,
        )
    else:
        summary["corpora"]["local_paytm"] = {"status": "missing", "path": str(args.local_paytm_path)}

    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote dataset artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
