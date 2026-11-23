#!/usr/bin/env python3
"""Compute corpus WER for each human annotator against the dataset reference."""

import argparse
from pathlib import Path

from compute_inter_annotator_agreement import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_OUTPUT_DIR,
    clean_text,
    levenshtein,
    load_annotation,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "tables" / "human_wer_summary.csv"
    )
    args = parser.parse_args()

    paths = sorted(args.annotation_dir.glob("*.json"))
    if not paths:
        parser.error(f"No annotation JSON files found in {args.annotation_dir}")

    by_annotator: dict[str, dict[str, tuple[str, str]]] = {}
    for path in paths:
        annotation = load_annotation(path)
        items = by_annotator.setdefault(annotation.name, {})
        for uid, item in annotation.items.items():
            reference = item.get("reference")
            hypothesis = item["selected_text"]
            if not isinstance(reference, str) or not isinstance(hypothesis, str):
                parser.error(f"{path.name}, UID {uid}: expected reference and transcription strings")
            pair = clean_text(reference), clean_text(hypothesis)
            if items.setdefault(uid, pair) != pair:
                parser.error(f"{path.name}, UID {uid}: conflicting exports for {annotation.name}")

    rows = []
    print("Annotator\tUtterances\tEdits\tReference words\tWER")
    for name, items in sorted(by_annotator.items()):
        edits = ref_words = 0
        for reference, hypothesis in items.values():
            ref_tokens = reference.split()
            edits += levenshtein(ref_tokens, hypothesis.split())
            ref_words += len(ref_tokens)
        wer = edits / ref_words if ref_words else float("nan")
        rows.append(dict(annotator=name, utterances=len(items), edits=edits, ref_words=ref_words, wer=wer))
        print(f"{name}\t{len(items)}\t{edits}\t{ref_words}\t{wer:.2%}")

    write_csv(args.output, rows, ["annotator", "utterances", "edits", "ref_words", "wer"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
