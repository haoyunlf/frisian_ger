#!/usr/bin/env python3
"""Compute inter-annotator agreement for human transcription corrections.

The script accepts any number of annotator JSON exports with the same schema as
the existing `Annotator*.json` files. It computes:

  1. pairwise output WER between annotators,
  2. token-level Cohen's kappa for binary edit decisions (edit vs. no edit),
     following the legacy overlap report's baseline-position method, and
  3. exact output match.

By default it reads `../data/annotations/*.json` and writes results under
`../outputs/`.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_ANNOTATION_DIR = BASE_DIR / "data" / "annotations"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

EXTRA_SYMBOL_RE = re.compile(r"[<>\[\]{}|\\^`~]")
DE_EQUIV_TOKENS = {"'e", "’e", "‘e", "`e"}


@dataclass(frozen=True)
class AnnotationSet:
    name: str
    path: Path
    items: dict[str, dict]


def clean_text(text: object, normalize_de: bool = False) -> str:
    text = "" if text is None else str(text)
    text = EXTRA_SYMBOL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if normalize_de:
        text = " ".join("de" if tok in DE_EQUIV_TOKENS else tok for tok in text.split())
    return text


def levenshtein(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    previous = list(range(len(hyp_tokens) + 1))
    for i, ref_token in enumerate(ref_tokens, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def wer_stats(reference: str, hypothesis: str) -> tuple[int, int, float]:
    ref_tokens = clean_text(reference).split()
    hyp_tokens = clean_text(hypothesis).split()
    edits = levenshtein(ref_tokens, hyp_tokens)
    if ref_tokens:
        return edits, len(ref_tokens), edits / len(ref_tokens)
    return edits, 0, 0.0 if not hyp_tokens else math.inf


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> tuple[float, float, float]:
    if len(labels_a) != len(labels_b):
        raise ValueError("label vectors must have the same length")
    n = len(labels_a)
    if n == 0:
        return math.nan, math.nan, math.nan

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b))
    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else math.nan
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return kappa, observed, expected


def fmt(value: float, digits: int = 6) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def edited_positions(src: str, tgt: str) -> set[int]:
    """Return baseline token/insertion positions touched by src -> tgt."""
    src_tokens = clean_text(src).split()
    tgt_tokens = clean_text(tgt).split()
    positions: set[int] = set()
    matcher = difflib.SequenceMatcher(None, src_tokens, tgt_tokens, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            positions.add(i1)
        else:
            positions.update(range(i1, i2))
    return positions


def load_annotation(path: Path) -> AnnotationSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers") or data.get("subset") or []
    if not isinstance(answers, list):
        raise ValueError(f"{path} has no list-valued 'answers' or 'subset'")

    items: dict[str, dict] = {}
    for item in answers:
        uid = item.get("uid")
        if not uid:
            continue
        selected = item.get("selected_text", item.get("output", item.get("corrected", item.get("baseline_1best", ""))))
        items[uid] = {
            **item,
            "selected_text": selected,
            "selected_clean": clean_text(selected),
            "baseline_clean": clean_text(item.get("baseline_1best", "")),
        }

    name = str(data.get("user_id") or path.stem)
    return AnnotationSet(name=name, path=path, items=items)


def output_wer(rows: list[tuple[dict, dict]], left_is_reference: bool) -> tuple[int, int, float]:
    edits_total = 0
    ref_tokens_total = 0
    for item_a, item_b in rows:
        reference = item_a["selected_clean"] if left_is_reference else item_b["selected_clean"]
        hypothesis = item_b["selected_clean"] if left_is_reference else item_a["selected_clean"]
        edits, ref_tokens, _wer = wer_stats(reference, hypothesis)
        edits_total += edits
        ref_tokens_total += ref_tokens
    return edits_total, ref_tokens_total, edits_total / ref_tokens_total if ref_tokens_total else math.nan


def edit_label(item: dict) -> int:
    return int(item["selected_clean"] != item["baseline_clean"])


def token_edit_metrics(rows: list[tuple[str, dict, dict]]) -> tuple[dict, dict[str, dict]]:
    labels_a: list[int] = []
    labels_b: list[int] = []
    both = a_only = b_only = neither = 0
    a_positions_total = 0
    b_positions_total = 0
    item_stats: dict[str, dict] = {}

    for uid, item_a, item_b in rows:
        baseline = item_a.get("baseline_clean", item_b.get("baseline_clean", ""))
        pos_a = edited_positions(baseline, item_a["selected_clean"])
        pos_b = edited_positions(baseline, item_b["selected_clean"])
        a_positions_total += len(pos_a)
        b_positions_total += len(pos_b)

        universe = set(range(len(baseline.split()))) | pos_a | pos_b
        item_both = item_a_only = item_b_only = item_neither = 0
        for pos in sorted(universe):
            label_a = int(pos in pos_a)
            label_b = int(pos in pos_b)
            labels_a.append(label_a)
            labels_b.append(label_b)
            if label_a and label_b:
                both += 1
                item_both += 1
            elif label_a:
                a_only += 1
                item_a_only += 1
            elif label_b:
                b_only += 1
                item_b_only += 1
            else:
                neither += 1
                item_neither += 1

        item_stats[uid] = {
            "positions_a": " ".join(map(str, sorted(pos_a))),
            "positions_b": " ".join(map(str, sorted(pos_b))),
            "shared_positions": " ".join(map(str, sorted(pos_a & pos_b))),
            "union_positions": " ".join(map(str, sorted(pos_a | pos_b))),
            "token_positions": len(universe),
            "token_both_edited": item_both,
            "token_a_only_edited": item_a_only,
            "token_b_only_edited": item_b_only,
            "token_neither_edited": item_neither,
        }

    kappa, observed, expected = cohen_kappa(labels_a, labels_b)
    return (
        {
            "edit_decision_kappa": kappa,
            "edit_decision_observed": observed,
            "edit_decision_expected": expected,
            "token_positions": len(labels_a),
            "token_both_edited": both,
            "token_a_only_edited": a_only,
            "token_b_only_edited": b_only,
            "token_neither_edited": neither,
            "a_edited_positions": a_positions_total,
            "b_edited_positions": b_positions_total,
        },
        item_stats,
    )


def pairwise_metrics(a: AnnotationSet, b: AnnotationSet) -> tuple[dict, list[dict]]:
    overlap = sorted(set(a.items) & set(b.items))
    rows = [(uid, a.items[uid], b.items[uid]) for uid in overlap]
    token_summary, item_token_stats = token_edit_metrics(rows)
    output_rows = [(item_a, item_b) for _uid, item_a, item_b in rows]

    sentence_labels_a = [edit_label(item_a) for _uid, item_a, _item_b in rows]
    sentence_labels_b = [edit_label(item_b) for _uid, _item_a, item_b in rows]
    exact_matches = sum(item_a["selected_clean"] == item_b["selected_clean"] for _uid, item_a, item_b in rows)
    edits_ab, ref_tokens_ab, wer_ab = output_wer(output_rows, left_is_reference=True)
    edits_ba, ref_tokens_ba, wer_ba = output_wer(output_rows, left_is_reference=False)
    symmetric_wer = (wer_ab + wer_ba) / 2 if not (math.isnan(wer_ab) or math.isnan(wer_ba)) else math.nan

    summary = {
        "annotator_a": a.name,
        "annotator_b": b.name,
        "file_a": a.path.name,
        "file_b": b.path.name,
        "overlap_items": len(overlap),
        "exact_matches": exact_matches,
        "exact_match_rate": pct(exact_matches, len(overlap)),
        **token_summary,
        "a_sentence_edit_count": sum(sentence_labels_a),
        "b_sentence_edit_count": sum(sentence_labels_b),
        "a_sentence_no_edit_count": len(sentence_labels_a) - sum(sentence_labels_a),
        "b_sentence_no_edit_count": len(sentence_labels_b) - sum(sentence_labels_b),
        "wer_a_as_ref": wer_ab,
        "wer_a_as_ref_edits": edits_ab,
        "wer_a_as_ref_tokens": ref_tokens_ab,
        "wer_b_as_ref": wer_ba,
        "wer_b_as_ref_edits": edits_ba,
        "wer_b_as_ref_tokens": ref_tokens_ba,
        "symmetric_output_wer": symmetric_wer,
    }

    detail_rows = []
    for uid, item_a, item_b in rows:
        sentence_label_a = edit_label(item_a)
        sentence_label_b = edit_label(item_b)
        edits_ab_item, ref_tokens_ab_item, wer_ab_item = wer_stats(item_a["selected_clean"], item_b["selected_clean"])
        edits_ba_item, ref_tokens_ba_item, wer_ba_item = wer_stats(item_b["selected_clean"], item_a["selected_clean"])
        detail_rows.append(
            {
                "annotator_a": a.name,
                "annotator_b": b.name,
                "uid": uid,
                "category": item_a.get("category", item_b.get("category", "")),
                "sentence_edit_a": sentence_label_a,
                "sentence_edit_b": sentence_label_b,
                "sentence_edit_decision_agree": int(sentence_label_a == sentence_label_b),
                **item_token_stats[uid],
                "exact_output_match": int(item_a["selected_clean"] == item_b["selected_clean"]),
                "wer_a_as_ref": wer_ab_item,
                "wer_a_as_ref_edits": edits_ab_item,
                "wer_a_as_ref_tokens": ref_tokens_ab_item,
                "wer_b_as_ref": wer_ba_item,
                "wer_b_as_ref_edits": edits_ba_item,
                "wer_b_as_ref_tokens": ref_tokens_ba_item,
                "baseline": item_a.get("baseline_clean", item_b.get("baseline_clean", "")),
                "output_a": item_a["selected_clean"],
                "output_b": item_b["selected_clean"],
            }
        )

    return summary, detail_rows


def all_annotator_rows(annotations: list[AnnotationSet]) -> list[dict]:
    common_uids = sorted(set.intersection(*(set(ann.items) for ann in annotations))) if annotations else []
    rows = []
    for uid in common_uids:
        outputs = [ann.items[uid]["selected_clean"] for ann in annotations]
        edit_labels = [edit_label(ann.items[uid]) for ann in annotations]
        rows.append(
            {
                "uid": uid,
                "n_annotators": len(annotations),
                "all_exact_output_match": int(len(set(outputs)) == 1),
                "all_edit_decision_match": int(len(set(edit_labels)) == 1),
                "outputs": " ||| ".join(outputs),
                "edit_labels": " ".join(map(str, edit_labels)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: fmt(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_report(
    path: Path,
    annotations: list[AnnotationSet],
    pairwise: list[dict],
    all_rows: list[dict],
    output_prefix: str,
) -> None:
    pair_count = len(pairwise)
    mean_exact = sum(row["exact_match_rate"] for row in pairwise) / pair_count if pair_count else math.nan
    mean_kappa = sum(row["edit_decision_kappa"] for row in pairwise) / pair_count if pair_count else math.nan
    mean_symmetric_wer = sum(row["symmetric_output_wer"] for row in pairwise) / pair_count if pair_count else math.nan
    all_exact = sum(row["all_exact_output_match"] for row in all_rows)
    all_edit = sum(row["all_edit_decision_match"] for row in all_rows)

    lines = [
        "# Inter-Annotator Agreement",
        "",
        "## Inputs",
        "",
    ]
    for ann in annotations:
        lines.append(f"- `{ann.name}`: `{ann.path.name}` ({len(ann.items)} annotated items)")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Annotators | {len(annotations)} |",
            f"| Pairwise comparisons | {pair_count} |",
            f"| Mean pairwise symmetric output WER | **{fmt(mean_symmetric_wer)}** |",
            f"| Mean pairwise exact output match | **{fmt(mean_exact, 2)}%** |",
            f"| Mean pairwise token-level edit/no-edit Cohen's kappa | **{fmt(mean_kappa)}** |",
            f"| Common items across all annotators | {len(all_rows)} |",
            f"| All-annotator exact output match | {all_exact}/{len(all_rows)} = **{fmt(pct(all_exact, len(all_rows)), 2)}%** |",
            f"| All-annotator sentence-level edit/no-edit match | {all_edit}/{len(all_rows)} = **{fmt(pct(all_edit, len(all_rows)), 2)}%** |",
            "",
            "Normalization: text is lowercased and stripped of extra annotation symbols. Standalone `'e`/`’e`/`‘e`/`` `e`` tokens are **not** normalized to `de` by default.",
            "",
            "WER here is pairwise output WER between annotators. Because WER is denominator-dependent, the report includes both directions and a symmetric average.",
            "",
            "Token-level kappa follows the legacy overlap report: each baseline token or insertion position receives a binary label for whether that position was edited by each annotator.",
            "",
            "## Pairwise Results",
            "",
            "| Annotator A | Annotator B | N | Symmetric output WER | Exact match | Token kappa | Token positions | Both edited | A only | B only | Neither |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| `{row['annotator_a']}` | `{row['annotator_b']}` | {row['overlap_items']} | "
            f"{fmt(row['symmetric_output_wer'])} | "
            f"{row['exact_matches']}/{row['overlap_items']} ({fmt(row['exact_match_rate'], 2)}%) | "
            f"{fmt(row['edit_decision_kappa'])} | {row['token_positions']} | "
            f"{row['token_both_edited']} | {row['token_a_only_edited']} | "
            f"{row['token_b_only_edited']} | {row['token_neither_edited']} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Pairwise summary: `outputs/tables/{output_prefix}_pairwise_summary.csv`",
            f"- Per-item pairwise details: `outputs/tables/{output_prefix}_pairwise_details.csv`",
            f"- All-annotator common-item details: `outputs/tables/{output_prefix}_all_common_items.csv`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing annotator JSON files. Default: %(default)s",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="*",
        help="Explicit annotator JSON files. Overrides --annotations-dir when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reports/tables. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_paths = args.annotations or sorted(args.annotations_dir.glob("*.json"))
    if len(annotation_paths) < 2:
        raise SystemExit("Need at least two annotator JSON files.")

    annotations = [load_annotation(path) for path in annotation_paths]
    pairwise_summaries: list[dict] = []
    pairwise_details: list[dict] = []
    for ann_a, ann_b in itertools.combinations(annotations, 2):
        summary, details = pairwise_metrics(ann_a, ann_b)
        pairwise_summaries.append(summary)
        pairwise_details.extend(details)

    common_rows = all_annotator_rows(annotations)
    tables_dir = args.output_dir / "tables"
    reports_dir = args.output_dir / "reports"
    output_prefix = f"inter_annotator_{len(annotations)}annotators"
    summary_path = tables_dir / f"{output_prefix}_pairwise_summary.csv"
    details_path = tables_dir / f"{output_prefix}_pairwise_details.csv"
    common_path = tables_dir / f"{output_prefix}_all_common_items.csv"
    report_path = reports_dir / f"{output_prefix}_agreement.md"

    write_csv(
        summary_path,
        pairwise_summaries,
        [
            "annotator_a",
            "annotator_b",
            "file_a",
            "file_b",
            "overlap_items",
            "exact_matches",
            "exact_match_rate",
            "edit_decision_kappa",
            "edit_decision_observed",
            "edit_decision_expected",
            "token_positions",
            "token_both_edited",
            "token_a_only_edited",
            "token_b_only_edited",
            "token_neither_edited",
            "a_edited_positions",
            "b_edited_positions",
            "a_sentence_edit_count",
            "b_sentence_edit_count",
            "a_sentence_no_edit_count",
            "b_sentence_no_edit_count",
            "wer_a_as_ref",
            "wer_a_as_ref_edits",
            "wer_a_as_ref_tokens",
            "wer_b_as_ref",
            "wer_b_as_ref_edits",
            "wer_b_as_ref_tokens",
            "symmetric_output_wer",
        ],
    )
    write_csv(
        details_path,
        pairwise_details,
        [
            "annotator_a",
            "annotator_b",
            "uid",
            "category",
            "sentence_edit_a",
            "sentence_edit_b",
            "sentence_edit_decision_agree",
            "positions_a",
            "positions_b",
            "shared_positions",
            "union_positions",
            "token_positions",
            "token_both_edited",
            "token_a_only_edited",
            "token_b_only_edited",
            "token_neither_edited",
            "exact_output_match",
            "wer_a_as_ref",
            "wer_a_as_ref_edits",
            "wer_a_as_ref_tokens",
            "wer_b_as_ref",
            "wer_b_as_ref_edits",
            "wer_b_as_ref_tokens",
            "baseline",
            "output_a",
            "output_b",
        ],
    )
    write_csv(
        common_path,
        common_rows,
        ["uid", "n_annotators", "all_exact_output_match", "all_edit_decision_match", "outputs", "edit_labels"],
    )
    write_report(report_path, annotations, pairwise_summaries, common_rows, output_prefix)

    print(f"Loaded {len(annotations)} annotators")
    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {common_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
