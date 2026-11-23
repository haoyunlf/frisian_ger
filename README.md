# Frisian GER

This repository contains the code and human annotations for our EACL 2027 paper.

**How Do LLMs Correct ASR Errors? A Systematic Analysis and Human Comparison in Low-Resource West Frisian**

Generative error correction (GER) uses a language model to correct Western Frisian automatic speech recognition (ASR) transcripts from ranked N-best hypotheses.

## Run GER

Use Python 3.12. Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python ger_hf.py --input examples/nbest.json --model_name Qwen/Qwen3-0.6B \
  --n_best 5
```

The first run downloads the model and WER metric. The command uses a small model and a synthetic example. It writes per-utterance corrections and corpus-level baseline and corrected word error rate (WER) to `results/`.

For your own data, replace `examples/nbest.json` with a JSON array of utterances in the same format. Each utterance needs a reference transcript and a nonempty `nhyps.hyps` list ordered best first. The first hypothesis is the baseline. Set `--n_best` to the number of candidates to use. For few-shot runs, add `--few_shot 5 --dev_input path/to/dev.json` and use a separate development file in the same format, with at least five examples that each have enough hypotheses.

## Human annotations

`data/annotations/` contains four pseudonymized sets of human corrections with 311, 100, 100, and 100 utterances. The four sets share 100 utterances. Join records by `uid`, the original audio filename. The `selected_text` field contains the human transcription, `reference` the dataset reference, and `nbest` the ASR candidates shown to annotators.

```bash
python scripts/compute_human_wer.py
python scripts/compute_inter_annotator_agreement.py
```
