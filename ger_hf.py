#!/usr/bin/env python3
import os
import json
import argparse
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from evaluate import load as load_metric

from ger_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SELECTION,
    SYSTEM_PROMPT_NGRAM,
    SYSTEM_PROMPT_DUAL,
    SYSTEM_PROMPT_MULTI2,
    SYSTEM_PROMPT_MULTI3,
    build_few_shot_prompt,
)


def build_language_system_prompt(language_name):
    return f"""You are a {language_name} language expert and an assistant that corrects ASR errors.
You will be presented with n-best ASR transcriptions ordered by probability, highest first.
Your task is to recover the most likely original {language_name} sentence.
Use the n-best candidates as acoustic and phonetic evidence.
If you identify ASR errors, make corrections that closely match the original utterance.
If you identify grammatical errors, provide a corrected version in {language_name}.
Do not translate into another language.
Do not change the case of the transcription unless the correction requires it.
Output only one corrected transcription.
Do not output any explanation."""


def build_prompt(nbest_list, scores=None, is_selection_mode=False):
    prompt = "N-best candidates:\n"
    for i, text in enumerate(nbest_list):
        if scores is not None and i < len(scores):
            prompt += f"{i+1}. {text} (score: {scores[i]:.3f})\n"
        else:
            prompt += f"{i+1}. {text}\n"
    
    if is_selection_mode:
        prompt += "\nSelect the best candidate number:\n"
    else:
        prompt += "\nCorrected text:\n"
    return prompt


def build_dual_prompt(acoustic_nbest, acoustic_scores, trigram_nbest, is_selection_mode=False):
    # 和你 ger_gpt.py 的 dual 输入格式保持一致
    prompt = "Acoustic model n-best candidates:\n"
    for i, text in enumerate(acoustic_nbest):
        if acoustic_scores is not None and i < len(acoustic_scores):
            prompt += f"{i+1}. {text} (score: {acoustic_scores[i]:.3f})\n"
        else:
            prompt += f"{i+1}. {text}\n"

    prompt += "\nTrigram n-best candidates:\n"
    for i, text in enumerate(trigram_nbest):
        prompt += f"{i+1}. {text}\n"

    if is_selection_mode:
        prompt += "\nSelect the best candidate number:\n"
    else:
        prompt += "\nCorrected text:\n"
    return prompt


def generate_correction(
    model,
    tokenizer,
    user_prompt,
    max_new_tokens=200,
    model_name=None,
    enable_thinking=False,
    system_prompt=None,
    temperature=0.0,
    top_p=0.9,
    top_k=64,
):
    device = model.device
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    is_gemma = model_name and "gemma" in model_name.lower()
    is_qwen3 = model_name and "Qwen3" in model_name
    use_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt},
    ]

    if use_chat_template:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if is_qwen3:
            # Qwen3: 只在关闭thinking时显式传 False
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
        elif is_gemma:
            # Gemma4: 始终传 enable_thinking
            template_kwargs["enable_thinking"] = enable_thinking
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
    else:
        prompt_text = system_prompt.strip() + "\n\n" + user_prompt

    # Gemma4需要 text= 关键字参数
    if is_gemma:
        inputs = tokenizer(text=prompt_text, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    input_len = inputs["input_ids"].shape[-1]

    # 安全获取 eos_token_id：Gemma4Processor 的 eos_token_id 在内部的 tokenizer 里
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.tokenizer.eos_token_id
    else:
        eos_token_id = None

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": eos_token_id,
    }
    if temperature > 0:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
        if top_k is not None and top_k > 0:
            generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    # 统一处理 thinking 输出解析
    if is_gemma and enable_thinking:
        # Gemma4 thinking模式：skip_special_tokens=False + parse_response
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = tokenizer.parse_response(response)
        gen = parsed.get("text", response).strip()
    else:
        gen = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        # 去掉可能出现的 <think>...</think>（Qwen3等）
        lower_gen = gen.lower()
        if "<think>" in lower_gen and "</think>" in lower_gen:
            start = lower_gen.find("<think>")
            end = lower_gen.find("</think>", start)
            if end != -1:
                after = gen[end + len("</think>"):].strip()
                if after:
                    gen = after

    return gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ── 与 ger_gpt.py 完全相同的参数 ──────────────────────────────────────────
    parser.add_argument("--input", required=True, help="Primary N-best JSON file (acoustic model)")
    parser.add_argument("--dev_input", default=None,
                        help="DEV N-best JSON file used to build dynamic few-shot examples")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--input_trigram", help="Secondary N-best JSON file (trigram model)")
    parser.add_argument("--input_mms", default=None, help="Secondary N-best JSON file (mms)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--num_sample", type=int, default=None,
                        help="Process only first N samples (default: None, process all)")
    parser.add_argument("--prompt",
                        choices=["zero_shot", "select", "ngram", "dual", "multi2", "multi3"],
                        default="zero_shot",
                        help="Prompt type")
    parser.add_argument("--few_shot", type=int, default=0,
                        help="Number of few-shot examples (0=zero-shot); overrides --prompt if >0")
    parser.add_argument("--n_best", type=int, default=5,
                        help="Number of n-best candidates to use (default: 5)")
    parser.add_argument("--test_mode", action="store_true",
                        help="Print prompts and generations for debugging")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B",
                        help="HF model name or local fine-tuned model dir")
    parser.add_argument("--language_name", default=None,
                        help="Language name used in the system prompt for non-Frisian experiments")
    parser.add_argument("--lora_path", default=None,
                        help="Optional: local LoRA adapter dir (PEFT)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Model dtype: bf16/fp16/fp32. Default: auto (let transformers decide)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--use_scores", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.prompt == "dual" and not args.input_trigram:
        raise ValueError("--input_trigram is required when using dual prompt mode")

    # ── 选择 system prompt（与 ger_gpt.py 逻辑完全一致）─────────────────────
    base_system_prompt = build_language_system_prompt(args.language_name) if args.language_name else SYSTEM_PROMPT
    if args.few_shot > 0:
        if args.dev_input is None:
            parser.error("--few_shot requires --dev_input")
        validation_file = args.dev_input
        if not os.path.exists(validation_file):
            raise FileNotFoundError(f"Few-shot mode requires validation file: {validation_file}")
        selected_prompt = build_few_shot_prompt(
            base_system_prompt,
            args.few_shot,
            args.n_best,
            validation_file=validation_file,
            show_scores=args.use_scores,
            is_selection_mode=(args.prompt == "select"),
        )
        print(f"Using dynamic few-shot prompt with {args.few_shot} examples and {args.n_best} candidates from validation data")
    else:
        prompt_map = {
            "zero_shot": base_system_prompt,
            "ngram":     SYSTEM_PROMPT_NGRAM,
            "dual":      SYSTEM_PROMPT_DUAL,
            "multi2":    SYSTEM_PROMPT_MULTI2,
            "multi3":    SYSTEM_PROMPT_MULTI3,
            "select":    SYSTEM_PROMPT_SELECTION,
        }
        selected_prompt = prompt_map[args.prompt]
        print(f"Using {args.prompt} prompt")

    # ── dtype & 模型加载 ──────────────────────────────────────────────────────
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"  # 不传时让 transformers 自动选择

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_gemma = "gemma" in args.model_name.lower()
    print(f"Loading model: {args.model_name} (dtype={args.dtype or 'auto'})")

    # Gemma4用AutoProcessor，其他用AutoTokenizer
    if is_gemma:
        tokenizer = AutoProcessor.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()

    if args.lora_path:
        from peft import PeftModel
        print(f"Attaching LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()

    wer_metric = load_metric("wer")

    # ── 读取 mms 数据 ─────────────────────────────────────────────────────────
    data_mms = None
    if args.input_mms:
        with open(args.input_mms) as f:
            data_mms = json.load(f)

    # ── 读取 trigram 数据 ─────────────────────────────────────────────────────
    data_trigram = None
    if args.input_trigram:
        with open(args.input_trigram) as f:
            data_trigram = json.load(f)

    # ── 读取 acoustic 主数据 ──────────────────────────────────────────────────
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data_acoustic = json.load(f)

    data = data_acoustic

    # trigram / mms 对齐
    if data_trigram is not None:
        if len(data) != len(data_trigram):
            min_len = min(len(data), len(data_trigram))
            print(f"Warning: acoustic({len(data)}) != trigram({len(data_trigram)}), truncating to {min_len}")
            data = data[:min_len]
            data_trigram = data_trigram[:min_len]

    # ── 截断样本数 ────────────────────────────────────────────────────────────
    if args.num_sample is not None:
        data = data[:args.num_sample]
        if data_trigram is not None:
            data_trigram = data_trigram[:args.num_sample]
        if data_mms is not None:
            data_mms = data_mms[:args.num_sample]
        print(f"Processing first {len(data)} samples")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    results = []
    baseline_preds = []
    corrected_preds = []
    references = []

    for i, item in enumerate(tqdm(data, desc=f"{args.model_name} GER(HF)")):
        hyps = item["nhyps"]["hyps"]
        ref = item.get("Reference", "")
        scores = item["nhyps"].get("scores") if args.use_scores else None
        baseline = hyps[0]

        trigram_hyps = data_trigram[i]["nhyps"]["hyps"] if data_trigram else None
        acoustic_nbest = hyps[:args.n_best]
        acoustic_scores = scores[:args.n_best] if scores else None

        is_selection_mode = (args.prompt == "select")

        # 构造 sources（multi2/multi3 用）
        sources = None
        if data_mms or data_trigram:
            sources = [{"name": "XLSR", "nbest": acoustic_nbest,
                        "scores": acoustic_scores}]
            if data_mms:
                sources.append({"name": "MMS", "nbest": data_mms[i]["nhyps"]["hyps"][:args.n_best]})
            if data_trigram:
                sources.append({"name": "NGRAM", "nbest": trigram_hyps[:args.n_best]})

        # 构造 user prompt
        if args.prompt == "dual" and trigram_hyps:
            user_prompt = build_dual_prompt(acoustic_nbest, acoustic_scores,
                                            trigram_hyps[:args.n_best], is_selection_mode)
        else:
            user_prompt = build_prompt(acoustic_nbest, acoustic_scores, is_selection_mode)

        if args.test_mode:
            print(f"\n[TEST MODE] uid={item.get('Uid')}")
            print(f"System prompt:\n{selected_prompt}")
            print(f"User prompt:\n{user_prompt}")

        try:
            corrected = generate_correction(
                model, tokenizer,
                user_prompt,
                max_new_tokens=(32768 if args.enable_thinking else args.max_new_tokens),
                model_name=args.model_name,
                enable_thinking=args.enable_thinking,
                system_prompt=selected_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            if args.test_mode:
                print(f"Generation: {corrected}")

            # 选择模式：编号 → 句子
            if is_selection_mode and corrected:
                selected_index = int(corrected) - 1
                if 0 <= selected_index < len(acoustic_nbest):
                    corrected = acoustic_nbest[selected_index]
                else:
                    raise ValueError(f"Selection index {corrected} out of range (1-{len(acoustic_nbest)})")

            if corrected == "":
                raise ValueError("Empty correction from model")

            baseline_wer = wer_metric.compute(predictions=[baseline], references=[ref])
            corrected_wer = wer_metric.compute(predictions=[corrected], references=[ref])

            baseline_preds.append(baseline)
            corrected_preds.append(corrected)
            references.append(ref)

            result_item = {
                "uid": item.get("Uid"),
                "reference": ref,
                "baseline": baseline,
                "corrected": corrected,
                "baseline_wer": baseline_wer,
                "corrected_wer": corrected_wer,
            }
            if args.prompt == "dual" and trigram_hyps:
                result_item["acoustic_hyps"] = acoustic_nbest
                result_item["trigram_hyps"] = trigram_hyps[:args.n_best]
            else:
                result_item["nbest_used"] = acoustic_nbest
            if args.save_prompts:
                result_item["prompt"] = {
                    "system": selected_prompt,
                    "user": user_prompt,
                }
            results.append(result_item)

        except Exception as e:
            print(f"[ERROR] Model failed for uid={item.get('Uid')}: {e}")

    # ── WER 汇总 ─────────────────────────────────────────────────────────────
    wer_baseline  = wer_metric.compute(predictions=baseline_preds,  references=references)
    wer_corrected = wer_metric.compute(predictions=corrected_preds, references=references)
    improvement   = (wer_baseline - wer_corrected) / wer_baseline * 100 if wer_baseline > 0 else 0.0

    # ── 文件命名 ──────────────────────────────────────────────────────────────
    name_parts = []
    if args.num_sample is not None:
        name_parts.append(str(args.num_sample))
    model_tag = os.path.basename(args.model_name.rstrip("/"))
    if args.lora_path:
        model_tag += "_lora-" + os.path.basename(args.lora_path.rstrip("/"))
    name_parts.append(model_tag)
    if args.prompt == "dual":
        name_parts.append("dual")
    if args.few_shot > 0:
        name_parts.append(f"{args.few_shot}shot")
    if args.n_best != 5:
        name_parts.append(f"n{args.n_best}")
    if args.suffix:
        name_parts.append(args.suffix)

    os.makedirs(args.output, exist_ok=True)
    result_file = os.path.join(args.output, "results_" + "_".join(name_parts) + ".json")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    output_data = {
        "model": args.model_name,
        "lora_path": args.lora_path,
        "prompt_type": args.prompt,
        "few_shot": args.few_shot,
        "n_best": args.n_best,
        "dev_input": args.dev_input,
        "language_name": args.language_name,
        "dual_input": args.input_trigram is not None,
        "wer_baseline": wer_baseline,
        "wer_corrected": wer_corrected,
        "improvement_percent": improvement,
        "total_success": len(results),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "results": results,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Baseline WER:  {wer_baseline:.4f}")
    print(f"Corrected WER: {wer_corrected:.4f}")
    print(f"Improvement:   {improvement:.2f}%")
    print(f"Total: {len(results)}")
    print(f"Results saved to: {result_file}")
