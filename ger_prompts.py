"""Shared GER system prompts and few-shot formatting."""

import json


SYSTEM_PROMPT = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with n-best ASR transcriptions (ordered by probability, highest first) and your task is to correct any errors in it.
If you come across errors in ASR transcription, make corrections that
closely match the original Frisian transcription acoustically or phonetically.
If you encounter grammatical errors, provide a corrected version
adhering to proper Frisian grammar.
Provide the most probable corrected transcription in string format.
Do not change the case, for example, lower case or upper case,
in the transcription.
Do not output any additional text that is not the corrected transcription.
Do not write any explanatory text that is not the corrected transcription."""

SYSTEM_PROMPT_SELECTION = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with an n-best list of ASR transcriptions, ordered from highest probability to lowest based on an acoustic model.
Your task is to select the single most probable corrected transcription from the provided candidates.

If you identify recognition errors, choose the candidate that most closely matches the intended Frisian utterance acoustically or phonetically.
If you encounter grammatical errors, prefer the candidate that adheres to proper Frisian grammar.
Do not generate a new sentence. Select only one sentence from the given n-best list.

Output ONLY the number of the best candidate (1, 2, 3, etc.).
Do not output the sentence itself, only the number.
Do not output any additional text or explanation."""

def load_few_shot_examples_from_validation(validation_file, n_examples, n_best, require_in_nbest=False):
    """
    从validation数据中动态加载few-shot例子

    Args:
        validation_file: validation数据文件路径
        n_examples: 需要的例子数量
        n_best: 每个例子中的候选数量

    Returns:
        加载的few-shot例子列表
    """
    try:
        with open(validation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        count = 0
        for item in data:
            if count >= n_examples:
                break

            # 确保有足够的候选和参考答案
            hyps = item["nhyps"]["hyps"]
            if len(hyps) < n_best:
                continue  # 跳过候选数量不足的样本

            reference = item.get("Reference", "")
            if not reference.strip():
                continue  # 跳过没有参考答案的样本

            scores = item["nhyps"].get("scores", [])

            # 找到 reference 在 nbest 中的位置（1-based），用于 select 模式
            ref_norm = reference.strip().lower()
            correct_index = None
            for idx, h in enumerate(hyps[:n_best], 1):
                if h.strip().lower() == ref_norm:
                    correct_index = idx
                    break

            example = {
                "nbest": hyps[:n_best],
                "scores": scores[:n_best] if scores else [],
                "corrected": reference,
                "correct_index": correct_index  # None 表示 reference 不在 nbest 中
            }

            # select 模式下跳过 reference 不在 nbest 中的样本
            if require_in_nbest and correct_index is None:
                continue

            examples.append(example)
            count += 1

        if len(examples) < n_examples:
            raise ValueError(f"Only found {len(examples)} valid examples in validation data, but {n_examples} examples were requested. "
                           f"Please check the validation file format or reduce the number of requested examples.")

        print(f"Successfully loaded {len(examples)} few-shot examples from validation data")
        return examples

    except Exception as e:
        raise RuntimeError(f"Error loading validation data: {e}")

def build_few_shot_prompt(base_system_prompt, num_examples=0, n_best=5, *, validation_file, show_scores=False, is_selection_mode=False):
    """
    构建包含 few-shot 示例的系统提示

    Args:
        base_system_prompt: 基础系统提示
        num_examples: 要包含的示例数量，0表示zero-shot
        n_best: 每个例子中的候选数量
        validation_file: 用于动态加载示例的验证集 JSON 路径
        show_scores: 是否在示例中显示 scores（默认 False，与测试数据保持一致）
        is_selection_mode: 是否为 select 模式，True 时 example 输出编号而非文本

    Returns:
        完整的系统提示
    """
    if num_examples <= 0:
        return base_system_prompt

    selected_examples = load_few_shot_examples_from_validation(
        validation_file, num_examples, n_best, require_in_nbest=is_selection_mode
    )

    # 构建包含示例的提示
    prompt_with_examples = base_system_prompt + "\n\nBelow are examples of how you should perform corrections.\n"

    for i, example in enumerate(selected_examples, 1):
        prompt_with_examples += f"\n### Example {i}\n"
        if show_scores and example.get("scores"):
            prompt_with_examples += "N-best candidates (scores are log-probabilities; higher means more likely):\n"
        else:
            prompt_with_examples += "N-best candidates:\n"

        for j, nbest_text in enumerate(example["nbest"], 1):
            if show_scores and "scores" in example and j <= len(example["scores"]):
                prompt_with_examples += f"{j}. {nbest_text} (score: {example['scores'][j-1]:.3f})\n"
            else:
                prompt_with_examples += f"{j}. {nbest_text}\n"

        if is_selection_mode:
            correct_index = example.get("correct_index")
            prompt_with_examples += f"\nSelect the number of the best candidate:\n{correct_index}\n"
        else:
            prompt_with_examples += f"\nCorrected text:\n{example['corrected']}\n"

    prompt_with_examples += "\n### End of examples.\n\nNow produce the corrected sentence for the next input."

    return prompt_with_examples

SYSTEM_PROMPT_NGRAM = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with n-best transcriptions generated by a trigram language model used during ASR decoding, and your task is to correct any errors.
If you come across errors in these transcriptions, make corrections that closely match the original Frisian acoustically or phonetically.
If you encounter grammatical errors, provide a corrected version adhering to proper Frisian grammar.
Provide the most probable corrected transcription in string format.
Do not change the case (lower case / upper case) in the transcription.
Do not output any additional text that is not the corrected transcription.
Do not write any explanation; output only the final corrected Frisian sentence.
"""

SYSTEM_PROMPT_DUAL = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with TWO sets of n-best ASR transcriptions from different sources:
1. Acoustic model n-best: Generated directly from the acoustic model without language model constraints
2. Trigram n-best: Generated using a trigram language model during ASR decoding

Your task is to analyze both sets of candidates and produce the most accurate corrected Frisian transcription by:
- Leveraging the acoustic accuracy from the acoustic model candidates
- Utilizing the linguistic knowledge from the trigram model candidates
- Making corrections that closely match the original Frisian acoustically or phonetically
- Ensuring proper Frisian grammar and syntax

Provide the most probable corrected transcription in string format.
Do not change the case (lower case / upper case) in the transcription.
Do not output any additional text that is not the corrected transcription.
Do not write any explanation; output only the final corrected Frisian sentence.
"""

SYSTEM_PROMPT_MULTI2 = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with TWO sets of n-best ASR transcriptions from different ASR systems:

1) MMS n-best: produced by an MMS ASR model
2) XLSR n-best: produced by an XLSR ASR model

Your task is to produce the single most accurate corrected Frisian transcription by:
- Comparing both n-best lists and using the strongest evidence across systems
- Preferring candidates that are acoustically plausible and phonetically consistent
- Fixing Frisian spelling and grammar when needed
- Resolving disagreements by choosing the most likely Frisian sentence overall

Important rules:
- Output ONLY the final corrected Frisian sentence.
- Do not output explanations.
- Do not change letter case (lowercase/uppercase) from the transcription style.
"""

SYSTEM_PROMPT_MULTI3 = """You are a Frisian language expert and an assistant that corrects ASR errors.
You will be presented with THREE sets of n-best ASR transcriptions:

1) MMS n-best: produced by an MMS ASR model
2) XLSR n-best: produced by an XLSR ASR model
3) NGRAM n-best: produced by decoding with an n-gram language model

Your task is to produce the single most accurate corrected Frisian transcription by:
- Using MMS/XLSR for acoustic plausibility
- Using NGRAM candidates as linguistic guidance when acoustics are ambiguous
- Fixing Frisian spelling and grammar when needed
- Avoiding over-correction: keep the sentence close to what is supported by the candidates

Important rules:
- Output ONLY the final corrected Frisian sentence.
- Do not output explanations.
- Do not change letter case (lowercase/uppercase) from the transcription style.
"""
