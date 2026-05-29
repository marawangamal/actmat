"""Multilingual eval tasks for LightEval: Global-MMLU-Lite and MGSM.

Trimmed from the Polyglot-Teachers paper's task file (ljvmiranda921/polyglot-
teachers, scripts/lighteval_tasks.py). The original also defines M-RewardBench
tasks, but those depend on `MRewardBenchWeightedAccuracy`, which lives only in
the authors' lighteval fork (0.13.1.dev0). To run on a stock lighteval release
we drop M-RewardBench here; Global-MMLU-Lite and MGSM are unchanged from the
paper. Re-add M-RewardBench by installing their fork and restoring that section.
"""

import logging
import sys
from string import ascii_uppercase

import numpy as np
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric  # fmt: skip
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.normalizations import LogProbCharNorm  # fmt: skip
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


# ==== Global-MMLU-Lite ====

GLOBAL_MMLU_LITE = [
    LightevalTaskConfig(
        name=f"global_mmlu_lite:{standardize_tag(language.value)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [
                    line["option_a"],
                    line["option_b"],
                    line["option_c"],
                    line["option_d"],
                ],
                "gold_idx": ascii_uppercase.index(line["answer"]),
            },
            formulation=MCFFormulation(),
        ),
        hf_repo="CohereForAI/Global-MMLU-Lite",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            MCFFormulation(),
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
]


# ==== MGSM (Multilingual Grade School Math) ====


def mgsm_prompt_number_only(
    line, task_name: str = None, language: Language = Language.ENGLISH
):
    """
    Prompt that asks model to output ONLY the numerical answer.
    """
    # Instructions per language to output only the number
    instructions = {
        Language.ENGLISH: "Answer with only the number.",
        Language.GERMAN: "Antworte nur mit der Zahl.",
        Language.SPANISH: "Responde solo con el número.",
        Language.JAPANESE: "数字のみで答えてください。",
        Language.ARABIC: "أجب بالرقم فقط.",
        Language.INDONESIAN: "Jawab hanya dengan angka.",
    }

    inst = instructions.get(language, instructions[Language.ENGLISH])

    # Extract gold answer (just the number from answer_number field)
    gold = str(line["answer_number"])

    return Doc(
        task_name=task_name,
        query=f"{line['question']}\n\n{inst}\nAnswer:",
        choices=[gold],
        gold_index=0,
    )


# MGSM tasks with extractive number matching
# Note: The prompt_function uses 'answer_number' instead of 'answer' field
# This ensures few-shot examples show only the numerical answer, not the full CoT
MGSM = [
    LightevalTaskConfig(
        name=f"mgsm_custom:{subset}",
        prompt_function=lambda line, task_name=None, lang=language: mgsm_prompt_number_only(
            line, task_name, lang
        ),
        # Paper used ljvmiranda921/mgsm (a private fork, now 404). juletxara/mgsm
        # is the standard MGSM benchmark with identical schema (question,
        # answer_number) and the same fixed test set — equivalent data.
        hf_repo="juletxara/mgsm",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",  # Use train split for few-shot examples
        few_shots_select="sequential",
        generation_size=50,  # Short generation for just the number
        stop_sequence=["\n"],  # Stop at newline to just get the answer
        metrics=[
            SampleLevelMetric(
                metric_name="extractive_match",
                sample_level_fn=MultilingualExtractiveMatchMetric(
                    language=language,
                    # Extract numbers/expressions from both gold and prediction
                    gold_extraction_target=(
                        ExprExtractionConfig(try_extract_without_anchor=True),
                    ),
                    pred_extraction_target=(
                        ExprExtractionConfig(try_extract_without_anchor=True),
                    ),
                    aggregation_function=max,
                    fallback_mode="first_match",
                    extraction_mode="first_match",
                    precision=2,  # Allow small rounding differences
                ),
                category=SamplingMethod.GENERATIVE,
                corpus_level_fn=np.mean,
                higher_is_better=True,
            )
        ],
    )
    # Only German, Spanish, and Japanese are in both your list and MGSM
    # (Arabic and Indonesian are not in MGSM dataset)
    for subset, language in [
        ("de", Language.GERMAN),
        ("es", Language.SPANISH),
        ("ja", Language.JAPANESE),
    ]
]


TASKS_TABLE: list[LightevalTaskConfig] = GLOBAL_MMLU_LITE + MGSM
