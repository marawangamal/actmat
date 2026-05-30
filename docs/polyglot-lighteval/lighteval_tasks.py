"""Defines multilingual evaluation tasks for LightEval, including Global-MMLU-Lite, MGSM, and M-RewardBench."""

import logging
import random
import sys
from string import ascii_uppercase
from typing import Any

import numpy as np
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric  # fmt: skip
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_corpus import MRewardBenchWeightedAccuracy
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.normalizations import LogProbCharNorm  # fmt: skip
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm
from lighteval.metrics.sample_preparator import LoglikelihoodPreparator  # fmt: skip
from lighteval.metrics.sample_preparator import LogprobCorpusMetricInput
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig
from lighteval.metrics.utils.metric_utils import CorpusLevelMetric, SampleLevelMetric
from lighteval.models.model_output import ModelResponse
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
        hf_repo="juletxara/mgsm",  # ljvmiranda921/mgsm fork was removed (404); juletxara/mgsm is parquet, same schema
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


# ==== M-RewardBench ====


# Custom preparator that includes source metadata
class MRewardBenchPreparator(LoglikelihoodPreparator):
    """Custom preparator for M-RewardBench that extracts and includes source metadata."""

    def prepare(
        self, doc: Doc, model_response: ModelResponse, **kwargs
    ) -> LogprobCorpusMetricInput:
        """Prepare loglikelihood data with source metadata.

        Args:
            doc: Document containing the source in specific metadata
            model_response: Model's response
            **kwargs: Additional arguments

        Returns:
            LogprobCorpusMetricInput with source attribute added
        """
        # Call parent prepare
        result = super().prepare(doc, model_response, **kwargs)

        # Extract source from doc.specific if available
        source = doc.specific.get("source", "Unknown") if doc.specific else "Unknown"

        # Add source as an attribute to the result
        # This is a bit hacky but works with the dataclass
        object.__setattr__(result, "source", source)

        return result


# Sample-level metric for parsing generative responses (A or B)
class GenerativeAccuracy(SampleLevelComputation):
    """Computes accuracy for generative M-RewardBench by parsing A/B choices."""

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """Parse 'A' or 'B' from the generated response and check if correct."""
        prediction = model_response.final_text[0].strip().upper()
        gold_index = (
            doc.gold_index[0] if isinstance(doc.gold_index, list) else doc.gold_index
        )

        # Extract the first occurrence of A or B
        predicted_choice = None
        for char in prediction:
            if char in ["A", "B"]:
                predicted_choice = char
                break

        # Map to index (A=0, B=1)
        pred_idx = (
            0 if predicted_choice == "A" else 1 if predicted_choice == "B" else -1
        )

        return 1.0 if pred_idx == gold_index else 0.0


# Create metrics at module level
generative_acc_metric = SampleLevelMetric(
    metric_name="acc",
    sample_level_fn=GenerativeAccuracy(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

# M-RewardBench weighted accuracy metrics (now imported from lighteval, avoiding pickling issues)
# Use custom preparator to include source metadata
# Average across all categories
mrewardbench_weighted_acc_metric = CorpusLevelMetric(
    metric_name="weighted_acc",
    sample_level_fn=MRewardBenchPreparator(is_single_token=True),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=MRewardBenchWeightedAccuracy(),
    higher_is_better=True,
)

# Per-category metrics
mrewardbench_chat_metric = CorpusLevelMetric(
    metric_name="weighted_acc_chat",
    sample_level_fn=MRewardBenchPreparator(is_single_token=True),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=MRewardBenchWeightedAccuracy(category="Chat"),
    higher_is_better=True,
)

mrewardbench_chat_hard_metric = CorpusLevelMetric(
    metric_name="weighted_acc_chat_hard",
    sample_level_fn=MRewardBenchPreparator(is_single_token=True),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=MRewardBenchWeightedAccuracy(category="Chat Hard"),
    higher_is_better=True,
)

mrewardbench_safety_metric = CorpusLevelMetric(
    metric_name="weighted_acc_safety",
    sample_level_fn=MRewardBenchPreparator(is_single_token=True),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=MRewardBenchWeightedAccuracy(category="Safety"),
    higher_is_better=True,
)

mrewardbench_reasoning_metric = CorpusLevelMetric(
    metric_name="weighted_acc_reasoning",
    sample_level_fn=MRewardBenchPreparator(is_single_token=True),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=MRewardBenchWeightedAccuracy(category="Reasoning"),
    higher_is_better=True,
)


def get_mrewardbench_prompt_function(language: Language):
    """Create a prompt function for M-RewardBench that includes source metadata."""

    # Get the base MCQ prompt function
    base_prompt_fn = get_mcq_prompt_function(
        language,
        lambda line: get_mrewardbench_eval_instances(line),
        formulation=MCFFormulation(),
    )

    def prompt_fn_with_source(line, task_name: str):
        """Wrapper that adds source to Doc.specific."""
        doc = base_prompt_fn(line, task_name)
        if doc is not None:
            # Add source to the specific metadata
            doc.specific = {"source": line.get("source", "Unknown")}
        return doc

    return prompt_fn_with_source


def get_mrewardbench_eval_instances(line: dict) -> dict[str, Any]:
    """Returns an eval instance of M-RewardBench containing the question, choices, and gold index.

    The chosen and rejected responses are shuffled randomly, and gold_idx tracks which position
    contains the chosen (correct) response.
    """
    # Reference: https://github.com/Cohere-Labs-Community/m-rewardbench/blob/main/scripts/generative.py#L218
    PROMPT_TEMPLATE = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "  # noqa
        "The question provided is in {src_lang}. "  # noqa
        "You should choose the assistant that follows the user's instructions and answers the user's question better. "  # noqa
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "  # noqa
        "Also, make sure that the assistant responses are in {tgt_lang}. "  # noqa
        "Begin your evaluation by comparing the two responses. "  # noqa
        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "  # noqa
        "Do not allow the length of the responses to influence your evaluation. "  # noqa
        "Do not favor certain names of the assistants. "  # noqa
        "Be as objective as possible. "  # noqa
        "After providing your explanation, output your final verdict by strictly following this format: "  # noqa
        '"A" if assistant A is better, "B" if assistant B is better.'  # noqa, removed tie option as , and \"[[C]]\ " for a tie
        "Don't put any quotation marks around your final verdict. "  # noqa
        "Here is the user question: {question} "  # noqa
    )

    chosen_response = line["chosen"]
    rejected_response = line["rejected"]

    # Shuffle chosen and rejected, keeping track of which is correct
    responses = [("chosen", chosen_response), ("rejected", rejected_response)]
    random.shuffle(responses)
    gold_idx = 0 if responses[0][0] == "chosen" else 1
    choices = [responses[0][1], responses[1][1]]

    question = PROMPT_TEMPLATE.format(
        src_lang=line["language"],
        tgt_lang=line["language"],
        question=line["prompt"],
    )

    return {
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
    }


iso2_to_extended = {
    "ar": "arb_Arab",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "id": "ind_Latn",
    "ja": "jpn_Jpan",
}


# M-RewardBench with MCF formulation (loglikelihood-based)
# Use this for local models that support logprob computation (HuggingFace, vLLM, etc.)
M_REWARDBENCH_MCF = [
    LightevalTaskConfig(
        name=f"mrewardbench_mcf:{standardize_tag(language.value)}",
        prompt_function=get_mrewardbench_prompt_function(language),
        hf_repo="CohereLabsCommunity/multilingual-reward-bench",
        hf_subset=iso2_to_extended.get(standardize_tag(language.value)),
        evaluation_splits=("test",),
        few_shots_split="test",
        metrics=[
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            mrewardbench_weighted_acc_metric,
            mrewardbench_chat_metric,
            mrewardbench_chat_hard_metric,
            mrewardbench_safety_metric,
            mrewardbench_reasoning_metric,
        ],
    )
    for language in [
        Language.ARABIC,
        Language.CZECH,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
]

# M-RewardBench with CF formulation (generative)
# Use this for API models via litellm (OpenAI, Anthropic, Cohere, etc.)
M_REWARDBENCH_CF = [
    LightevalTaskConfig(
        name=f"mrewardbench_cf:{standardize_tag(language.value)}",
        prompt_function=get_mrewardbench_prompt_function(language),
        hf_repo="CohereLabsCommunity/multilingual-reward-bench",
        hf_subset=iso2_to_extended.get(standardize_tag(language.value)),
        evaluation_splits=("test",),
        few_shots_split="test",
        metrics=[
            generative_acc_metric,
            mrewardbench_weighted_acc_metric,
            mrewardbench_chat_metric,
            mrewardbench_chat_hard_metric,
            mrewardbench_safety_metric,
            mrewardbench_reasoning_metric,
        ],
    )
    for language in [
        Language.ARABIC,
        Language.CZECH,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
]

TASKS_TABLE: list[LightevalTaskConfig] = (
    GLOBAL_MMLU_LITE + MGSM + M_REWARDBENCH_MCF + M_REWARDBENCH_CF
)
