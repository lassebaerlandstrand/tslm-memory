"""Scoring: OpenTSLM's ECG-QA-CoT answer rule, AUROC and a paired bootstrap."""

import re

import numpy as np
from sklearn.metrics import roc_auc_score


def extract_answer(text: str) -> str | None:
    # Same rule as OpenTSLM evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py, except that a
    # missing "Answer: " returns None instead of the whole text (which never matches anyway).
    if "Answer: " not in text:
        return None
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    return re.sub(r"\.$", "", answer).strip()


def normalise(answer: str) -> str:
    return answer.lower().strip().rstrip(".,!?;:")


def ecgqa_correct(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    return pred is not None and normalise(pred) == normalise(gold)


def auroc_within(y: np.ndarray, score: np.ndarray, strata: np.ndarray) -> float:
    """Mean AUROC over strata (finding x current state), so persistence scores 0.5."""
    return float(
        np.mean([roc_auc_score(y[strata == s], score[strata == s]) for s in np.unique(strata)])
    )


def paired_bootstrap(stat, groups: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of stat(idx) when resampling whole groups (patients) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq))
        vals.append(stat(np.concatenate([members[g] for g in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
