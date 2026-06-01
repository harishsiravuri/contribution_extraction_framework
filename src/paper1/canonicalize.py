"""Canonicalize extracted entity surface forms to standard canonical names.

The Phase 4 / NLP-TDMS triple F1 was near-zero because the LLM emits
"avg accuracy" / "Avg Accuracy" / "Average Accuracy" while the gold says
"accuracy", and similar surface variation across method/task/dataset names.
This module gives every loader and metric the same canonical-form lookup,
so set-match F1 is comparing semantics rather than surface strings.

Pure Python; lookup tables + simple regex. No LLM calls.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_METRIC_ALIASES = {
    # F1 family
    "f1": "F1",
    "f1 score": "F1",
    "f1-score": "F1",
    "f-score": "F1",
    "f score": "F1",
    "macro f1": "F1",
    "micro f1": "F1",
    "weighted f1": "F1",
    # Exact match / EM
    "em": "EM",
    "exact match": "EM",
    "exact-match accuracy": "EM",
    # Accuracy
    "accuracy": "accuracy",
    "acc": "accuracy",
    "avg accuracy": "accuracy",
    "average accuracy": "accuracy",
    "mean accuracy": "accuracy",
    "test accuracy": "accuracy",
    "top-1 accuracy": "accuracy",
    "top1 accuracy": "accuracy",
    "top-5 accuracy": "top-5 accuracy",
    "top5 accuracy": "top-5 accuracy",
    "classification accuracy": "accuracy",
    # BLEU
    "bleu": "BLEU",
    "bleu-1": "BLEU",
    "bleu-2": "BLEU",
    "bleu-3": "BLEU",
    "bleu-4": "BLEU",
    "bleu1": "BLEU",
    "bleu2": "BLEU",
    "bleu3": "BLEU",
    "bleu4": "BLEU",
    "sacrebleu": "BLEU",
    # ROUGE
    "rouge-l": "ROUGE-L",
    "rouge l": "ROUGE-L",
    "rougel": "ROUGE-L",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    # AUC
    "auc": "AUC",
    "roc-auc": "AUC",
    "roc auc": "AUC",
    "auroc": "AUC",
    "area under the roc curve": "AUC",
    "area under roc curve": "AUC",
    # Object detection / segmentation
    "map": "mAP",
    "m ap": "mAP",
    "mean average precision": "mAP",
    "ap": "AP",
    "miou": "mIoU",
    "iou": "IoU",
    # NLP modeling
    "perplexity": "perplexity",
    "ppl": "perplexity",
    # MT/QA distance
    "wer": "WER",
    "cer": "CER",
    "meteor": "METEOR",
    "chrf": "chrF",
    "spearman": "Spearman",
    "pearson": "Pearson",
    # Smatch
    "smatch": "Smatch",
}


def canonicalize_metric_name(name: str | None) -> str | None:
    if not name:
        return None
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    if s in _METRIC_ALIASES:
        return _METRIC_ALIASES[s]
    # Try stripping trailing "score" / "value"
    for suffix in (" score", " value"):
        if s.endswith(suffix):
            base = s[: -len(suffix)].strip()
            if base in _METRIC_ALIASES:
                return _METRIC_ALIASES[base]
    return name.strip()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_TASK_PREFIXES = (
    "the task of ",
    "task of ",
    "the problem of ",
    "problem of ",
    "the challenge of ",
)

_TASK_ALIASES = {
    "qa": "question answering",
    "question-answering": "question answering",
    "nli": "natural language inference",
    "ner": "named entity recognition",
    "pos": "part-of-speech tagging",
    "pos tagging": "part-of-speech tagging",
    "part of speech tagging": "part-of-speech tagging",
    "mt": "machine translation",
    "asr": "automatic speech recognition",
    "speech recognition": "automatic speech recognition",
    "summarization": "summarization",
    "summarisation": "summarization",
    "image classification": "image classification",
    "object detection": "object detection",
    "semantic segmentation": "semantic segmentation",
    "instance segmentation": "instance segmentation",
    "amr parsing": "amr parsing",
    "constituency parsing": "constituency parsing",
    "dependency parsing": "dependency parsing",
    "sentiment classification": "sentiment analysis",
    "sentiment analysis": "sentiment analysis",
    "language modeling": "language modeling",
    "language modelling": "language modeling",
}


def canonicalize_task_name(name: str | None) -> str | None:
    if not name:
        return None
    s = name.strip().lower()
    for pre in _TASK_PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("_", " ")
    if s in _TASK_ALIASES:
        return _TASK_ALIASES[s]
    return s


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

_METHOD_TRAILING = (
    " model",
    " architecture",
    " approach",
    " method",
    " algorithm",
    " framework",
    " network",
    " system",
)

# Names we want to preserve as the canonical form regardless of input case.
_METHOD_CANONICAL_KEYS = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "albert": "ALBERT",
    "gpt": "GPT",
    "gpt-2": "GPT-2",
    "gpt2": "GPT-2",
    "gpt-3": "GPT-3",
    "gpt3": "GPT-3",
    "gpt-4": "GPT-4",
    "gpt4": "GPT-4",
    "t5": "T5",
    "elmo": "ELMo",
    "transformer": "Transformer",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "gru": "GRU",
    "cnn": "CNN",
    "rnn": "RNN",
    "resnet": "ResNet",
    "resnet-50": "ResNet-50",
    "resnet50": "ResNet-50",
    "resnet-101": "ResNet-101",
    "vgg": "VGG",
    "vgg-16": "VGG-16",
    "vgg16": "VGG-16",
    "vit": "ViT",
    "yolo": "YOLO",
    "fasterrcnn": "Faster R-CNN",
    "faster r-cnn": "Faster R-CNN",
    "faster-rcnn": "Faster R-CNN",
    "mask r-cnn": "Mask R-CNN",
}


def canonicalize_method_name(name: str | None) -> str | None:
    if not name:
        return None
    s = name.strip()
    s_low = s.lower()
    s_low_clean = s_low
    # strip trailing words
    changed = True
    while changed:
        changed = False
        for tail in _METHOD_TRAILING:
            if s_low_clean.endswith(tail):
                s_low_clean = s_low_clean[: -len(tail)].strip()
                changed = True
    if s_low_clean in _METHOD_CANONICAL_KEYS:
        return _METHOD_CANONICAL_KEYS[s_low_clean]
    # Recover surface form proportional to cleaned string
    if s_low_clean != s_low:
        # Re-derive the surface form from the original by stripping the same suffix length
        n_strip = len(s) - len(s_low_clean)
        if n_strip > 0 and n_strip < len(s):
            s = s[: -n_strip].rstrip()
    return s


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_DATASET_VERSION_RE = re.compile(
    r"\s*[\(\[]?\s*(?:v|version)?\s*(\d+(?:\.\d+)*)\s*[\)\]]?$",
    re.IGNORECASE,
)

_DATASET_ALIASES = {
    "squad v1.1": ("SQuAD", "1.1"),
    "squad v2": ("SQuAD", "2.0"),
    "squad v2.0": ("SQuAD", "2.0"),
    "squad 2.0": ("SQuAD", "2.0"),
    "squad": ("SQuAD", None),
    "imagenet-1k": ("ImageNet", "1k"),
    "imagenet 1k": ("ImageNet", "1k"),
    "imagenet": ("ImageNet", None),
    "coco": ("COCO", None),
    "ms-coco": ("COCO", None),
    "ms coco": ("COCO", None),
    "cifar-10": ("CIFAR-10", None),
    "cifar10": ("CIFAR-10", None),
    "cifar-100": ("CIFAR-100", None),
    "cifar100": ("CIFAR-100", None),
    "glue": ("GLUE", None),
    "superglue": ("SuperGLUE", None),
    "wmt14": ("WMT", "14"),
    "wmt 14": ("WMT", "14"),
    "wmt'14": ("WMT", "14"),
    "wmt16": ("WMT", "16"),
    "wmt 16": ("WMT", "16"),
    "penn treebank": ("Penn Treebank", None),
    "ptb": ("Penn Treebank", None),
    "cityscapes": ("Cityscapes", None),
}


def canonicalize_dataset_name(name: str | None) -> tuple[str | None, str | None]:
    """Return (base_name, version_or_None)."""
    if not name:
        return None, None
    s = name.strip()
    low = s.lower()
    if low in _DATASET_ALIASES:
        return _DATASET_ALIASES[low]
    # Try to peel off a trailing version
    m = _DATASET_VERSION_RE.search(low)
    version: str | None = None
    if m:
        version = m.group(1)
        low_base = low[: m.start()].strip()
        s_base = s[: m.start()].strip()
        if low_base in _DATASET_ALIASES:
            base, _ = _DATASET_ALIASES[low_base]
            return base, version
        return s_base or s, version
    return s, None


def canonicalize_dataset_full(name: str | None) -> str | None:
    """Return a single canonical string ('SQuAD 1.1') for set-match purposes."""
    base, ver = canonicalize_dataset_name(name)
    if base is None:
        return None
    if ver:
        return f"{base} {ver}"
    return base
