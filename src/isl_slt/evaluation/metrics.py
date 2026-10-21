"""Corpus metrics with explicit signatures, scales, normalization and missing-resource status."""

import importlib.metadata
import re
import unicodedata


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for row, word in enumerate(reference, 1):
        current = [row]
        for column, generated in enumerate(prediction, 1):
            current.append(min(previous[column] + 1, current[-1] + 1,
                               previous[column - 1] + (word != generated)))
        previous = current
    return previous[-1]


def score_predictions(predictions: list[str], references: list[list[str]], metrics: list[str],
                      bertscore_model: str | None = None, bertscore_revision: str | None = None) -> dict:
    if not predictions or len(predictions) != len(references) or any(not refs for refs in references):
        raise ValueError("Metrics require aligned, nonempty predictions and reference lists")
    predictions = [normalize(text) for text in predictions]
    references = [[normalize(text) for text in refs] for refs in references]
    columns = [[refs[index] if index < len(refs) else refs[0] for refs in references]
               for index in range(max(map(len, references)))]
    result = {"count": len(predictions), "scores": {}, "signatures": {}, "unavailable": {},
              "scale": "percent; TER/WER may exceed 100", "normalization": "NFC + whitespace collapse; case preserved",
              "aggregation": {"bleu": "cumulative corpus", "chrf": "corpus chrF++ word_order=2",
                              "rouge": "macro best-reference F1, Porter stemming",
                              "meteor": "macro NLTK best-reference, its documented lowercase/stem/WordNet policy",
                              "wer": "sum edit errors / sum chosen-reference words, 13a tokens, case preserved"}}
    for metric in metrics:
        try:
            if metric in ("bleu", "chrf", "ter"):
                from sacrebleu.metrics import BLEU, CHRF, TER
                engines = ({f"bleu{order}": BLEU(tokenize="13a", lowercase=False, max_ngram_order=order,
                                                 smooth_method="exp", effective_order=False) for order in range(1, 5)}
                           if metric == "bleu" else {"chrf++": CHRF(word_order=2)} if metric == "chrf" else {
                               "ter": TER(case_sensitive=True)})
                for name, engine in engines.items():
                    result["scores"][name] = engine.corpus_score(predictions, columns).score
                    result["signatures"][name] = str(engine.get_signature())
            elif metric == "rouge":
                from rouge_score.rouge_scorer import RougeScorer
                scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                totals = dict.fromkeys(("rouge1", "rouge2", "rougeL"), 0.0)
                for prediction, refs in zip(predictions, references):
                    candidates = [scorer.score(reference, prediction) for reference in refs]
                    for name in totals:
                        totals[name] += max(scores[name].fmeasure for scores in candidates)
                result["scores"].update({name: total * 100 / len(predictions) for name, total in totals.items()})
            elif metric == "meteor":
                import nltk
                from nltk.translate.meteor_score import meteor_score
                try:
                    nltk.data.find("corpora/wordnet")
                except LookupError:
                    nltk.data.find("corpora/wordnet.zip")
                from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
                tokenize = Tokenizer13a()
                scores = [meteor_score([tokenize(ref).split() for ref in refs], tokenize(pred).split())
                          for pred, refs in zip(predictions, references)]
                result["scores"]["meteor"] = sum(scores) * 100 / len(scores)
            elif metric == "wer":
                from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
                tokenize = Tokenizer13a()
                errors, words = 0, 0
                for prediction, refs in zip(predictions, references):
                    candidates = [(edit_distance(tokenize(ref).split(), tokenize(prediction).split()),
                                   len(tokenize(ref).split())) for ref in refs]
                    error, length = min(candidates, key=lambda pair: (pair[0] / max(pair[1], 1), pair[0], pair[1]))
                    errors, words = errors + error, words + length
                if words == 0:
                    raise ValueError("Corpus WER is undefined with zero reference words")
                result["scores"]["wer"] = errors / words * 100
            elif metric == "bertscore":
                from pathlib import Path
                if not bertscore_model or not bertscore_revision or not Path(bertscore_model).is_dir():
                    raise FileNotFoundError("BERTScore needs an explicit local model directory and recorded revision")
                from bert_score import BERTScorer
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(bertscore_model, local_files_only=True, trust_remote_code=False)
                scorer = BERTScorer(model_type=bertscore_model, num_layers=config.num_hidden_layers,
                                    device="cpu", lang="en", rescale_with_baseline=False)
                _, _, f1 = scorer.score(predictions, references)
                result["scores"]["bertscore_f1"] = f1.mean().item() * 100
                result["signatures"]["bertscore"] = f"{scorer.hash};revision={bertscore_revision}"
            else:
                raise ValueError(f"Unknown metric {metric}")
        except (ImportError, LookupError, OSError) as error:
            result["unavailable"][metric] = f"{type(error).__name__}: prepare the requested dependency/resource locally"
    for package in ("sacrebleu", "rouge-score", "nltk", "bert-score"):
        try:
            result["signatures"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["signatures"][package] = "not installed"
    return result
