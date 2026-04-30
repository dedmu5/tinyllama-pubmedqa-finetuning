from llm_ft_eval.metrics import bleu1, rouge1_f1, rouge_l_f1, score_overlap


def test_overlap_scores_are_one_for_identical_text() -> None:
    text = "mitochondria generate cellular energy"
    scores = score_overlap(text, text)
    assert scores.bleu1 == 1.0
    assert scores.rouge1_f1 == 1.0
    assert scores.rouge_l_f1 == 1.0


def test_empty_prediction_scores_zero() -> None:
    assert bleu1("", "reference") == 0.0
    assert rouge1_f1("", "reference") == 0.0
    assert rouge_l_f1("", "reference") == 0.0
