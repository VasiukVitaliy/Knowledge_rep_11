import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_rag as E


def test_numbers_normalization():
    assert E.numbers("25 781 аварія, 39,4% і 983") == {"25781", "39.4", "983"}


def test_context_recall():
    assert E.context_recall(["a b", "c d"], E.norm("тут A  B але не друге")) == 0.5
    assert E.context_recall([], "x") is None


def test_lexical_faithfulness_numbers_must_match():
    ctx = E.norm("У населених пунктах рух дозволяється із швидкістю не більше 50 км/год")
    assert E.lexical_faithfulness("Швидкість у населених пунктах не більше 50 км/год [a#1].", ctx) == 1.0
    assert E.lexical_faithfulness("Швидкість у населених пунктах не більше 90 км/год [a#1].", ctx) == 0.0
    assert E.lexical_faithfulness(E.NO_ANSWER, ctx) is None


def test_citation_support():
    CH = {"a#1": {"context": "Обгін заборонено на перехресті"}, "b#1": {"context": "зовсім інше про паркування"}}
    ans = "Обгін заборонено на перехресті [a#1]. Обгін заборонено на перехресті [b#1]. Вигадка [x#9]."
    total, good = E.citation_support(ans, [], {"a#1", "b#1"}, set(), CH)
    assert (total, good) == (3, 1)


def test_kappa():
    po, k = E.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert po == 1 and k == 1
    po, k = E.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])
    assert po == 0.5 and k == 0


def test_percentile():
    assert E.percentile([1, 2, 3, 4, 5], 50) == 3 and E.percentile([], 95) is None
