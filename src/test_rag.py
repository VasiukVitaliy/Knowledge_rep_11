import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag


def test_extract_and_check():
    ans = "Швидкість 50 км/год [pdr_12#str002, pdr_12#str003]. Граф [g1]. Вигадано [fake#1]."
    chk = rag.check_citations(ans, ["pdr_12#str002", "pdr_12#str003"], ["g1"])
    assert chk["chunks"] == ["pdr_12#str002", "pdr_12#str003"]
    assert chk["graph"] == ["g1"] and chk["invalid"] == ["fake#1"]


def test_strip_invalid_keeps_valid_ids():
    assert rag.strip_invalid("А [a#1, zz] Б [zz].", ["zz"]) == "А [a#1] Б ."


def test_no_citations():
    assert rag.extract_citations("Просто текст без цитат.") == []


def test_prompt_has_required_rules():
    p = rag.load_prompt()
    assert "[id]" in p and rag.NO_ANSWER in p and "суперечат" in p and "[g1]" in p


def test_abstain():
    assert rag.is_abstain(rag.NO_ANSWER) and not rag.is_abstain("50 км/год [a#1]")


def test_log_has_all_fields(tmp_path):
    import json
    f = tmp_path / "l.jsonl"
    rag.log_interaction({"query": "q", "sources": ["a"], "answer": "x", "tokens_in": 1,
                         "tokens_out": 2, "latency_ms": 3.0}, path=f)
    row = json.loads(f.read_text(encoding="utf-8"))
    assert {"query", "sources", "answer", "tokens_in", "tokens_out", "latency_ms"} <= row.keys()
