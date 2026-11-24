import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from graph_context import MAX_ENTITIES, GraphContext, linearize, pick_mode

NODES = [
    {"uid": "INC_317", "code": "INC-317", "name_uk": "ДТП INC-317"},
    {"uid": "INC_31", "code": "INC-31", "name_uk": "ДТП INC-31"},
    {"uid": "SEC_ST_07", "code": "SEC-ST-07", "name_uk": "Вулиця Городоцька (SEC-ST-07)"},
    {"uid": "SEC_ST_21", "code": "SEC-ST-21", "name_uk": "Вулиця Городоцька (SEC-ST-21)"},
    {"uid": "SEC_ST_12", "code": "SEC-ST-12", "name_uk": "Вулиця Зелена (SEC-ST-12)"},
    {"uid": "COLLECT_1", "code": None, "name_uk": "Збір даних №1"},
]
EDGES = [{"a": "INC_317", "la": ["TrafficEvent", "TrafficAccident"], "t": "CAUSES",
          "b": "CONG_881", "lb": ["TrafficCongestion"], "src": "synthetic:seed42", "conf": 0.9, "d": 0}]


class Fake(GraphContext):
    def __init__(self):
        super().__init__(connect=False)

    def _rows(self, cypher, **params):
        return NODES if "RETURN n.uid" in cypher else EDGES


def test_link_by_uid_and_code_with_boundary():
    g = Fake()
    assert g.link("Що спричинила ДТП INC_317?") == ["INC_317"]
    assert g.link("Що спричинила ДТП INC-317?") == ["INC_317"]   # INC-31 не збігається
    assert g.link("Що залежить від COLLECT_1?") == ["COLLECT_1"]


def test_link_street_with_inflection():
    g = Fake()
    assert g.link("затори на Городоцькій") == ["SEC_ST_07", "SEC_ST_21"]


def test_green_wave_is_not_a_street():
    g = Fake()
    assert g.link("що таке зелена хвиля") == []
    assert g.link("затори на вулиці Зеленій") == ["SEC_ST_12"]


def test_entity_cap_and_empty_context():
    g = Fake()
    assert len(g.link("INC_317 INC-31 Городоцька COLLECT_1 вулиця Зелена")) <= MAX_ENTITIES
    assert g.context("яке сьогодні небо")["text"] == ""


def test_mode_auto():
    assert pick_mode("На що вплине ДТП INC_317?") == "impact"
    assert pick_mode("Які датчики на перехресті INT-LV-12?") == "neighborhood"
    assert Fake().context("Що залежить від COLLECT_1?")["mode"] == "impact"


def test_linearize_has_ids_and_synthetic_marker():
    t = linearize(EDGES)
    assert t.startswith("[g1] INC_317 (TrafficAccident) -CAUSES-> CONG_881 (TrafficCongestion)")
    assert "[synthetic]" in t


def test_limits_validated():
    with pytest.raises(ValueError):
        Fake().subgraph(["INC_317"], "neighborhood", hops=5)
    with pytest.raises(ValueError):
        Fake().subgraph(["INC_317"], "impact", depth=9)
    with pytest.raises(ValueError):
        Fake().subgraph(["INC_317"], "other")
