import os
import re
import sys

MAIN_LABELS = ["Intersection", "RoadSegment", "SensorDevice", "TrafficLight", "Vehicle",
               "Driver", "TrafficAccident", "TrafficCongestion", "Route", "DataProcess"]

MAX_ENTITIES = 4
MAX_EDGES = 25

STREET_RE = re.compile(r"^(?:Вулиця|Проспект|Площа|Бульвар|Провулок|Шосе)\s+([^()]+?)\s*(?:\(|$)")
STREET_PREFIX = r"(?:вул\w*\.?|просп\w*\.?|пл\w*\.?|бульв\w*\.?)\s+"
# основи, що збігаються зі звичайними словами («зелена хвиля»): без слова «вулиця» не вважаємо вулицею
AMBIGUOUS_BARE = {"зелен", "науко"}

IMPACT_WORDS = ("вплив", "вплин", "залеж", "зачеп", "наслід", "постраж", "спричин", "призвед")
IMPACT_RELS = ("CAUSES|AFFECTS_ROUTE|RECOMMENDED_TO|SUPPLIES_DATA_TO|IS_BASIS_FOR|"
               "PRODUCES_ROUTE|RECOMMENDS_TO|ADJUSTS|NOTIFIES")

RETURN_EDGES = """
UNWIND range(0, size(relationships(path)) - 1) AS i
WITH relationships(path)[i] AS r, i
WITH r, min(i) AS d
RETURN startNode(r).uid AS a, labels(startNode(r)) AS la, type(r) AS t,
       endNode(r).uid AS b, labels(endNode(r)) AS lb,
       r.source AS src, r.confidence AS conf, d
ORDER BY d, a, t, b
LIMIT $limit
"""
NEIGHBORHOOD_CYPHER = ("MATCH (n) WHERE n.uid IN $uids\n"
                       "MATCH path = (n)-[*1..{n}]-()" + RETURN_EDGES)
IMPACT_CYPHER = ("MATCH (n) WHERE n.uid IN $uids\n"
                 "MATCH path = (n)-[:{rels}*1..{n}]->()" + RETURN_EDGES)


def main_label(labels):
    return next((l for l in MAIN_LABELS if l in labels), labels[0] if labels else "?")


def linearize(rows):
    lines = []
    for k, r in enumerate(rows, 1):
        tag = " [synthetic]" if str(r.get("src") or "").startswith("synthetic") else ""
        lines.append(f"[g{k}] {r['a']} ({main_label(r['la'])}) "
                     f"-{r['t']}-> {r['b']} ({main_label(r['lb'])}){tag}")
    return "\n".join(lines)


def pick_mode(question):
    q = question.casefold()
    return "impact" if any(w in q for w in IMPACT_WORDS) else "neighborhood"


class GraphContext:
    def __init__(self, uri=None, user=None, password=None, database=None, connect=True):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.db = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None
        if connect:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(user or os.getenv("NEO4J_USER", "neo4j"),
                      password or os.getenv("NEO4J_PASSWORD", "")))
            self._driver.verify_connectivity()
        self.keys = {}      # uid/code (casefold) -> [uid]
        self.streets = {}   # основа назви вулиці -> [uid]
        self.load_index()

    def _rows(self, cypher, **params):
        from neo4j import READ_ACCESS, Query
        with self._driver.session(database=self.db, default_access_mode=READ_ACCESS) as s:
            return [r.data() for r in s.run(Query(cypher, timeout=5), **params)]

    def load_index(self):
        """Один раз завантажує імена вузлів у пам'ять (граф невеликий)."""
        rows = self._rows("MATCH (n) WHERE n.uid IS NOT NULL "
                          "RETURN n.uid AS uid, n.code AS code, n.name_uk AS name_uk")
        for r in rows:
            for key in (r["uid"], r.get("code")):
                if key:
                    self.keys.setdefault(str(key).casefold(), []).append(r["uid"])
            m = STREET_RE.match(r.get("name_uk") or "")
            if m:
                name = m.group(1).strip().casefold()
                stem = name[:max(5, len(name) - 2)]          # Городоцька -> городоць (Городоцькій)
                self.streets.setdefault(stem, []).append(r["uid"])

    def link(self, question):
        """Entity linking: питання -> список uid (не більше MAX_ENTITIES)."""
        q, found = question.casefold(), []
        for key, uids in self.keys.items():
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", q):
                found += uids
        for stem, uids in self.streets.items():
            with_type = re.search(rf"(?<!\w){STREET_PREFIX}{re.escape(stem)}\w*", q)
            bare = re.search(rf"(?<!\w){re.escape(stem)}\w*", q) and stem not in AMBIGUOUS_BARE
            if with_type or bare:
                found += sorted(uids)
        return list(dict.fromkeys(found))[:MAX_ENTITIES]

    def subgraph(self, uids, mode="neighborhood", hops=1, depth=4, limit=MAX_EDGES):
        limit = min(int(limit), MAX_EDGES)
        if mode == "impact":
            n = int(depth)
            if not 1 <= n <= 5:
                raise ValueError("depth має бути від 1 до 5")
            cypher = IMPACT_CYPHER.format(rels=IMPACT_RELS, n=n)
        elif mode == "neighborhood":
            n = int(hops)
            if n not in (1, 2):
                raise ValueError("hops має бути 1 або 2")
            cypher = NEIGHBORHOOD_CYPHER.format(n=n)
        else:
            raise ValueError(f"невідомий режим: {mode!r}")
        return self._rows(cypher, uids=list(uids), limit=limit)

    def context(self, question, mode="auto", hops=1, depth=4, limit=MAX_EDGES):
        uids = self.link(question)
        mode = pick_mode(question) if mode == "auto" else mode
        if not uids:
            return {"entities": [], "edge_ids": [], "text": "", "mode": mode}
        rows = self.subgraph(uids, mode, hops, depth, limit)
        return {"entities": uids, "mode": mode,
                "edge_ids": [f"g{k}" for k in range(1, len(rows) + 1)],
                "text": linearize(rows)}


if __name__ == "__main__":
    g = GraphContext()
    if len(sys.argv) < 2:
        n = g._rows("MATCH (n) RETURN count(n) AS c")[0]["c"]
        r = g._rows("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        print(f"Підключення працює: вузлів {n}, зв'язків {r} (у ЛР 9 мало бути 227 і 486)")
    else:
        res = g.context(" ".join(sys.argv[1:]))
        print("режим:", res["mode"], "| сутності:", res["entities"])
        print(res["text"] or "(підграф порожній)")
