import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instaquery.domain_db import build_quickcommerce_db  # noqa: E402
from instaquery.prompts import build_messages, extract_sql, normalize_sql  # noqa: E402
from instaquery.retrieval import ExampleRetriever, pick_fixed_shots  # noqa: E402
from instaquery.safety import check_sql  # noqa: E402
from instaquery.schema import serialize_schema  # noqa: E402
from instaquery.sql_eval import categorize_error, exec_match, execute_sql, result_eq, score  # noqa: E402


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_quickcommerce_db(tmp_path_factory.mktemp("qc") / "qc.sqlite", n_customers=400, n_orders=3000)


# ------------------------------------------------------------ prompts
def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert extract_sql("SQL: SELECT  a\n FROM t ; -- done") == "SELECT a FROM t"
    assert extract_sql("Here is the query:\nSELECT x FROM y WHERE z = 'a;b'") == "SELECT x FROM y WHERE z = 'a;b'"
    assert extract_sql("WITH c AS (SELECT 1) SELECT * FROM c") == "WITH c AS (SELECT 1) SELECT * FROM c"
    assert extract_sql("SELECT a FROM t\n\nThis query selects a.") == "SELECT a FROM t"


def test_normalize_keeps_string_literals():
    assert normalize_sql("SELECT  a FROM t WHERE n = 'two  spaces' ;") == "SELECT a FROM t WHERE n = 'two  spaces'"


def test_messages_shape():
    m = build_messages("CREATE TABLE t (a INT);", "How many?", [{"question": "q", "query": "SELECT 1"}], "SELECT count(*) FROM t")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert "Q: q" in m[1]["content"] and m[2]["content"] == "SELECT count(*) FROM t"


# ------------------------------------------------------------ schema
def test_schema_has_tables_values_and_fks(db):
    s = serialize_schema(str(db), 3)
    for t in ["dark_stores", "products", "orders", "order_items", "inventory"]:
        assert f"CREATE TABLE {t} (" in s
    assert "-- e.g." in s and "FOREIGN KEY (store_id) REFERENCES dark_stores(store_id)" in s
    assert "-- e.g." not in serialize_schema(str(db), 0)


# ------------------------------------------------------------ evaluation
def test_result_eq_semantics():
    assert result_eq([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], order_matters=False)
    assert not result_eq([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], order_matters=True)
    assert result_eq([(1, "a")], [("a", 1)], order_matters=False)          # column permutation
    assert not result_eq([(1,), (1,)], [(1,)], order_matters=False)        # bag semantics
    assert result_eq([], [], order_matters=False)


def test_exec_match_and_errors(db):
    gold = "SELECT COUNT(*) FROM dark_stores WHERE city = 'Bengaluru'"
    assert exec_match(db, "SELECT count(store_id) FROM dark_stores WHERE city='Bengaluru'", gold).correct
    r = exec_match(db, "SELECT COUNT(*) FROM stores", gold)
    assert not r.correct and categorize_error(r) == "unknown table"
    r = exec_match(db, "SELECT COUNT(*) FROM dark_stores WHERE city = 'Mumbai'", gold)
    assert r.pred_executable and categorize_error(r) == "wrong result (values)"


def test_timeout_stops_runaway_query(db):
    flag, msg = execute_sql(db, "SELECT COUNT(*) FROM order_items a, order_items b, order_items c", timeout_s=0.5)
    assert flag == "error" and msg == "timeout"


def test_readonly(db):
    flag, msg = execute_sql(db, "DELETE FROM orders")
    assert flag == "error"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3000


def test_domain_gold_queries_all_execute(db):
    qs = json.loads((ROOT / "data" / "quickcommerce_eval.json").read_text())
    assert len(qs) == 50
    for q in qs:
        flag, res = execute_sql(db, q["query"])
        assert flag == "ok", (q["question"], res)


def test_score_gold_is_perfect(db):
    qs = json.loads((ROOT / "data" / "quickcommerce_eval.json").read_text())
    exs = [dict(q, db_id="qc") for q in qs]
    sc = score(exs, [q["query"] for q in qs], lambda _: db, progress=False, keep_distinct=True)
    assert sc.ex == 1.0 and sc.exec_rate == 1.0


# ------------------------------------------------------------ safety
@pytest.mark.parametrize("sql,ok", [
    ("SELECT * FROM orders", True),
    ("WITH x AS (SELECT 1) SELECT * FROM x", True),
    ("DROP TABLE orders", False),
    ("DELETE FROM orders WHERE 1=1", False),
    ("SELECT 1; DROP TABLE orders", False),
    ("UPDATE orders SET status='x'", False),
    ("", False),
])
def test_guardrail(sql, ok):
    assert check_sql(sql)[0] is ok


# ------------------------------------------------------------ retrieval
def test_tfidf_retriever_excludes_same_db():
    pool = [{"question": "how many singers are there", "query": "SELECT count(*) FROM singer", "db_id": "a"},
            {"question": "how many singers do we have", "query": "SELECT count(*) FROM singer s", "db_id": "b"},
            {"question": "list all cars", "query": "SELECT * FROM cars", "db_id": "c"}]
    r = ExampleRetriever(pool, model_name=None)
    got = r.retrieve_many(["how many singers"], k=1, exclude_db=["a"])[0]
    assert got[0]["db_id"] == "b"


def test_pick_fixed_shots_deterministic():
    pool = [{"question": f"q{i}", "query": q, "db_id": "x"} for i, q in enumerate([
        "SELECT name FROM t", "SELECT count(*) FROM t WHERE a > 1",
        "SELECT a FROM t GROUP BY a HAVING count(*) > 2",
        "SELECT T1.a FROM t AS T1 JOIN s AS T2 ON T1.id = T2.id ORDER BY T1.b DESC LIMIT 1"])]
    shots = pick_fixed_shots(pool)
    assert [s["question"] for s in shots] == ["q1", "q2", "q3"]
