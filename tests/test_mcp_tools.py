"""Tests for the MCP tool implementations (pure functions in mcp_server/tools)."""
from tools import csv_profile_tools as cp
from tools import sql_tools as sq
from tools import data_quality_tools as dq
from tools import kpi_tools as kp
from tools import ml_tools as ml
from tools import report_tools as rp
from tools._safety import UnsafePathError, UnsafeSQLError, assert_read_only_sql
import pytest


def test_profile_csv_shape():
    out = cp.profile_csv("events_sample.csv")
    assert out["rows"] == 5000 and out["columns"] == 6


def test_data_dictionary_meanings():
    out = cp.create_data_dictionary("customers_sample.csv")
    names = {c["name"]: c["possible_meaning"] for c in out["columns"]}
    assert names["customer_id"] == "Unique identifier"


def test_validate_sql_blocks_drop():
    out = sq.validate_sql("DROP TABLE data")
    assert out["is_safe"] is False


def test_run_duckdb_query_groupby():
    out = sq.run_duckdb_query(
        "SELECT event_type, count(*) AS n FROM data GROUP BY event_type", "events_sample.csv")
    assert out["ok"] is True and out["row_count"] == 6


def test_run_duckdb_query_blocks_write():
    out = sq.run_duckdb_query("DELETE FROM data", "events_sample.csv")
    assert out["ok"] is False


def test_data_quality_finds_issues():
    out = dq.detect_data_quality_issues("transactions_sample.csv")
    types = {i["type"] for i in out["issues"]}
    assert "duplicate_rows" in types


def test_kpi_catalog_ecommerce():
    out = kp.generate_kpi_catalog("ecommerce", ["revenue", "order_id"])
    assert out["kpi_count"] >= 5


def test_ml_use_cases_churn():
    out = ml.recommend_ml_use_cases(["customer_id", "is_active", "revenue", "order_date"])
    assert any(c["use_case"] == "churn prediction" for c in out["ml_use_cases"])


def test_anomaly_zscore():
    out = ml.anomaly_detection_summary("events_sample.csv", "duration_ms", "zscore")
    assert out["ok"] is True and out["anomaly_count"] >= 0


def test_path_sandbox_blocks_escape():
    with pytest.raises(UnsafePathError):
        cp.profile_csv("../../etc/passwd")


def test_assert_read_only_blocks_multi_statement():
    with pytest.raises(UnsafeSQLError):
        assert_read_only_sql("SELECT 1; SELECT 2")


def test_report_has_all_sections():
    md = rp.generate_report_markdown({"Dataset Summary": "5000 rows"})
    for section in ("Dataset Summary", "Recommended KPIs", "Next Steps"):
        assert section in md


def test_kpi_catalog_accepts_string_columns():
    out = kp.generate_kpi_catalog("ecommerce", "revenue, order_id")
    assert out["kpi_count"] >= 5


def test_ml_use_cases_accepts_string_columns():
    out = ml.recommend_ml_use_cases("customer_id, is_active, revenue, order_date")
    assert any(c["use_case"] == "churn prediction" for c in out["ml_use_cases"])
