import json
import pytest
from function_tools import analyst_tools as an


def test_profile_dataframe_counts():
    out = json.loads(an.profile_dataframe("customers_sample.csv"))
    assert out["row_count"] == 500
    assert "customer_id" in out["columns"]


def test_profile_rejects_outside_path():
    out = json.loads(an.profile_dataframe("/etc/passwd"))
    assert "error" in out


def test_suggest_kpis_ecommerce():
    out = json.loads(an.suggest_kpi_metrics("ecommerce", ["revenue", "order_id"]))
    assert "Total revenue" in out["recommended_kpis"]


def test_validate_sql_blocks_destructive():
    out = json.loads(an.validate_sql_safety("DROP TABLE data"))
    assert out["is_safe"] is False
    assert "DROP" in out["blocked_keywords"]


def test_validate_sql_allows_select():
    out = json.loads(an.validate_sql_safety(
        "SELECT a FROM data WHERE order_date > '2025-01-01' LIMIT 5"))
    assert out["is_safe"] is True


def test_explain_query_result_negative():
    text = an.explain_query_result("monthly_revenue", "decreasing", -12.5)
    assert "decreased by 12.5%" in text


def test_dashboard_layout_has_sections():
    out = json.loads(an.generate_dashboard_layout("events", ["event_time", "status"]))
    assert len(out["sections"]) >= 3


def test_suggest_kpis_accepts_comma_string():
    # small local models often pass a plain string instead of a list
    out = json.loads(an.suggest_kpi_metrics("ecommerce", "revenue, order_id"))
    assert "Total revenue" in out["recommended_kpis"]


def test_suggest_kpis_accepts_json_string():
    out = json.loads(an.suggest_kpi_metrics("", '["event_type", "session_id"]'))
    assert "Active sessions" in out["recommended_kpis"]
