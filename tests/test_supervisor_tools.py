import json
from function_tools import supervisor_tools as sup


def test_classify_dashboard_goes_to_analyst():
    out = json.loads(sup.classify_user_request("Build a revenue dashboard with KPIs"))
    assert out["intent"] in ("dashboard", "analytics")
    assert "Analyst" in out["recommended_agent"]


def test_classify_mixed_request():
    out = json.loads(sup.classify_user_request("Build a dashboard and a churn prediction model"))
    assert out["intent"] == "mixed"


def test_work_plan_has_steps():
    out = json.loads(sup.create_agent_work_plan("profile csv and recommend ml use cases"))
    assert len(out["steps"]) >= 3
    assert out["steps"][-1].lower().startswith("combine")


def test_validate_structure_flags_missing():
    out = json.loads(sup.validate_final_response_structure("Direct Answer only here"))
    assert out["is_complete"] is False
    assert "Next Steps" in out["missing_sections"]


def test_estimate_context_usage_percent():
    out = json.loads(sup.estimate_context_usage("a" * 8192, context_window=8192))
    assert out["estimated_input_tokens"] == 2048
    assert 0 < out["usage_percent"] <= 100


def test_summarize_empty_history():
    assert "No prior" in sup.summarize_chat_history("")
