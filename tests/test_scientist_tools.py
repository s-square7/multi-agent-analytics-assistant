import json
from function_tools import scientist_tools as sc


def test_problem_type_classification():
    out = json.loads(sc.recommend_ml_problem_type("predict whether a customer will churn"))
    assert out["problem_type"] == "classification"


def test_problem_type_forecasting():
    out = json.loads(sc.recommend_ml_problem_type("forecast next month demand"))
    assert out["problem_type"] == "forecasting"


def test_feature_engineering_events():
    out = json.loads(sc.suggest_feature_engineering("event and session activity logs"))
    assert any("session" in f for f in out["feature_ideas"])


def test_detect_ml_risks_time_split():
    out = json.loads(sc.detect_ml_data_risks("events_sample.csv"))
    assert out["risk_count"] >= 1
    assert any("time-based" in r for r in out["risks"])


def test_eval_metrics_classification():
    out = json.loads(sc.recommend_evaluation_metrics("classification"))
    assert "ROC-AUC" in out["metrics"]


def test_pipeline_plan_forecasting_uses_time_split():
    out = json.loads(sc.create_ml_pipeline_plan("forecasting"))
    assert len(out["pipeline"]) == 10
    assert any("time-based" in step for step in out["pipeline"])
