"""KPI catalog MCP tool."""

from typing import Any, Dict, List

from ._safety import as_list

# Domain -> candidate KPI definitions.
_KPI_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "ecommerce": [
        {"name": "Total Revenue", "formula": "sum(revenue)", "grain": "daily", "business_use": "Top-line performance"},
        {"name": "Average Order Value", "formula": "sum(revenue) / count(order_id)", "grain": "daily", "business_use": "Basket size health"},
        {"name": "Conversion Rate", "formula": "orders / sessions", "grain": "daily", "business_use": "Funnel effectiveness"},
        {"name": "Repeat Purchase Rate", "formula": "repeat_customers / total_customers", "grain": "monthly", "business_use": "Loyalty signal"},
        {"name": "Cancellation Rate", "formula": "cancelled_orders / total_orders", "grain": "weekly", "business_use": "Fulfilment quality"},
    ],
    "saas": [
        {"name": "MRR", "formula": "sum(monthly_recurring_revenue)", "grain": "monthly", "business_use": "Recurring revenue"},
        {"name": "Churn Rate", "formula": "churned / active_start", "grain": "monthly", "business_use": "Retention"},
        {"name": "Activation Rate", "formula": "activated / signups", "grain": "weekly", "business_use": "Onboarding quality"},
        {"name": "DAU/MAU", "formula": "daily_active / monthly_active", "grain": "daily", "business_use": "Stickiness"},
    ],
    "events": [
        {"name": "Active Sessions", "formula": "count(distinct session_id)", "grain": "daily", "business_use": "Engagement volume"},
        {"name": "Error Rate", "formula": "error_events / total_events", "grain": "hourly", "business_use": "Reliability"},
        {"name": "Avg Session Duration", "formula": "avg(duration_ms)", "grain": "daily", "business_use": "Experience quality"},
        {"name": "Checkout Rate", "formula": "checkout_events / add_to_cart_events", "grain": "daily", "business_use": "Purchase intent"},
    ],
}

_GENERIC = [
    {"name": "Record Volume", "formula": "count(*)", "grain": "daily", "business_use": "Throughput"},
    {"name": "Distinct Entities", "formula": "count(distinct id)", "grain": "daily", "business_use": "Coverage"},
    {"name": "Completeness", "formula": "non_null / total", "grain": "daily", "business_use": "Data trust"},
]


def _match_by_columns(columns: List[str]) -> List[Dict[str, str]]:
    cols = {c.lower() for c in columns}
    extra = []
    if "revenue" in cols:
        extra.append({"name": "Revenue per Customer", "formula": "sum(revenue) / count(distinct customer_id)",
                      "grain": "monthly", "business_use": "Customer value"})
    if {"event_type", "session_id"} & cols:
        extra.append({"name": "Events per Session", "formula": "count(event_id) / count(distinct session_id)",
                      "grain": "daily", "business_use": "Depth of engagement"})
    return extra


def generate_kpi_catalog(domain: str, columns=None) -> Dict[str, Any]:
    """Build a KPI catalog from a domain and dataset columns (list/JSON/comma all fine)."""
    domain_key = (domain or "").strip().lower()
    kpis = list(_KPI_LIBRARY.get(domain_key, _GENERIC))
    kpis.extend(_match_by_columns(as_list(columns)))
    seen, unique = set(), []
    for k in kpis:
        if k["name"] not in seen:
            seen.add(k["name"])
            unique.append(k)
    return {"domain": domain_key or "generic", "kpi_count": len(unique), "kpis": unique}
