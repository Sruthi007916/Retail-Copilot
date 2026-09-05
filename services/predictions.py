from services.analytics import get_inventory_status


def get_stockout_predictions():
    """
    Predict products that may run out of stock soon.

    The calculation is deterministic and uses only
    the available inventory and sales data.
    """

    inventory = get_inventory_status().copy()

    inventory["estimated_stockout_days"] = (
        inventory["current_stock"] /
        inventory["average_daily_sales"]
    )

    inventory["days_until_reorder"] = (
        (
            inventory["current_stock"] -
            inventory["reorder_level"]
        )
        /
        inventory["average_daily_sales"]
    )

    def get_risk(row):
        days = row["estimated_stockout_days"]

        if days < 2:
            return "HIGH"
        elif days < 5:
            return "MEDIUM"
        else:
            return "LOW"

    inventory["stockout_risk"] = inventory.apply(
        get_risk,
        axis=1
    )

    def get_action(row):
        if row["current_stock"] <= row["reorder_level"]:
            return "Reorder now"

        if row["stockout_risk"] == "HIGH":
            return "Reorder immediately"

        elif row["stockout_risk"] == "MEDIUM":
            return "Plan replenishment soon"

        else:
            return "Continue monitoring"

    inventory["recommended_action"] = inventory.apply(
        get_action,
        axis=1
    )

    predictions = inventory[
        [
            "product_id",
            "product_name",
            "current_stock",
            "reorder_level",
            "average_daily_sales",
            "estimated_stockout_days",
            "days_until_reorder",
            "stockout_risk",
            "recommended_action"
        ]
    ].copy()

    risk_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    predictions["risk_order"] = (
        predictions["stockout_risk"].map(risk_order)
    )

    predictions = predictions.sort_values(
        by=["risk_order", "estimated_stockout_days"]
    )

    predictions = predictions.drop(
        columns=["risk_order"]
    )

    return predictions