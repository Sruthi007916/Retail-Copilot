import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def get_sales_trends():
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))

    # Make sure date is treated as a date
    sales["date"] = pd.to_datetime(sales["date"])

    # Get daily sales for each product
    daily_sales = (
        sales.groupby(["product_id", "date"])["quantity_sold"]
        .sum()
        .reset_index()
    )

    # Sort by product and date
    daily_sales = daily_sales.sort_values(
        ["product_id", "date"]
    )

    results = []

    for product_id, group in daily_sales.groupby("product_id"):

        group = group.sort_values("date")

        quantities = group["quantity_sold"].tolist()

        # Recent sales = latest day's sales
        recent_sales = quantities[-1]

        # Average sales across available days
        average_sales = group["quantity_sold"].mean()

        # Percentage change between latest day and average
        if average_sales == 0:
            percentage_change = 0
        else:
            percentage_change = (
                (recent_sales - average_sales)
                / average_sales
            ) * 100

        # Determine trend
        if percentage_change >= 25:
            trend_status = "SPIKE"
        elif percentage_change <= -25:
            trend_status = "DROP"
        else:
            trend_status = "STABLE"

        results.append({
            "product_id": product_id,
            "recent_sales": int(recent_sales),
            "average_sales": round(float(average_sales), 2),
            "percentage_change": round(float(percentage_change), 2),
            "trend_status": trend_status
        })

    result = pd.DataFrame(results)

    # Add product names
    result = result.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="left"
    )

    # Put product name first
    result = result[
        [
            "product_id",
            "product_name",
            "recent_sales",
            "average_sales",
            "percentage_change",
            "trend_status"
        ]
    ]

    # Sort biggest changes first
    result = result.sort_values(
        "percentage_change",
        ascending=False
    ).reset_index(drop=True)

    return result