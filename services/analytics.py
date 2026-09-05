import pandas as pd
import os


# Get the data folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# Load CSV files
products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
inventory = pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"))


def get_inventory_status():

    # Calculate average daily sales for each product
    avg_sales = (
        sales.groupby("product_id")["quantity_sold"]
        .mean()
        .reset_index()
    )

    avg_sales.rename(
        columns={"quantity_sold": "average_daily_sales"},
        inplace=True
    )

    # Combine inventory with sales information
    result = inventory.merge(
        avg_sales,
        on="product_id",
        how="left"
    )

    # Add product names
    result = result.merge(
        products[["product_id", "product_name", "category"]],
        on="product_id",
        how="left"
    )

    # Calculate days of stock remaining
    result["days_remaining"] = (
        result["current_stock"] /
        result["average_daily_sales"]
    )

    # Determine status
    def get_status(days):
        if days < 2:
            return "CRITICAL"
        elif days < 5:
            return "WARNING"
        else:
            return "HEALTHY"

    result["status"] = result["days_remaining"].apply(get_status)

    return result


def get_stock_alerts():

    inventory_status = get_inventory_status()

    alerts = inventory_status[
        inventory_status["status"].isin(["CRITICAL", "WARNING"])
    ]

    return alerts


def get_overstocked_products():

    inventory_status = get_inventory_status()

    overstocked = inventory_status[
        inventory_status["days_remaining"] > 15
    ]

    return overstocked


def get_top_selling_products():

    top_products = (
        sales.groupby("product_id")["quantity_sold"]
        .sum()
        .reset_index()
        .sort_values("quantity_sold", ascending=False)
    )

    top_products = top_products.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="left"
    )

    return top_products