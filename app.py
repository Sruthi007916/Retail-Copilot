from flask import Flask, render_template, jsonify, request
import pandas as pd

from services.analytics import (
    get_inventory_status,
    get_stock_alerts,
    get_overstocked_products,
    get_top_selling_products
)

from services.predictions import get_stockout_predictions
from services.sales_trends import get_sales_trends

from services.retrieval_service import (
    build_knowledge_base,
    retrieve_relevant_context
)

from services.gemini_service import ask_gemini


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# BUILD LOCAL KNOWLEDGE BASE
# ============================================================

print("Building local knowledge base...")

try:
    knowledge_base = build_knowledge_base()
    print(f"Knowledge base ready: {len(knowledge_base)} cards")

except Exception as e:
    print("Knowledge base warning:", e)
    knowledge_base = []


# ============================================================
# HELPER
# ============================================================

def dataframe_to_records(dataframe):
    """
    Convert pandas DataFrame into JSON-safe records.
    """

    if dataframe is None or dataframe.empty:
        return []

    result = dataframe.copy()

    result = result.replace(
        {
            float("inf"): None,
            float("-inf"): None
        }
    )

    result = result.where(
        pd.notnull(result),
        None
    )

    return result.to_dict(orient="records")


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_answer(question):
    """
    Generate a grounded answer directly from the retail data.

    Gemini is NOT required for this function.

    All numbers come directly from the CSV-based analytics.
    """

    question_lower = question.lower().strip()

    inventory = get_inventory_status()
    predictions = get_stockout_predictions()
    alerts = get_stock_alerts()
    overstock = get_overstocked_products()
    top_products = get_top_selling_products()
    trends = get_sales_trends()

    # ========================================================
    # SALES SPIKE
    # ========================================================

    spike_keywords = [
        "sales spike",
        "biggest sales spike",
        "largest sales spike",
        "highest sales spike",
        "sales spikes",
        "biggest spike",
        "largest spike"
    ]

    if any(keyword in question_lower for keyword in spike_keywords):

        spikes = trends[
            trends["trend_status"] == "SPIKE"
        ].copy()

        if spikes.empty:
            return (
                "No significant sales spike was identified "
                "in the available sales data.",
                [],
                "deterministic"
            )

        spikes = spikes.sort_values(
            by="percentage_change",
            ascending=False
        )

        strongest = spikes.iloc[0]

        answer = (
            f"The product with the biggest sales spike is "
            f"{strongest['product_name']} "
            f"({strongest['product_id']}).\n\n"
            f"• Recent sales: {int(strongest['recent_sales'])} units\n"
            f"• Average sales: {float(strongest['average_sales']):.1f} units\n"
            f"• Change: {float(strongest['percentage_change']):+.1f}%\n"
            f"• Trend status: SPIKE\n\n"
            f"Recommended action: Monitor inventory for this product "
            f"and consider replenishment if the higher sales level continues."
        )

        return (
            answer,
            dataframe_to_records(spikes),
            "deterministic"
        )

    # ========================================================
    # SALES DROP
    # ========================================================

    drop_keywords = [
        "sales drop",
        "biggest sales drop",
        "largest sales drop",
        "highest sales drop",
        "sales decline",
        "sales decrease"
    ]

    if any(keyword in question_lower for keyword in drop_keywords):

        drops = trends[
            trends["trend_status"] == "DROP"
        ].copy()

        if drops.empty:
            return (
                "No significant sales drop was identified "
                "in the available sales data.",
                [],
                "deterministic"
            )

        drops = drops.sort_values(
            by="percentage_change",
            ascending=True
        )

        strongest = drops.iloc[0]

        answer = (
            f"The biggest sales drop is for "
            f"{strongest['product_name']} "
            f"({strongest['product_id']}).\n\n"
            f"• Recent sales: {int(strongest['recent_sales'])} units\n"
            f"• Average sales: {float(strongest['average_sales']):.1f} units\n"
            f"• Change: {float(strongest['percentage_change']):+.1f}%\n"
            f"• Trend status: DROP\n\n"
            f"Recommended action: Review recent demand for this product "
            f"before increasing inventory."
        )

        return (
            answer,
            dataframe_to_records(drops),
            "deterministic"
        )

    # ========================================================
    # STOCKOUT
    # ========================================================

    stockout_keywords = [
        "run out",
        "run-out",
        "stockout",
        "stock out",
        "stock-out",
        "most likely to run out",
        "will run out"
    ]

    if any(keyword in question_lower for keyword in stockout_keywords):

        if predictions.empty:
            return (
                "The available data is not sufficient to identify "
                "a stockout risk.",
                [],
                "deterministic"
            )

        highest_risk = predictions.iloc[0]

        matching = inventory[
            inventory["product_id"] == highest_risk["product_id"]
        ]

        answer = (
            f"The product most likely to run out of stock is "
            f"{highest_risk['product_name']} "
            f"({highest_risk['product_id']}).\n\n"
            f"• Current stock: {int(highest_risk['current_stock'])} units\n"
            f"• Reorder level: {int(highest_risk['reorder_level'])} units\n"
            f"• Average daily sales: "
            f"{float(highest_risk['average_daily_sales']):.1f} units/day\n"
            f"• Estimated stock remaining: "
            f"{float(highest_risk['estimated_stockout_days']):.2f} days\n"
            f"• Stockout risk: {highest_risk['stockout_risk']}\n\n"
            f"Recommended action: {highest_risk['recommended_action']}."
        )

        return (
            answer,
            dataframe_to_records(matching),
            "deterministic"
        )

    # ========================================================
    # REORDER
    # ========================================================

    reorder_keywords = [
        "reorder",
        "restock",
        "replenish",
        "replenishment",
        "which products need stock",
        "which product needs stock"
    ]

    if any(keyword in question_lower for keyword in reorder_keywords):

        reorder_products = predictions[
            predictions["recommended_action"].isin(
                [
                    "Reorder now",
                    "Reorder immediately"
                ]
            )
        ]

        if reorder_products.empty:
            return (
                "No products currently require immediate reordering "
                "based on the available inventory and sales data.",
                [],
                "deterministic"
            )

        lines = [
            "The following products should be considered "
            "for immediate replenishment:"
        ]

        for _, row in reorder_products.iterrows():

            lines.append(
                f"• {row['product_name']} ({row['product_id']}): "
                f"{int(row['current_stock'])} units in stock, "
                f"reorder level {int(row['reorder_level'])}, "
                f"estimated stockout in "
                f"{float(row['estimated_stockout_days']):.2f} days."
            )

        return (
            "\n".join(lines),
            dataframe_to_records(reorder_products),
            "deterministic"
        )

    # ========================================================
    # TOP SELLING
    # ========================================================

    top_keywords = [
        "top selling",
        "top-selling",
        "best selling",
        "best-selling",
        "most sold",
        "highest sales",
        "sales leader"
    ]

    if any(keyword in question_lower for keyword in top_keywords):

        if top_products.empty:
            return (
                "The available data is not sufficient to determine "
                "the top-selling product.",
                [],
                "deterministic"
            )

        top = top_products.iloc[0]

        answer = (
            f"The top-selling product is {top['product_name']} "
            f"with {int(top['quantity_sold'])} units sold.\n\n"
            f"The next strongest products are:\n"
        )

        for _, row in top_products.head(5).iloc[1:].iterrows():

            answer += (
                f"• {row['product_name']}: "
                f"{int(row['quantity_sold'])} units\n"
            )

        return (
            answer,
            dataframe_to_records(top_products.head(5)),
            "deterministic"
        )

    # ========================================================
    # OVERSTOCK
    # ========================================================

    overstock_keywords = [
        "overstock",
        "overstocked",
        "excess stock",
        "too much stock",
        "slow moving stock"
    ]

    if any(keyword in question_lower for keyword in overstock_keywords):

        if overstock.empty:
            return (
                "No products currently meet the overstock threshold "
                "in the available data.",
                [],
                "deterministic"
            )

        lines = [
            "The following products are currently classified "
            "as overstocked:"
        ]

        for _, row in overstock.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{int(row['current_stock'])} units in stock, "
                f"approximately {float(row['days_remaining']):.1f} "
                f"days of stock remaining."
            )

        return (
            "\n".join(lines),
            dataframe_to_records(overstock),
            "deterministic"
        )

    # ========================================================
    # SALES TRENDS
    # ========================================================

    trend_keywords = [
        "trend",
        "trends",
        "sales performance",
        "sales momentum",
        "increase",
        "decrease"
    ]

    if any(keyword in question_lower for keyword in trend_keywords):

        significant = trends[
            trends["trend_status"].isin(
                ["SPIKE", "DROP"]
            )
        ]

        if significant.empty:
            return (
                "No significant sales spikes or drops were identified "
                "in the available sales data.",
                [],
                "deterministic"
            )

        lines = [
            "The strongest sales signals in the available data are:"
        ]

        for _, row in significant.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{int(row['recent_sales'])} recent units, "
                f"average {float(row['average_sales']):.1f}, "
                f"{float(row['percentage_change']):+.1f}% "
                f"({row['trend_status']})."
            )

        return (
            "\n".join(lines),
            dataframe_to_records(significant),
            "deterministic"
        )

    # ========================================================
    # LOW STOCK
    # ========================================================

    low_stock_keywords = [
        "low stock",
        "low inventory",
        "stock alerts",
        "inventory alerts",
        "running low"
    ]

    if any(keyword in question_lower for keyword in low_stock_keywords):

        if alerts.empty:
            return (
                "There are currently no low-stock alerts "
                "in the available inventory data.",
                [],
                "deterministic"
            )

        lines = [
            "Current low-stock alerts:"
        ]

        for _, row in alerts.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{int(row['current_stock'])} units remaining, "
                f"approximately {float(row['days_remaining']):.1f} "
                f"days of stock."
            )

        return (
            "\n".join(lines),
            dataframe_to_records(alerts),
            "deterministic"
        )

    # ========================================================
    # PRODUCT PERFORMANCE
    # ========================================================

    if (
        "product performance" in question_lower
        or "performance" in question_lower
    ):

        top = top_products.head(5)

        lines = [
            "Top product performance based on total quantity sold:"
        ]

        for _, row in top.iterrows():

            lines.append(
                f"• {row['product_name']}: "
                f"{int(row['quantity_sold'])} units sold."
            )

        return (
            "\n".join(lines),
            dataframe_to_records(top),
            "deterministic"
        )

    # ========================================================
    # UNKNOWN QUESTION
    # ========================================================

    return (
        "The available data is not sufficient to answer this question.",
        [],
        "deterministic"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# INVENTORY API
# ============================================================

@app.route("/api/inventory")
def inventory_api():

    try:
        data = get_inventory_status()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Inventory API error:", error)

        return jsonify({
            "error": "Unable to load inventory data."
        }), 500


# ============================================================
# ALERTS API
# ============================================================

@app.route("/api/alerts")
def alerts_api():

    try:
        data = get_stock_alerts()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Alerts API error:", error)

        return jsonify({
            "error": "Unable to load stock alerts."
        }), 500


# ============================================================
# OVERSTOCK API
# ============================================================

@app.route("/api/overstock")
def overstock_api():

    try:
        data = get_overstocked_products()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Overstock API error:", error)

        return jsonify({
            "error": "Unable to load overstock data."
        }), 500


# ============================================================
# TOP PRODUCTS API
# ============================================================

@app.route("/api/top-products")
def top_products_api():

    try:
        data = get_top_selling_products()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Top products API error:", error)

        return jsonify({
            "error": "Unable to load top product data."
        }), 500


# ============================================================
# PREDICTIONS API
# ============================================================

@app.route("/api/predictions")
def predictions_api():

    try:
        data = get_stockout_predictions()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Predictions API error:", error)

        return jsonify({
            "error": "Unable to load stockout predictions."
        }), 500


# ============================================================
# SALES TRENDS API
# ============================================================

@app.route("/api/sales-trends")
def sales_trends_api():

    try:
        data = get_sales_trends()

        return jsonify(
            dataframe_to_records(data)
        )

    except Exception as error:

        print("Sales trends API error:", error)

        return jsonify({
            "error": "Unable to load sales trends."
        }), 500


# ============================================================
# ASK COPILOT
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    # ========================================================
    # STEP 1 — VALIDATE REQUEST
    # ========================================================

    try:
        data = request.get_json(
            silent=True
        )
    except Exception:
        data = None

    if not data:

        return jsonify({
            "answer": "No request data received.",
            "evidence": [],
            "source": "system",
            "status": "error"
        }), 400

    question = str(
        data.get("question", "")
    ).strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question.",
            "evidence": [],
            "source": "system",
            "status": "error"
        }), 400


    # ========================================================
    # STEP 2 — DETERMINISTIC ANSWER
    # ========================================================

    try:

        fallback_answer, fallback_evidence, fallback_source = (
            deterministic_answer(question)
        )

    except Exception as deterministic_error:

        print(
            "Deterministic analysis error:",
            deterministic_error
        )

        return jsonify({
            "answer": (
                "The available retail data could not be processed "
                "for this question."
            ),
            "evidence": [],
            "source": "system",
            "status": "error"
        }), 200


    # ========================================================
    # STEP 3 — BUILD GEMINI CONTEXT
    # ========================================================

    context = ""

    try:

        inventory = get_inventory_status()
        alerts = get_stock_alerts()
        overstock = get_overstocked_products()
        top_products = get_top_selling_products()
        predictions = get_stockout_predictions()
        trends = get_sales_trends()

        context_parts = []

        context_parts.append(
            "INVENTORY DATA:\n"
            + inventory.to_string(index=False)
        )

        context_parts.append(
            "STOCK ALERTS:\n"
            + alerts.to_string(index=False)
        )

        context_parts.append(
            "OVERSTOCK DATA:\n"
            + overstock.to_string(index=False)
        )

        context_parts.append(
            "TOP SELLING PRODUCTS:\n"
            + top_products.to_string(index=False)
        )

        context_parts.append(
            "STOCKOUT PREDICTIONS:\n"
            + predictions.to_string(index=False)
        )

        context_parts.append(
            "SALES TRENDS:\n"
            + trends.to_string(index=False)
        )

        context = "\n\n".join(
            context_parts
        )

    except Exception as data_error:

        print(
            "Context building warning:",
            data_error
        )

        context = ""


    # ========================================================
    # STEP 4 — LOCAL RETRIEVAL
    # ========================================================

    retrieved_context = []

    try:

        if knowledge_base:

            retrieved = retrieve_relevant_context(
                question,
                knowledge_base,
                top_k=3
            )

            for item in retrieved:

                if item.get("text"):

                    retrieved_context.append(
                        item["text"]
                    )

    except Exception as retrieval_error:

        print(
            "Retrieval warning:",
            retrieval_error
        )


    if retrieved_context:

        context += (
            "\n\nRETRIEVED KNOWLEDGE:\n"
            + "\n".join(
                retrieved_context
            )
        )


    # ========================================================
    # STEP 5 — TRY GEMINI
    # ========================================================

    try:

        if context:

            gemini_answer = ask_gemini(
                question,
                context
            )

            if (
                gemini_answer
                and gemini_answer.strip()
            ):

                return jsonify({
                    "answer": gemini_answer,
                    "evidence": fallback_evidence,
                    "source": "gemini",
                    "status": "success"
                }), 200

    except Exception as gemini_error:

        error_text = str(gemini_error)

        # -----------------------------------------------
        # Gemini quota / rate-limit error
        # -----------------------------------------------

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):

            print(
                "Gemini quota unavailable. "
                "Using deterministic fallback."
            )

        # -----------------------------------------------
        # Other Gemini errors
        # -----------------------------------------------

        else:

            print(
                "Gemini unavailable. "
                "Using deterministic fallback."
            )

            print(
                gemini_error
            )


    # ========================================================
    # STEP 6 — DETERMINISTIC FALLBACK
    # ========================================================

    fallback_message = fallback_answer

    if fallback_answer:

        fallback_message += (
            "\n\n"
            "ℹ️ Gemini is temporarily unavailable. "
            "This answer was generated using deterministic "
            "analysis from the available retail data."
        )

    return jsonify({
        "answer": fallback_message,
        "evidence": fallback_evidence,
        "source": "deterministic",
        "status": "fallback"
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "error": "Internal server error."
    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" Retail Sales & Inventory Copilot")
    print("==========================================")
    print(" Server: http://localhost:8000")
    print("==========================================")
    print()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )