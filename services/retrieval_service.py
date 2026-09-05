import os
import numpy as np

from services.embedding_service import create_embedding


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# --------------------------------------------------
# Create retail knowledge cards
# --------------------------------------------------

def create_knowledge_cards():

    cards = [

        {
            "id": "inventory_status",
            "text": """
            Inventory status contains current stock, reorder level,
            average daily sales, days remaining and stock status
            for all products.
            """
        },

        {
            "id": "stock_alerts",
            "text": """
            Stock alerts identify products whose inventory is getting low
            and may need attention or replenishment.
            """
        },

        {
            "id": "overstock",
            "text": """
            Overstocked products are products with high inventory levels
            compared with their average daily sales.
            """
        },

        {
            "id": "top_sales",
            "text": """
            Top selling products contain products ranked by their
            total quantity sold.
            """
        },

        {
            "id": "product_performance",
            "text": """
            Product performance can be evaluated using sales quantity,
            current inventory, average daily sales and inventory status.
            """
        },

        {
            "id": "reorder",
            "text": """
            Reorder decisions should consider current stock,
            reorder level and average daily sales.
            """
        },

        {
            "id": "sales_trends",
            "text": """
            Sales data contains daily quantities sold for each product
            and can be used to identify sales increases or decreases.
            """
        }

    ]

    return cards


# --------------------------------------------------
# Cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    denominator = (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


# --------------------------------------------------
# Build embeddings for knowledge cards
# --------------------------------------------------

def build_knowledge_base():

    cards = create_knowledge_cards()

    knowledge_base = []

    for card in cards:

        embedding = create_embedding(
            card["text"]
        )

        knowledge_base.append({
            "id": card["id"],
            "text": card["text"],
            "embedding": embedding
        })

    return knowledge_base


# --------------------------------------------------
# Retrieve relevant knowledge
# --------------------------------------------------

def retrieve_relevant_context(
    question,
    knowledge_base,
    top_k=3
):

    question_embedding = create_embedding(
        question
    )

    results = []

    for item in knowledge_base:

        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "id": item["id"],
            "text": item["text"],
            "score": score
        })


    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return results[:top_k]