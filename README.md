# Retail Sales & Inventory Copilot

A data-grounded AI copilot for small retail operations that helps managers
understand sales performance, inventory health, stockout risks, overstock,
and replenishment needs.


## Deployment link

https://retail-copilot-owbv.onrender.com/

## Overview

Retail Sales & Inventory Copilot combines deterministic retail analytics
with Gemini-powered natural-language assistance.

A manager can ask questions such as:

- Which product has the biggest sales spike?
- Which product is most likely to run out of stock?
- Which products need to be reordered?
- Which products are overstocked?
- What is the top-selling product?
- Are there any low-stock alerts?

The system provides answers using the available sales and inventory data
and includes supporting evidence.

The application is designed to avoid inventing business data. Numerical
analysis is performed by Python, while Gemini is used to explain the
available information in a natural-language format.

---

## Key Features

### 1. Product Catalogue

The application maintains product information including:

- Product ID
- Product name
- Category
- Price

### 2. Inventory Monitoring

Tracks:

- Current stock
- Reorder level
- Average daily sales
- Estimated days of stock remaining
- Inventory health

### 3. Stockout Risk Prediction

Identifies products that may run out of stock soon.

Risk levels:

- HIGH
- MEDIUM
- LOW

The system also provides a recommended action such as:

- Reorder now
- Reorder immediately
- Plan replenishment soon
- Continue monitoring

### 4. Sales Trend Detection

Detects significant changes in sales activity:

- SPIKE
- DROP
- STABLE

### 5. Overstock Detection

Identifies products with excessive inventory based on the available
inventory and sales data.

### 6. Top Product Performance

Ranks products based on total quantity sold.

### 7. Natural-Language Copilot

Managers can ask questions in plain English instead of manually
analysing tables.

### 8. Evidence-Grounded Answers

Answers are supported by the underlying retail data.

The system does not intentionally fabricate unavailable numbers or
business information.

### 9. Graceful Gemini Fallback

If Gemini is temporarily unavailable, including quota or network
availability issues, the application falls back to deterministic
Python-based analysis where possible.

The user receives a professional response instead of a raw API error.

---

## Technology Stack

### Backend

- Python 3.11
- Flask
- Pandas
- NumPy

### AI

- Google Gemini API
- Gemini Flash model for natural-language reasoning
- Gemini Embedding model for local semantic retrieval

### Frontend

- HTML5
- CSS3
- JavaScript
- Chart-based dashboard visualizations

### Data

- CSV-based retail datasets
- Local knowledge cards
- Local semantic retrieval

### Development

- VS Code
- Git
- GitHub

---

## Architecture

```text
                    ┌─────────────────────┐
                    │       Manager       │
                    │   Natural Language  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Flask App       │
                    │     app.py          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Deterministic│ │    Local     │ │    Gemini    │
      │  Analytics   │ │  Retrieval   │ │   Reasoning  │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Grounded Copilot    │
                    │      Response       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Dashboard       │
                    │ Answer + Evidence   │
                    └─────────────────────┘
