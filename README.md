# Amazon Sales Intelligence & AI Demand Forecasting System

> **IBM Internship Project** — Nakul Mehra | MCA | Tula's Institute | 2024

A fully interactive, professional-grade Streamlit business intelligence dashboard built on a real Amazon India e-commerce dataset. The system covers the complete data science pipeline — from data quality auditing and cleaning through to machine learning demand forecasting and auto-generated business insights.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Features](#features)
5. [Dataset Description](#dataset-description)
6. [Technologies Used](#technologies-used)
7. [Dashboard Modules](#dashboard-modules)
8. [Machine Learning Methodology](#machine-learning-methodology)
9. [AI Forecasting](#ai-forecasting)
10. [Installation](#installation)
11. [Running Instructions](#running-instructions)
12. [Project Structure](#project-structure)
13. [Limitations](#limitations)
14. [Future Scope](#future-scope)

---

## Project Overview

**Amazon Sales Intelligence & AI Demand Forecasting System** transforms 128,975 raw Amazon India transaction records (March–June 2022) into an actionable multi-page business intelligence dashboard. It features interactive filters, Plotly visualisations, ML-powered demand forecasting, and dynamically generated business insights — all in a single Python Streamlit application with no backend server, no JavaScript framework, and no database.

---

## Problem Statement

Amazon sellers and category managers working with large transaction datasets lack accessible tools to:

1. Rapidly assess business health (revenue, orders, cancellations) from a single view.
2. Identify which products, categories, sizes, and regions drive revenue.
3. Detect data quality issues that could distort analysis.
4. Forecast future demand per category for inventory planning.
5. Generate data-driven recommendations without manual statistical analysis.

This project solves all five problems in a single Python application.

---

## Objectives

1. Perform a comprehensive data quality audit of the raw Amazon Sales dataset.
2. Design and implement a robust data cleaning and preprocessing pipeline.
3. Build an interactive multi-page dashboard with global filtering capabilities.
4. Visualise sales performance across time, category, size, geography, and product dimensions.
5. Analyse order fulfilment efficiency and identify cancellation drivers.
6. Train and evaluate machine learning models for demand forecasting.
7. Generate automated, data-driven business insights and recommendations.
8. Package the application as a professional, internship-quality deliverable.

---

## Features

| Module | What it Does |
|---|---|
| **Executive Overview** | 6 live KPI cards + trend charts responding to global filters |
| **Data Quality** | 11-check quality audit with charts, tables, and IQR outlier analysis |
| **Sales Analytics** | Daily/Weekly/Monthly switchable trends, SKU ranking, day-of-week heatmap |
| **Geographic Analytics** | State/city revenue ranking, revenue share pie, bubble scatter |
| **Order & Fulfilment** | Status/courier/fulfilment breakdowns, B2B vs B2C, cancellation by category |
| **Product Analytics** | Top/bottom 15 products, category × size heatmap, top 30 SKU table |
| **AI Demand Forecasting** | RF/GB regression with lag features, actual vs predicted, future forecast |
| **AI Business Insights** | 12 auto-generated insight cards + prioritised recommendation table |
| **Global Filters** | Date range, Category, Fulfilment, B2B/B2C — all charts update live |

---

## Dataset Description

| Property | Value |
|---|---|
| **File** | `Amazon Sale Report.csv` |
| **Source** | [Kaggle — Amazon Sale Report](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data) |
| **Rows** | 128,975 |
| **Columns** | 24 (23 usable) |
| **Date Range** | 31 March 2022 – 29 June 2022 (91 days) |
| **Unique Orders** | 120,378 |
| **Total Revenue** | ₹75.40 M (INR) |
| **Total Quantity** | 116,649 units |
| **Avg Order Value** | ₹626 |
| **Cancellation Rate** | 14.2% |
| **Categories** | 9 (Set, Kurta, Western Dress, Top, Ethnic Dress, Blouse, Bottom, Saree, Dupatta) |
| **States covered** | 35 Indian states/UTs (after normalisation) |
| **Platform** | Amazon.in (99.9%) |

> The original CSV is never modified. The application opens it in read-only mode.

---

## Technologies Used

| Layer | Library | Version |
|---|---|---|
| UI / Frontend | Streamlit | ≥ 1.32.0 |
| Data Processing | Pandas | ≥ 2.0.0 |
| Numerical | NumPy | ≥ 1.26.0 |
| Visualisation | Plotly | ≥ 5.19.0 |
| Machine Learning | Scikit-learn | ≥ 1.4.0 |
| Language | Python | 3.10+ |

**Not used:** React, Node.js, JavaScript, Flask, Django, any database.

---

## Dashboard Modules

### 🏠 Executive Overview
Six KPI cards: Total Orders, Total Revenue (₹75.40 M on full dataset), Total Quantity, Average Order Value, Cancellation Rate, and Top Category. Supporting charts: daily revenue area, category revenue bar, status donut, size distribution, fulfilment split.

### 📊 Data Quality
Complete data quality audit across six tabs: missing values bar chart + table, data types, unique value cardinality, numerical statistics (Qty/Amount), IQR outlier box-plots, and inconsistency analysis. All computed from the raw unmodified CSV.

### 📈 Sales Analytics
Granularity-switchable trend charts (Daily / Weekly / Monthly) with dual-axis Revenue + Orders. Category-wise and size-wise breakdowns. Top 20 SKUs chart. Day-of-week × Category revenue heatmap.

### 🗺️ Geographic Analytics
Top 20 states by revenue (horizontal bar), revenue share pie (top 10 states), orders vs revenue bubble scatter (bubble = quantity), top 25 cities bar, full state expandable table.

### 📦 Order & Fulfilment
Order status pie (13 statuses), fulfilment type pie, courier status pie. Daily shipped vs cancelled stacked bar. Service level distribution. B2B vs B2C split. Cancellation rate by category.

### 🛍️ Product Analytics
Top 15 and bottom 15 products by revenue (by Style/Category). Category performance summary table. Category × Size quantity heatmap. Top 30 SKUs interactive table.

### 🤖 AI Demand Forecasting
User selects Category (All or specific), Model (Random Forest / Gradient Boosting), and Forecast Horizon (7 / 14 / 30 days). The model trains on 61 days and tests on 16 days. Outputs: MAE, RMSE, R² metrics; actual vs predicted chart; historical + future forecast chart; forecasted quantities table; feature importance chart.

### 💡 AI Business Insights
12 insight cards generated entirely from live data: top category, high-cancellation category, top region, sales trend direction, best/worst SKU, best/worst day, cancellation health, fulfilment recommendation, B2B opportunity, promotion analysis, geographic concentration risk. Prioritised recommendation table (High / Medium / Low).

---

## Machine Learning Methodology

### Problem Formulation
Supervised regression on a daily time-series. Target variable: total daily quantity sold. Future demand predicted using lagged historical values and calendar features.

### Features (15 total)

| Type | Features |
|---|---|
| Calendar | `day_of_week`, `day_of_month`, `week_of_year`, `month`, `is_weekend` |
| Lag | `lag_1`, `lag_3`, `lag_7`, `lag_14` |
| Rolling mean | `rolling_mean_3`, `rolling_mean_7`, `rolling_mean_14` |
| Rolling std | `rolling_std_3`, `rolling_std_7`, `rolling_std_14` |

### Train / Test Split
- **Strategy:** Temporal 80/20 split — no shuffling, no cross-validation
- **Train:** First 61 days (chronological)
- **Test:** Last 16 days
- **No data leakage:** All lag/rolling features use `shift(1)`

### Models
- **Random Forest Regressor** — 200 estimators
- **Gradient Boosting Regressor** — 200 estimators

---

## AI Forecasting

### Actual Model Metrics (All Categories, Random Forest)

| Metric | Value |
|---|---|
| MAE | 119.97 units/day |
| RMSE | 171.69 units/day |
| R² | −0.39 |

> **Note on R²:** Negative R² is expected with only 91 days of data (16-day test set). The models provide directionally reasonable trend continuation rather than precise point-forecasting.

Future-period forecasting uses iterative auto-regressive prediction — each forecast day uses the previously predicted value as the lag-1 input.

---

## Installation

### Prerequisites
- Python 3.10 or later
- `Amazon Sale Report.csv` in the project directory

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Instructions

```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501**.

The first load caches the dataset and cleans it (a few seconds). Subsequent page navigation is instant.

---

## Project Structure

```
IBM_PROJECT/
│
├── app.py                                        ← Streamlit application (all logic)
├── Amazon Sale Report.csv                        ← Source dataset (READ-ONLY)
├── requirements.txt                              ← Python dependencies (5 packages)
├── README.md                                     ← This file
├── Nakul_Mehra_Amazon_Sales_ProjectReport.docx   ← Full project report
└── .gitignore                                    ← Version control exclusions
```

---

## Limitations

1. **91-day data window** — insufficient for capturing seasonality; limits forecasting accuracy.
2. **No customer IDs** — customer-level analysis (CLV, churn) is not possible.
3. **No cost/margin data** — only revenue (selling price) is available; profit analysis is not possible.
4. **No inventory data** — stock level management features cannot be built.
5. **Negative R² scores** — tree-based models underperform a naïve mean baseline on the 16-day test set due to the short data window.

---

## Future Scope

1. Extend dataset to 2+ years for seasonality modelling.
2. Add Facebook Prophet or SARIMA for time-series forecasting.
3. Integrate Amazon Seller Central API for live data refresh.
4. Add customer-level analytics if customer IDs become available.
5. Deploy to Streamlit Community Cloud or IBM Cloud.

---

*Built with Python · Streamlit · Plotly · Scikit-learn*
*Nakul Mehra | MCA | Tula's Institute | IBM Internship Project 2024*
*Dataset: Amazon Sale Report.csv (128,975 rows, 91 days)*
