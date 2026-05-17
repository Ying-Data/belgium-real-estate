# Belgium Real Estate — Property Valuation Intelligence

**End-to-end data analysis and machine learning project on 15.254 Belgian residential property listings from Immoweb.**

---

## Project Summary

Belgian residential property is listed at widely varying prices across regions, property types, and conditions. This project delivers a data-driven valuation model and market analysis to help investors and agencies identify over- and under-priced listings systematically.

**Core business question:** Which combination of region, property type, size, and condition drives the highest price per square metre — and can we predict fair market value to flag mispriced listings?

---

## Key Findings

| Metric | Value |
|---|---|
| National median listing price | € 353.750 |
| Brussels median price per m² | € 3.333 |
| Brussels premium over Wallonie | +93% |
| Renovation value uplift (AS_NEW vs TO_RENOVATE) | +61% · € 168.000 |
| Buy-renovate-sell ROI (Flanders scenario) | 9,6% · € 38.325 net profit |

**Living area is the strongest price driver**, explaining 58,2% of model variance — ahead of region (8,8%) and property subtype (5,3%).

---

## Model Purpose

This model estimates the fair market value of a residential property based on its location, size, type, and condition — so investors and agencies can identify over- and under-priced listings relative to market comparables.

---

## Repository Structure

This repository uses a **flat folder structure** — all deliverable files are in the root, with only `screenshots/` as a subfolder. This is intentional so that recruiters and hiring managers see the full scope of work immediately without navigating subfolders.

```
/
├── belgium_re_pipeline.py             # Python pipeline: cleaning → EDA → ML → exports
├── belgium_re_executive_report.html   # A4 executive report (3 pages, print-ready)
├── belgium_re_powerbi_export.csv      # Cleaned, enriched Power BI-ready dataset
├── belgium_re_feature_importance.csv  # ML feature importance scores
├── README.md                          # This file
├── .gitignore                         # Excludes raw data, pycache, OS files, .pbix
└── screenshots/                       # Dashboard preview images
    ├── 01_market_overview.png
    ├── 02_regional_analysis.png
    ├── 03_valuation_gap.png
    └── 04_executive_report.png
```

---

## Dashboard Preview

### Page 1 — Market Overview
![Market Overview](screenshots/01_market_overview.png)

### Page 2 — Regional Analysis
![Regional Analysis](screenshots/02_regional_analysis.png)

### Page 3 — Valuation Gap — Opportunity Finder
![Valuation Gap](screenshots/03_valuation_gap.png)

### Executive Report — Cover Page
![Executive Report Cover](screenshots/04_executive_report.png)

---

## Deliverables

### 1. Python Pipeline (`belgium_re_pipeline.py`)
Full end-to-end pipeline covering data loading, quality auditing, EDA, feature engineering, Random Forest regression, ROI scenario model, and CSV exports. Runs from the folder containing the raw data file — no path changes required.

**Prerequisites:**
```
pip install pandas numpy scikit-learn
```

**Run:**
```
python belgium_re_pipeline.py
```

### 2. Executive Report (`belgium_re_executive_report.html`)
A4 print-ready, 3-page report with:
- Page 1: 4 business-facing KPI tiles
- Page 2: Executive summary with findings and recommendations (quantified impact for each)
- Page 3: Data quality log, ML methodology, ROI assumption log

**Print settings:** Margins → None · Scale → 100% · Background graphics → ON


### 3. Power BI Export (`belgium_re_powerbi_export.csv`)
Cleaned, feature-engineered dataset produced by the pipeline. This is **not raw data** — it includes derived columns:

| Column | Description |
|---|---|
| `price_per_sqm` | price ÷ living_area (€ per m²) |
| `predicted_price` | Random Forest model estimate of fair market value |
| `valuation_gap` | Actual price − predicted price (positive = overpriced signal) |
| `valuation_gap_pct` | valuation_gap as % of predicted_price |
| `kitchen_score` | Ordinal: 0 (not installed) → 3 (hyper equipped) |
| `building_score` | Ordinal: 0 (to restore) → 5 (as new) |

### 4. Feature Importance (`belgium_re_feature_importance.csv`)
Random Forest feature importances for the 12 model inputs. Use as data source for a bar chart in the methodology section of the Power BI report.

---

## Data Quality Summary

| Issue | Records | Treatment |
|---|---|---|
| Duplicate URLs (same listing scraped twice) | 2.821 | Deduplicated — prices identical |
| Corrupted living_area values | 1.636 | Set to NaN — scraper artefact |
| surface_land = 'UNKNOWN' | 2.385 | Set to NaN |
| Missing price | 498 | Excluded — target variable |
| Missing region | 121 | Excluded — key predictor |
| Price outliers (below p1 or above p99) | ~305 | Excluded — non-residential |
| living_area > 2.000 m² | 8 | Excluded — implausible residential |

**Final clean dataset:** 15.254 listings · 3 regions · 11 provinces · 21 property subtypes

---

## ML Model Performance

- **Algorithm:** Random Forest Regressor · 200 trees
- **Train / Test split:** 80% / 20% (random_state = 42)
- **R²:** 0,635 · **MAE:** € 128.858 · **MAPE:** 30,9%

R² of 0,635 means the model explains 63,5% of price variation using publicly observable listing attributes. Unobserved factors — micro-location quality, interior finish, views, floor level — account for the remainder. This is consistent with academic real estate valuation benchmarks where models trained solely on listing metadata typically achieve R² of 0,55–0,70.

---

## ROI Scenario Model

**Scenario:** Buy TO_RENOVATE in Flanders → renovate → sell at AS_NEW median.

| Input | Value | Source |
|---|---|---|
| Purchase price | € 289.000 | From data (Flanders TO_RENOVATE median) |
| Renovation cost | € 90.000 | Assumed: € 600/m² × 150 m² |
| Transaction costs | € 21.675 | Assumed: 7,5% of purchase price |
| Resale price | € 439.000 | From data (Flanders AS_NEW median) |
| **Net profit** | **€ 38.325** | |
| **ROI** | **9,6%** | |

---

## Methodology Notes

- **Data source:** Immoweb (Belgium's largest property portal) — publicly scraped listing data.
- **Scope:** Residential properties only. Commercial, industrial, and mixed-use listings retained where labelled as HOUSE-type (APARTMENT_BLOCK, MIXED_USE_BUILDING) as they appear in the residential search results.
- **Price definition:** Asking price as listed. No adjustment for negotiation margin.
- **Number format:** Dutch locale — period (.) = thousands separator, comma (,) = decimal (e.g. € 353.750 · € 2.237,40/m²).
- **Flat folder structure:** All files in the root of the repository. Only `screenshots/` is a subfolder. Recruiters see the complete deliverable set immediately on landing on the repo page.
- **.gitignore:** Raw source data (`Belgium_real_estate_market.csv`), Python bytecode (`__pycache__/`), OS metadata (`.DS_Store`, `Thumbs.db`), virtual environments, Power BI working files (`.pbix`), and Excel lock files (`~$*.xlsx`) are all excluded from the repository.

---

## About This Project

**Author:** Ying — Data Analyst  
**Tools:** Python (pandas, scikit-learn), Power BI Desktop, HTML/CSS  
**Dataset:** Belgium Real Estate Market (Immoweb scrape)  
**Portfolio focus:** End-to-end analytical thinking — from raw data to executive recommendations with quantified business impact.
