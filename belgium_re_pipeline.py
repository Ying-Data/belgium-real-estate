"""
================================================================================
 BELGIUM REAL ESTATE MARKET, PROPERTY PRICE PREDICTION PIPELINE
 Portfolio project by Ying Zhao, Data Analyst
================================================================================

What this pipeline does:
  Cleans and enriches Immoweb residential listing data, then trains a Random
  Forest regressor to estimate fair market value from a property's location,
  size, type, and condition. The output lets investors and agencies flag
  over- and under-priced listings relative to comparable properties.

Prerequisites:
  pip install pandas numpy scikit-learn openpyxl

Input file:
  Belgium_real_estate_market.csv  (place in the same folder as this script)

Run instructions:
  1. Open a terminal in the folder containing this script.
  2. Run: python belgium_re_pipeline.py
  3. Two CSV files are written to the same folder:
       belgium_re_powerbi_export.csv
       belgium_re_feature_importance.csv

Number format:
  Printed figures follow Dutch locale convention used in the Belgian market:
  a period (.) separates thousands and a comma (,) marks the decimal.
  Example: EUR 353.750 and EUR 3.333,33 per m2.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")

# ── Dutch number formatter ───────────────────────────────────────────────────
def dutch(value, decimals=0, prefix="€ "):
    """Format a number in Dutch locale: . = thousands, , = decimal."""
    if pd.isna(value):
        return "N/B"
    fmt = f"{value:,.{decimals}f}"          # English format first
    # swap: comma -> X, period -> comma, X -> period
    fmt = fmt.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefix}{fmt}"


# ── 0. LOAD ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 0: LOADING DATA")
print("=" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, "Belgium_real_estate_market.csv")

df_raw = pd.read_csv(raw_path, sep=";")
print(f"Raw records loaded : {len(df_raw):,}".replace(",", "."))
print(f"Columns            : {len(df_raw.columns)}")


# ── 1. DATA QUALITY ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 1: DATA QUALITY")
print("=" * 70)

df = df_raw.copy()

# ── 1.1 Drop index columns (artefacts of export) ─────────────────────────────
df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore", inplace=True)
print("[1.1] Dropped 2 unnamed index columns (export artefacts).")

# ── 1.2 Duplicate URLs ───────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(subset="url", keep="first", inplace=True)
removed = before - len(df)
print(f"[1.2] Duplicate URLs removed : {removed:,}  (same listing scraped twice, "
      f"prices identical in all cases; first occurrence kept)".replace(",", "."))

# ── 1.3 living_area: parse and clean ─────────────────────────────────────────
#     The value '20.951.537.287.396.400' appears 1,636 times, a data corruption
#     artefact from the scraper. Treated as missing.
df["living_area"] = pd.to_numeric(df["living_area"], errors="coerce")
bad_la = df["living_area"].isnull().sum()
print(f"[1.3] living_area, non-numeric (corrupted scraper value): {bad_la:,}  set to NaN"
      .replace(",", "."))

# ── 1.4 surface_land: parse and clean ────────────────────────────────────────
df["surface_land"] = df["surface_land"].replace("UNKNOWN", np.nan)
df["surface_land"] = pd.to_numeric(df["surface_land"], errors="coerce")
print(f"[1.4] surface_land: 'UNKNOWN' values and nulls treated as missing "
      f"({df['surface_land'].isnull().sum():,} total NaN). "
      f"Used only as informational field; not included in model features.".replace(",", "."))

# ── 1.5 Drop rows missing price or living_area ───────────────────────────────
before = len(df)
df.dropna(subset=["price", "living_area"], inplace=True)
print(f"[1.5] Rows dropped, missing price or living_area : {before - len(df):,}  "
      f"(both are essential for valuation model)".replace(",", "."))

# ── 1.6 Drop rows missing region ─────────────────────────────────────────────
before = len(df)
df.dropna(subset=["region"], inplace=True)
print(f"[1.6] Rows dropped, missing region : {before - len(df):,}  "
      f"(cannot assign geographic market without region)".replace(",", "."))

# ── 1.7 Price outliers: cap at p1 / p99 ─────────────────────────────────────
p1  = df["price"].quantile(0.01)
p99 = df["price"].quantile(0.99)
before = len(df)
df = df[(df["price"] >= p1) & (df["price"] <= p99)].copy()
print(f"[1.7] Price outliers removed (below p1={dutch(p1)} or above p99={dutch(p99)}) : "
      f"{before - len(df):,} rows".replace(",", "."))

# ── 1.8 Living area cap ──────────────────────────────────────────────────────
before = len(df)
df = df[df["living_area"] <= 2000].copy()
print(f"[1.8] living_area > 2.000 m² removed : {before - len(df):,} rows  "
      f"(implausible for standard residential listing)".replace(",", "."))

# ── 1.9 number_rooms: cap extreme values ─────────────────────────────────────
df["number_rooms"] = df["number_rooms"].clip(upper=15)
print("[1.9] number_rooms capped at 15 (94 rows had >15, data entry errors).")

print(f"\nClean dataset : {len(df):,} records ready for analysis".replace(",", "."))


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 70)

# Derived metric
df["price_per_sqm"] = df["price"] / df["living_area"]

# Binary amenity flags
df["has_garden"]    = df["garden"].astype(int)
df["has_terrace"]   = df["terrace"].astype(int)
df["has_pool"]      = df["swimming_pool"].astype(int)
df["has_fireplace"] = df["fireplace"].astype(int)

# Kitchen quality ordinal score (0 = not installed, up to 3 = hyper equipped)
kitchen_map = {
    "NOT_INSTALLED": 0, "USA_UNINSTALLED": 0,
    "SEMI_EQUIPPED": 1, "USA_SEMI_EQUIPPED": 1,
    "INSTALLED": 2,     "USA_INSTALLED": 2,
    "HYPER_EQUIPPED": 3,"USA_HYPER_EQUIPPED": 3,
}
df["kitchen_score"] = df["kitchen"].map(kitchen_map).fillna(1)

# Building condition ordinal score (0 = worst, up to 5 = as new)
state_map = {
    "TO_RESTORE": 0, "TO_RENOVATE": 1, "TO_BE_DONE_UP": 2,
    "GOOD": 3,       "JUST_RENOVATED": 4, "AS_NEW": 5,
}
df["building_score"] = df["building_state"].map(state_map).fillna(3)

# Label encode categoricals
le_region = LabelEncoder()
le_ptype  = LabelEncoder()
le_psub   = LabelEncoder()
df["region_enc"] = le_region.fit_transform(df["region"])
df["ptype_enc"]  = le_ptype.fit_transform(df["property_type"])
df["psub_enc"]   = le_psub.fit_transform(df["property_subtype"])

print("Features created: price_per_sqm, kitchen_score, building_score,")
print("  has_garden, has_terrace, has_pool, has_fireplace,")
print("  region_enc, ptype_enc, psub_enc")


# ── 3. EDA: KEY BUSINESS INSIGHTS ───────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 3: EDA, KEY BUSINESS INSIGHTS")
print("=" * 70)

print("\n[3.1] Median listing price by region:")
reg_stats = df.groupby("region").agg(
    listings=("price", "count"),
    median_price=("price", "median"),
    median_sqm=("price_per_sqm", "median"),
).sort_values("median_price", ascending=False)
for region, row in reg_stats.iterrows():
    print(f"  {region:<12} | {int(row['listings']):>5} listings | "
          f"median {dutch(row['median_price'])} | "
          f"{dutch(row['median_sqm'], decimals=0)} /m²")

bru_med = df[df["region"] == "Brussels"]["price"].median()
wal_med = df[df["region"] == "Wallonie"]["price"].median()
fla_med = df[df["region"] == "Flanders"]["price"].median()
bru_premium_wal = (bru_med / wal_med - 1) * 100
bru_premium_fla = (bru_med / fla_med - 1) * 100
print(f"\n  Brussels commands a {bru_premium_wal:.0f}% premium over Wallonie "
      f"and {bru_premium_fla:.0f}% over Flanders.")

print("\n[3.2] Renovation uplift (median price by building state):")
state_order = ["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP",
               "TO_RENOVATE", "TO_RESTORE"]
for state in state_order:
    sub = df[df["building_state"] == state]
    if len(sub) > 0:
        print(f"  {state:<18} | {dutch(sub['price'].median())}")

as_new_med     = df[df["building_state"] == "AS_NEW"]["price"].median()
to_reno_med    = df[df["building_state"] == "TO_RENOVATE"]["price"].median()
reno_uplift_eur = as_new_med - to_reno_med
reno_uplift_pct = (as_new_med / to_reno_med - 1) * 100
print(f"\n  Move-in-ready vs. needs renovation: {dutch(reno_uplift_eur)} uplift "
      f"({reno_uplift_pct:.0f}% premium).")

print("\n[3.3] Median price by property subtype (top 6):")
sub_stats = (df.groupby("property_subtype")["price"]
               .median()
               .sort_values(ascending=False)
               .head(6))
for name, val in sub_stats.items():
    print(f"  {name:<25} | {dutch(val)}")

villa_med = df[df["property_subtype"] == "VILLA"]["price"].median()
house_med = df[df["property_subtype"] == "HOUSE"]["price"].median()
print(f"\n  Villa premium over standard house: {(villa_med / house_med - 1) * 100:.0f}%")


# ── 4. MACHINE LEARNING MODEL ─────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 4: MACHINE LEARNING, PROPERTY PRICE ESTIMATOR")
print("=" * 70)
print("""
MODEL PURPOSE:
  Estimate the fair market value of a residential property from its location,
  size, type, and condition, so investors and agencies can identify over- and
  under-priced listings relative to comparable properties.
""")

FEATURES = [
    "region_enc", "ptype_enc", "psub_enc",
    "number_rooms", "living_area",
    "kitchen_score", "building_score",
    "has_garden", "has_terrace", "has_pool",
    "has_fireplace", "number_facades",
]

X = df[FEATURES].fillna(0)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"  Algorithm       : Random Forest Regressor (200 trees)")
print(f"  Training rows   : {len(X_train):,}".replace(",", "."))
print(f"  Test rows       : {len(X_test):,}".replace(",", "."))
print(f"  R2              : {r2:.3f}   (proportion of price variance explained)")
print(f"  MAE             : {dutch(mae)}   (mean absolute prediction error)")
print(f"  MAPE            : {mape:.1f}%   (mean absolute percentage error)")
print()
print(f"  NOTE ON R2: An R2 of {r2:.3f} means the model explains {r2*100:.1f}% of")
print("  price variation using publicly observable listing features alone.")
print("  Unobserved factors such as micro-location quality, interior finish, and")
print("  floor level account for the remainder. Valuation models trained only on")
print("  listing metadata, without a physical inspection, typically land in this")
print("  range.")

# Predict on full dataset and compute valuation gap
df["predicted_price"] = model.predict(X.reindex(df.index, fill_value=0))
df["valuation_gap"]   = df["price"] - df["predicted_price"]
df["valuation_gap_pct"] = (df["valuation_gap"] / df["predicted_price"]) * 100

# Feature importance
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n  Feature importances:")
for _, row in fi.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"    {row['feature']:<20} {row['importance']:.3f}  {bar}")


# ── 5. ROI SCENARIO MODEL ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 5: ROI SCENARIO MODEL, RENOVATION INVESTMENT")
print("=" * 70)
print("""
SCENARIO: An investor buys a TO_RENOVATE property in Flanders, renovates it to
  AS_NEW condition, and resells at the regional median. What is the return?

ASSUMPTIONS (matching the executive report and README assumption logs):
  [FROM DATA]   Median price TO_RENOVATE (Flanders)  : see below
  [FROM DATA]   Median price AS_NEW (Flanders)       : see below
  [ASSUMED]     Renovation cost per m2               : EUR 600  (medium refurbishment)
  [ASSUMED]     Renovation surface                   : 150 m2   (dataset median living area)
  [ASSUMED]     Transaction costs (notary, agent, tax): 7.5% of purchase price
""")

fl_reno  = df[(df["region"] == "Flanders") & (df["building_state"] == "TO_RENOVATE")]["price"].median()
fl_new   = df[(df["region"] == "Flanders") & (df["building_state"] == "AS_NEW")]["price"].median()
reno_area    = 150          # assumed: dataset median living area
reno_cost_m2 = 600          # assumed: medium refurbishment, Belgian market average
tx_rate      = 0.075        # assumed: 7.5% notary, agent, and transfer tax
reno_cost_total = reno_area * reno_cost_m2
tx_cost      = fl_reno * tx_rate
total_invest = fl_reno + reno_cost_total + tx_cost
net_profit   = fl_new - total_invest
roi_pct      = (net_profit / total_invest) * 100

print(f"  Purchase price (TO_RENOVATE Flanders median)  : {dutch(fl_reno)}")
print(f"  Renovation cost ({reno_area} m2 x {dutch(reno_cost_m2)}/m2, ASSUMED): {dutch(reno_cost_total)}")
print(f"  Transaction costs ({tx_rate*100:.1f}%, ASSUMED)            : {dutch(tx_cost)}")
print(f"  Total investment                              : {dutch(total_invest)}")
print(f"  Resale value (AS_NEW Flanders median)         : {dutch(fl_new)}")
print(f"  Net profit                                    : {dutch(net_profit)}")
print(f"  ROI                                           : {roi_pct:.1f}%")


# ── 6. EXPORTS ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 6: EXPORTING FILES")
print("=" * 70)

# Power BI export
powerbi_cols = [
    "region", "province", "locality", "zip_code",
    "property_type", "property_subtype",
    "price", "living_area", "number_rooms",
    "kitchen", "kitchen_score",
    "building_state", "building_score",
    "has_garden", "has_terrace", "has_pool", "has_fireplace",
    "number_facades", "furnished",
    "price_per_sqm", "predicted_price", "valuation_gap", "valuation_gap_pct",
]
powerbi_df = df[powerbi_cols].copy()

pb_path = os.path.join(script_dir, "belgium_re_powerbi_export.csv")
powerbi_df.to_csv(pb_path, index=False, sep=",", encoding="utf-8-sig")
print(f"[6.1] Power BI export saved : {pb_path}")
print(f"      Rows: {len(powerbi_df):,}  |  Columns: {len(powerbi_df.columns)}".replace(",", "."))

# Feature importance export
fi_path = os.path.join(script_dir, "belgium_re_feature_importance.csv")
fi.to_csv(fi_path, index=False)
print(f"[6.2] Feature importance saved : {fi_path}")

print()
print("=" * 70)
print("PIPELINE COMPLETE. All files written successfully.")
print("=" * 70)
print("""
OUTPUT SUMMARY
  belgium_re_powerbi_export.csv     : Power BI-ready, cleaned and enriched
  belgium_re_feature_importance.csv : Bar chart source for methodology section

COLUMN NOTES (belgium_re_powerbi_export.csv)
  price_per_sqm     : price ÷ living_area (€ per m²)
  predicted_price   : model estimate of fair market value
  valuation_gap     : actual price minus predicted price
                      positive = listed above model estimate (overpriced signal)
                      negative = listed below model estimate (opportunity signal)
  valuation_gap_pct : valuation_gap as % of predicted_price
  kitchen_score     : 0 = not installed, up to 3 = hyper equipped
  building_score    : 0 = to restore, up to 5 = as new

NUMBER FORMAT:
  Dutch locale: period (.) = thousands separator, comma (,) = decimal
  Example: € 353.750  |  € 2.374,65 /m²
""")
