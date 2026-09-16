"""Chatbot icin ek baglam katmani.

Uygulamada bugune kadar kullanilmayan verileri (model tablosundaki kampanya,
kategori, magaza, saat kolonlari; kampanya/kupon gecmisi; urun kirilimi;
demografi; portfoy ozeti) chatbot promptuna hazir sozluklere cevirir.

Streamlit import etmez: cache'leme src/data.py tarafinda yapilir.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TABLE_PATH = DATA / "processed" / "household_snapshot_model_table.parquet"
TX_PATH = DATA / "interim" / "transaction_data.parquet"
PRODUCT_PATH = DATA / "interim" / "product.parquet"
CAMPAIGN_TABLE_PATH = DATA / "interim" / "campaign_table.parquet"
CAMPAIGN_DESC_PATH = DATA / "interim" / "campaign_desc.parquet"
REDEMPT_PATH = DATA / "interim" / "coupon_redempt.parquet"
DEMO_PATH = DATA / "interim" / "hh_demographic.parquet"

CURRENT = 655
WINDOW_DAYS = 182

# Model tablosunda var olan ama uygulamaya hic tasinmayan kolonlar
EXTRA_COLUMNS: list[str] = [
    # kategori derinligi
    "unique_departments", "unique_commodities", "unique_subcommodities",
    "repeat_product_rate", "department_entropy", "top_department_revenue_share",
    "top3_department_revenue_share", "private_brand_revenue_share",
    "lost_departments_8w", "new_departments_8w", "department_retention_rate",
    # magaza ve saat
    "unique_stores", "dominant_store_share", "avg_transaction_hour",
    "morning_basket_rate", "afternoon_basket_rate", "evening_basket_rate", "night_basket_rate",
    # kampanya / kupon
    "completed_campaigns_received", "completed_campaigns_received_TypeA",
    "completed_campaigns_received_TypeB", "completed_campaigns_received_TypeC",
    "active_campaigns_at_snapshot", "campaigns_started_last_26w",
    "coupon_redemption_events_obs", "unique_coupons_redeemed_obs",
    "campaigns_with_redemption_obs", "days_since_last_campaign",
    "days_since_last_redemption", "bc_coupon_redemption_rate", "has_redeemed_coupon",
    # pencere karsilastirmalari
    "customer_spend_last_12w", "customer_spend_prev_8w", "baskets_prev_8w",
    "spend_change_4w", "recent_spend_share_8w", "max_purchase_gap", "gap_cv",
    # akaryakit
    "fuel_revenue_share", "has_fuel_purchase", "days_since_last_fuel_purchase",
]

DEMO_LABELS = {
    "classification_1": "Yas grubu (anonim kod)",
    "classification_2": "classification_2",
    "classification_3": "classification_3",
    "classification_4": "classification_4",
    "classification_5": "classification_5",
    "homeowner_desc": "Ev sahipligi",
    "kid_category_desc": "Cocuk durumu",
}


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100


# ---------------------------------------------------------------- 1) model tablosu ekstralari
def extra_features(snapshot_day: int = CURRENT) -> pd.DataFrame:
    """household_key index'li, EXTRA_COLUMNS kolonlu tablo (secili snapshot)."""
    if not _exists(TABLE_PATH):
        return pd.DataFrame()
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(TABLE_PATH).schema_arrow.names)
    cols = ["household_key", "snapshot_day"] + [c for c in EXTRA_COLUMNS if c in available]
    df = pd.read_parquet(TABLE_PATH, columns=cols)
    df = df[df["snapshot_day"] == snapshot_day].drop(columns=["snapshot_day"])
    return df.set_index(df["household_key"].astype(int)).drop(columns=["household_key"])


# ---------------------------------------------------------------- 2) demografi
def demographics() -> dict[int, dict]:
    if not _exists(DEMO_PATH):
        return {}
    df = pd.read_parquet(DEMO_PATH)
    return {int(k): v for k, v in df.set_index("household_key").to_dict("index").items()}


# ---------------------------------------------------------------- 3) kampanya ve kupon gecmisi
def campaign_history(household_key: int, snapshot_day: int = CURRENT) -> dict:
    """Snapshot gununden ONCE bilinen kampanya ve kupon kullanim gecmisi."""
    key = int(household_key)
    out: dict = {"campaigns": [], "redemptions": [], "active_now": []}
    if not (_exists(CAMPAIGN_TABLE_PATH) and _exists(CAMPAIGN_DESC_PATH)):
        return out
    ct = pd.read_parquet(CAMPAIGN_TABLE_PATH)
    ct = ct[ct["household_key"].astype(int) == key]
    cd = pd.read_parquet(CAMPAIGN_DESC_PATH)
    m = ct.merge(cd, on=["campaign", "description"], how="left")
    m = m[m["start_day"] <= snapshot_day].sort_values("start_day")
    out["campaigns"] = [
        {"campaign": int(r.campaign), "type": str(r.description),
         "start_day": int(r.start_day), "end_day": int(r.end_day),
         "active_at_snapshot": bool(r.start_day <= snapshot_day <= r.end_day)}
        for r in m.itertuples()
    ]
    out["active_now"] = [c for c in out["campaigns"] if c["active_at_snapshot"]]
    if _exists(REDEMPT_PATH):
        rd = pd.read_parquet(REDEMPT_PATH)
        rd = rd[(rd["household_key"].astype(int) == key) & (rd["day"] <= snapshot_day)]
        out["redemptions"] = [
            {"day": int(r.day), "campaign": int(r.campaign), "coupon_upc": str(r.coupon_upc)}
            for r in rd.sort_values("day").itertuples()
        ]
    return out


# ---------------------------------------------------------------- 4) urun / kategori kirilimi
def top_commodities(household_key: int, snapshot_day: int = CURRENT,
                    window_days: int = WINDOW_DAYS, top_n: int = 8) -> pd.DataFrame:
    """Gozlem penceresinde hanenin en cok harcadigi kategoriler (commodity_desc)."""
    if not (_exists(TX_PATH) and _exists(PRODUCT_PATH)):
        return pd.DataFrame()
    tx = pd.read_parquet(TX_PATH, filters=[("household_key", "==", int(household_key))])
    tx = tx[(tx["day"] > snapshot_day - window_days) & (tx["day"] <= snapshot_day)]
    if tx.empty:
        return pd.DataFrame()
    prod = pd.read_parquet(PRODUCT_PATH, columns=["product_id", "department", "commodity_desc", "brand"])
    m = tx.merge(prod, on="product_id", how="left")
    agg = (m.groupby("commodity_desc", observed=True)
             .agg(spend=("sales_value", "sum"), lines=("sales_value", "size"),
                  baskets=("basket_id", "nunique"))
             .sort_values("spend", ascending=False).head(top_n).reset_index())
    total = float(m["sales_value"].sum()) or 1.0
    agg["share"] = agg["spend"] / total
    return agg


def last_baskets(household_key: int, snapshot_day: int = CURRENT, n: int = 3) -> list[dict]:
    """Snapshot gununden onceki son n sepetin ozeti."""
    if not _exists(TX_PATH):
        return []
    tx = pd.read_parquet(TX_PATH, filters=[("household_key", "==", int(household_key))])
    tx = tx[tx["day"] <= snapshot_day]
    if tx.empty:
        return []
    g = (tx.groupby("basket_id")
           .agg(day=("day", "max"), spend=("sales_value", "sum"), lines=("product_id", "size"),
                store=("store_id", "max"), discount=("retail_disc", "sum"))
           .sort_values("day", ascending=False).head(n).reset_index())
    return [{"day": int(r.day), "days_ago": int(snapshot_day - r.day), "spend": float(r.spend),
             "lines": int(r.lines), "store": int(r.store), "discount": float(abs(r.discount))}
            for r in g.itertuples()]


# ---------------------------------------------------------------- 5) portfoy ozeti
def portfolio_summary(hh_table: pd.DataFrame, threshold: float) -> dict:
    """Chatbotun 'portfoyde durum ne' sorularini yanitlayabilmesi icin toplu ozet."""
    d = hh_table
    out = {
        "n_households": int(len(d)),
        "mean_probability": float(d["lapse_probability"].mean()),
        "median_probability": float(d["lapse_probability"].median()),
        "above_threshold": int((d["lapse_probability"] >= threshold).sum()),
        "top_k_count": int(d["selected_top_k"].sum()) if "selected_top_k" in d else None,
        "actual_inactive": int(d["target_inactive"].sum()) if "target_inactive" in d else None,
    }
    if "segment_name" in d.columns:
        g = d.groupby("segment_name").agg(
            hane=("household_key", "size"), ort_risk=("lapse_probability", "mean"),
            top10=("selected_top_k", "sum"), gercek_pasif=("target_inactive", "sum"))
        out["segments"] = g.reset_index().to_dict("records")
    top = d.nlargest(10, "lapse_probability")
    cols = [c for c in ["household_key", "segment_name", "lapse_probability"] if c in d.columns]
    out["top10_riskli"] = top[cols].to_dict("records")
    return out
