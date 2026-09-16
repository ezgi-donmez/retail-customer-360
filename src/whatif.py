"""What-if (senaryo) analizi: riskli bir hanenin davranisi degisseydi model ne derdi?

Model tabanli bir simulasyondur, nedensel kanit degildir. Secili hanenin 37 ozellikli
vektoru alinir, birkac 'aksiyon' senaryosunda ilgili ozellikler degistirilir ve ayni
pipeline ile olasilik yeniden hesaplanir.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "processed" / "inactivity_risk_model_bundle.joblib"
TABLE = ROOT / "data" / "processed" / "household_snapshot_model_table.parquet"


def load_bundle() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(BUNDLE)


def load_features(snapshot_day: int = 655) -> pd.DataFrame | None:
    """Test snapshot'indaki hane x 37 ozellik tablosu; dosya yoksa None."""
    if not TABLE.exists() or TABLE.stat().st_size < 100:
        return None
    df = pd.read_parquet(TABLE)
    return df[df["snapshot_day"] == snapshot_day].reset_index(drop=True)


def predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    feats = bundle["feature_names"]
    return bundle["pipeline"].predict_proba(X[feats])[:, 1]


def _apply_basket(row: pd.Series, n_baskets: int, basket_value: float, days_ago: int) -> pd.Series:
    """Haneye 'son days_ago gun icinde n_baskets sepet daha yapti' senaryosunu uygular."""
    r = row.copy()
    r["recency_days"] = min(r["recency_days"], days_ago)
    r["frequency_baskets"] += n_baskets
    r["customer_spend_total"] += n_baskets * basket_value
    new_weeks = min(n_baskets, 4)
    r["active_weeks"] += new_weeks
    r["active_weeks_last_4w"] = min(r["active_weeks_last_4w"] + new_weeks, 4)
    r["active_weeks_last_8w"] = min(r["active_weeks_last_8w"] + new_weeks, 8)
    r["baskets_last_4w"] += n_baskets
    r["baskets_last_8w"] += n_baskets
    r["customer_spend_last_4w"] += n_baskets * basket_value
    r["customer_spend_last_8w"] += n_baskets * basket_value
    r["spend_per_active_week"] = r["customer_spend_total"] / max(r["active_weeks"], 1)
    r["baskets_per_active_week"] = r["frequency_baskets"] / max(r["active_weeks"], 1)
    # Ritim: son bosluklar kisalir
    typical = r["median_purchase_gap"] if pd.notna(r["median_purchase_gap"]) and r["median_purchase_gap"] > 0 else 14
    r["last_purchase_gap"] = min(r["last_purchase_gap"], typical) if pd.notna(r["last_purchase_gap"]) else typical
    r["recent_mean_purchase_gap_3"] = min(r["recent_mean_purchase_gap_3"], typical) if pd.notna(r["recent_mean_purchase_gap_3"]) else typical
    r["recency_to_typical_gap"] = r["recency_days"] / max(typical, 1)
    r["gap_acceleration"] = min(r["gap_acceleration"], 1.0) if pd.notna(r["gap_acceleration"]) else 1.0
    return r


def scenarios(row: pd.Series) -> list[tuple[str, str, pd.Series]]:
    """(ad, aciklama, degistirilmis satir) listesi. Sepet degeri: hanenin medyan sepeti."""
    bv = float(row["median_basket_value"]) if pd.notna(row["median_basket_value"]) and row["median_basket_value"] > 0 else 15.0
    out = [
        ("Bu hafta 1 alışveriş", "Kupon/hatırlatma ile bu hafta 1 sepet (medyan sepet tutarında)", _apply_basket(row, 1, bv, 3)),
        ("2 hafta üst üste alışveriş", "Son 4 haftada 2 sepet, 2 farklı haftada", _apply_basket(row, 2, bv, 3)),
        ("Haftalık ritme dönüş", "Son 4 haftada 4 sepet, her hafta 1", _apply_basket(row, 4, bv, 3)),
    ]
    # Indirim/kupon kullanimi senaryosu
    r = _apply_basket(row, 1, bv, 3)
    r["discounted_basket_rate"] = max(r["discounted_basket_rate"], 0.5)
    r["discounted_basket_rate_last_8w"] = max(r["discounted_basket_rate_last_8w"], 0.5)
    r["coupon_basket_rate"] = max(r["coupon_basket_rate"], 0.2)
    r["coupon_basket_rate_last_8w"] = max(r["coupon_basket_rate_last_8w"], 0.2)
    out.append(("1 alışveriş + kupon kullanımı", "Bu hafta 1 sepet ve kuponlu/indirimli alışveriş", r))
    return out


def run(bundle: dict, feats: pd.DataFrame, household_key: int, threshold: float) -> pd.DataFrame | None:
    sel = feats[feats["household_key"] == household_key]
    if sel.empty:
        return None
    row = sel.iloc[0]
    base = float(predict(bundle, sel)[0])
    rows = [{"Senaryo": "Mevcut durum", "Açıklama": "Değişiklik yok", "Olasılık": base, "Değişim": 0.0,
             "Eşik altı": base < threshold}]
    for name, desc, r in scenarios(row):
        p = float(predict(bundle, r.to_frame().T)[0])
        rows.append({"Senaryo": name, "Açıklama": desc, "Olasılık": p, "Değişim": p - base, "Eşik altı": p < threshold})
    return pd.DataFrame(rows)
