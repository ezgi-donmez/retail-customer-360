"""RFM tabanli K-Means musteri segmentasyonu (snapshot 655 ile tutarli)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TX_PATH = ROOT / "data/interim/transaction_data.parquet"
PRED_PATH = ROOT / "data/processed/test_inactivity_risk_predictions.parquet"
OUT_SEG = ROOT / "data/processed/customer_segments.parquet"
OUT_PROF = ROOT / "data/processed/segment_profiles.csv"

SNAPSHOT_DAY = 655
WINDOW_DAYS = 182
START_DAY = SNAPSHOT_DAY - WINDOW_DAYS + 1  # 474
FEATURES = ["recency_days", "frequency_baskets", "monetary", "avg_basket_value", "active_weeks", "unique_products"]
K_RANGE = range(3, 7)
RANDOM_STATE = 42


def build_rfm(households: pd.Series) -> pd.DataFrame:
    tx = pd.read_parquet(TX_PATH, columns=["household_key", "basket_id", "day", "sales_value", "coupon_disc", "product_id"])
    tx = tx[(tx["day"].between(START_DAY, SNAPSHOT_DAY)) & (tx["sales_value"] > 0) & tx["household_key"].isin(households)]
    tx["net_value"] = tx["sales_value"] + tx["coupon_disc"]
    tx["week"] = tx["day"] // 7
    g = tx.groupby("household_key")
    rfm = pd.DataFrame({
        "recency_days": SNAPSHOT_DAY - g["day"].max(),
        "frequency_baskets": g["basket_id"].nunique(),
        "monetary": g["net_value"].sum(),
        "active_weeks": g["week"].nunique(),
        "unique_products": g["product_id"].nunique(),
    })
    rfm["avg_basket_value"] = rfm["monetary"] / rfm["frequency_baskets"]
    # Pencerede alisverisi olmayan haneler: recency = pencere uzunlugu, digerleri 0
    rfm = rfm.reindex(households)
    rfm["recency_days"] = rfm["recency_days"].fillna(WINDOW_DAYS)
    rfm = rfm.fillna(0)
    rfm[["frequency_baskets", "active_weeks", "unique_products", "recency_days"]] = rfm[
        ["frequency_baskets", "active_weeks", "unique_products", "recency_days"]].astype(int)
    return rfm[FEATURES].reset_index()


def select_k(X: np.ndarray):
    scores = {}
    models = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
        models[k] = km
    best_k = max(scores, key=scores.get)
    return best_k, models[best_k], scores


def name_segments(rfm: pd.DataFrame, labels: np.ndarray, k: int):
    """Kume merkezlerine gore kural tabanli, benzersiz Turkce isimlendirme."""
    df = rfm.assign(cluster=labels)
    cent = df.groupby("cluster")[["recency_days", "frequency_baskets", "monetary"]].median()
    # Deger skoru: yuksek frequency/monetary ve dusuk recency -> yuksek
    rank = lambda s, asc: s.rank(ascending=asc, method="first")
    cent["score"] = rank(cent["frequency_baskets"], True) + rank(cent["monetary"], True) - rank(cent["recency_days"], True)
    order = cent.sort_values(["score", "monetary"], ascending=False).index.tolist()

    catalog = [
        ("Şampiyonlar", "En sık ve en yüksek harcamayla alışveriş yapan, yakın zamanda aktif çekirdek müşteriler."),
        ("Sadık Müşteriler", "Düzenli ve istikrarlı alışveriş yapan, sepet değeri güçlü kalıcı müşteriler."),
        ("Potansiyel Sadıklar", "Orta sıklıkta alışveriş yapan, doğru teşvikle sadık gruba taşınabilecek müşteriler."),
        ("Seyrek Alışverişçiler", "Düşük sıklık ve düşük harcamayla arada bir uğrayan müşteriler."),
        ("Uzaklaşanlar", "Son alışverişi uzun süre önce olan, aktifliğini kaybetmiş ve kaybedilme riski yüksek müşteriler."),
        ("Kaybedilmek Üzere Olanlar", "Neredeyse hiç alışveriş yapmayan, en yüksek pasiflik riskindeki müşteriler."),
    ]
    # k'ya gore katalogdan secim (deger sirasina gore, isimler benzersiz)
    k_to_idx = {3: [0, 1, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5]}
    chosen = [catalog[i] for i in k_to_idx[k]]

    mapping = {}
    for new_id, (cl, (name, desc)) in enumerate(zip(order, chosen)):
        mapping[cl] = (new_id, name, desc)
    return mapping


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    pred = pd.read_parquet(PRED_PATH)
    pred = pred[pred["snapshot_day"] == SNAPSHOT_DAY]
    households = pred["household_key"].drop_duplicates().reset_index(drop=True)
    print(f"Hane sayisi: {len(households)} | Pencere: gun {START_DAY}..{SNAPSHOT_DAY} ({WINDOW_DAYS} gun)")

    rfm = build_rfm(households)
    X = StandardScaler().fit_transform(np.log1p(rfm[FEATURES].values))

    best_k, km, scores = select_k(X)
    print("\nSilhouette skorlari:")
    for k, s in scores.items():
        print(f"  k={k}: {s:.4f}{'  <- secildi' if k == best_k else ''}")

    mapping = name_segments(rfm, km.labels_, best_k)
    rfm["segment_id"] = [mapping[c][0] for c in km.labels_]
    rfm["segment_name"] = [mapping[c][1] for c in km.labels_]
    desc_map = {v[0]: v[2] for v in mapping.values()}

    seg = rfm.merge(pred[["household_key", "lapse_probability", "risk_rank", "risk_percentile", "selected_top_k", "target_inactive"]],
                    on="household_key", how="left")

    g = seg.groupby(["segment_id", "segment_name"])
    prof = pd.DataFrame({
        "n_households": g.size(),
        "share_pct": g.size() / len(seg) * 100,
        "median_recency_days": g["recency_days"].median(),
        "median_frequency_baskets": g["frequency_baskets"].median(),
        "median_monetary": g["monetary"].median(),
        "median_avg_basket_value": g["avg_basket_value"].median(),
        "mean_lapse_probability": g["lapse_probability"].mean(),
        "top10_risk_share_pct": g["selected_top_k"].mean() * 100,
        "actual_inactive_rate_pct": g["target_inactive"].mean() * 100,
    }).reset_index().sort_values("segment_id")
    prof.insert(2, "description", prof["segment_id"].map(desc_map))
    prof = prof.round(3)

    OUT_SEG.parent.mkdir(parents=True, exist_ok=True)
    seg.drop(columns=["target_inactive"])[
        ["household_key", *FEATURES, "segment_id", "segment_name",
         "lapse_probability", "risk_rank", "risk_percentile", "selected_top_k"]
    ].to_parquet(OUT_SEG, index=False)
    prof.to_csv(OUT_PROF, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nSecilen k = {best_k}\n")
    print(prof.drop(columns=["description"]).to_string(index=False))
    print(f"\nYazildi: {OUT_SEG}\n         {OUT_PROF}")


if __name__ == "__main__":
    main()
