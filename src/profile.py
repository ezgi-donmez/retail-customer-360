"""Customer 360 profil katmani: tek bir hane icin karta hazir veri sozlugu.

Streamlit import etmez. Agir hesaplar build_context() icinde bir kez yapilir
(Streamlit tarafi @st.cache_resource ile saklar); household_profile() yalnizca
bellekteki dizilerden okur. Hicbir dosya yazilmaz.

Model aciklamasi: pipeline = StandardScaler + LogisticRegression oldugu icin
dogrusal katki = coef * (x_t - mu), mu = egitim snapshotlarinin (431, 487, 543)
donusturulmus ozellik ortalamasi (log-odds olceginde).
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import whatif
from src.names import name_for

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLE_PATH = DATA / "processed" / "household_snapshot_model_table.parquet"
PRED_PATH = DATA / "processed" / "test_inactivity_risk_predictions.parquet"
SEGMENTS_PATH = DATA / "processed" / "customer_segments.parquet"
PROFILES_PATH = DATA / "processed" / "segment_profiles.csv"
DEMO_PATH = DATA / "interim" / "hh_demographic.parquet"


@lru_cache(maxsize=1)
def _bundle() -> dict:
    return whatif.load_bundle()


THRESHOLD: float = float(_bundle()["decision_threshold"])
SNAPSHOTS = [431, 487, 543, 599, 655]
CURRENT = 655
HORIZON_DAYS = 56

FEATURE_LABELS: dict[str, str] = {
    "recency_days": "Son alışverişten beri geçen gün",
    "frequency_baskets": "Sepet sayısı (182 gün)",
    "customer_spend_total": "Toplam harcama (182 gün)",
    "active_weeks": "Aktif hafta sayısı",
    "spend_per_active_week": "Aktif hafta başına harcama",
    "baskets_per_active_week": "Aktif hafta başına sepet",
    "unique_products": "Farklı ürün sayısı",
    "unique_stores": "Farklı mağaza sayısı",
    "discount_dependency": "İndirim bağımlılığı",
    "discounted_basket_rate": "İndirimli sepet oranı",
    "coupon_basket_rate": "Kuponlu sepet oranı",
    "median_basket_value": "Tipik sepet tutarı",
    "basket_value_cv": "Sepet tutarı dalgalanması",
    "avg_products_per_basket": "Sepet başına ürün sayısı",
    "single_product_basket_rate": "Tek ürünlü sepet oranı",
    "dominant_store_share": "Ana mağaza payı",
    "median_purchase_gap": "Tipik alışveriş aralığı",
    "last_purchase_gap": "Son alışveriş aralığı",
    "recent_mean_purchase_gap_3": "Son 3 alışveriş aralığı ortalaması",
    "gap_acceleration": "Son aralığın tipik aralıktan farkı",
    "recency_to_typical_gap": "Alışveriş arasının normale göre uzaması",
    "gap_trend_last_5": "Son 5 aralığın eğilimi",
    "customer_spend_last_4w": "Son 4 haftadaki harcama",
    "baskets_last_4w": "Son 4 haftadaki sepet sayısı",
    "active_weeks_last_4w": "Son 4 haftadaki aktif hafta",
    "customer_spend_last_8w": "Son 8 haftadaki harcama",
    "baskets_last_8w": "Son 8 haftadaki sepet sayısı",
    "active_weeks_last_8w": "Son 8 haftadaki aktif hafta",
    "discounted_basket_rate_last_8w": "Son 8 haftada indirimli sepet oranı",
    "coupon_basket_rate_last_8w": "Son 8 haftada kuponlu sepet oranı",
    "spend_change_8w": "Harcama değişimi (8 hafta)",
    "basket_change_8w": "Sepet sayısı değişimi (8 hafta)",
    "avg_basket_value_change_8w": "Sepet tutarı değişimi (8 hafta)",
    "spend_slope_12w": "Harcama eğilimi (12 hafta)",
    "basket_slope_12w": "Sepet sayısı eğilimi (12 hafta)",
    "weekly_spend_cv_12w": "Haftalık harcama dalgalanması (12 hafta)",
    "weekly_basket_cv_12w": "Haftalık sepet dalgalanması (12 hafta)",
}

DEPT_LABELS: dict[str, str] = {
    "grocery": "Grocery",
    "drug_gm": "Drug GM",
    "kiosk_gas": "Akaryakıt",
    "produce": "Produce",
    "meat": "Meat",
    "meat_pckgd": "Paketli et",
    "deli": "Deli",
    "misc_sales_tran": "Diğer satış",
}
DEPT_COLS = [f"dept_revenue_share_{k}" for k in DEPT_LABELS]

HOMEOWNER_TR = {
    "Homeowner": "Ev sahibi",
    "Renter": "Kiracı",
    "Probable Owner": "Muhtemelen ev sahibi",
    "Probable Renter": "Muhtemelen kiracı",
}
KID_TR = {"1": "1 çocuk", "2": "2 çocuk", "3+": "3 ve üzeri çocuk"}
UNKNOWN_VALUES = {"Unknown", "None/Unknown", "", "nan", "None"}

EXTRA_COLS = ["private_brand_revenue_share", "has_demographic"]


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1254"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _tr_suffix(n: int) -> str:
    """Sayi icin 3. tekil iyelik eki: 65 -> "i", 66 -> "sı", 30 -> "u"."""
    ones = {1: "i", 2: "si", 3: "ü", 4: "ü", 5: "i", 6: "sı", 7: "si", 8: "i", 9: "u"}
    tens = {10: "u", 20: "si", 30: "u", 40: "ı", 50: "si", 60: "ı", 70: "i", 80: "i", 90: "ı"}
    n = abs(int(n))
    if n == 0:
        return "ı"
    if n % 10:
        return ones[n % 10]
    if n % 100:
        return tens[n % 100]
    return "ü"


def _pct_short(x: float) -> str:
    v = float(x) * 100
    if v >= 10:
        return f"%{v:.0f}"
    return "%" + f"{v:.1f}".replace(".", ",")


def _initials(name: str) -> str:
    return "".join(p[0] for p in name.split() if p)[:2]


# ---------------------------------------------------------------------------
# Baglam
# ---------------------------------------------------------------------------
def build_context() -> dict:
    bundle = _bundle()
    fn = list(bundle["feature_names"])
    pipe = bundle["pipeline"]
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    coef = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])
    train_days = [int(d) for d in bundle.get("train_snapshot_days", [431, 487, 543])]

    cols = ["household_key", "snapshot_day"] + fn + DEPT_COLS + EXTRA_COLS
    panel = pd.read_parquet(TABLE_PATH, columns=list(dict.fromkeys(cols)))
    panel = panel.sort_values(["household_key", "snapshot_day"]).reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_t = np.asarray(pre.transform(panel[fn]), dtype=float)
    panel["probability"] = 1.0 / (1.0 + np.exp(-(X_t @ coef + intercept)))

    train_mask = panel["snapshot_day"].isin(train_days).to_numpy()
    mu = X_t[train_mask].mean(axis=0)

    cur_mask = (panel["snapshot_day"] == CURRENT).to_numpy()
    cur_pos = np.flatnonzero(cur_mask)
    cur = panel.iloc[cur_pos].reset_index(drop=True)
    contrib = (X_t[cur_mask] - mu) * coef  # (n_hane, 37) log-odds
    cur_keys = cur["household_key"].astype(int).to_numpy()

    pred = pd.read_parquet(PRED_PATH)
    pred = pred[pred["snapshot_day"] == CURRENT].reset_index(drop=True)

    seg = pd.read_parquet(SEGMENTS_PATH) if SEGMENTS_PATH.exists() else None
    seg_map = dict(zip(seg["household_key"].astype(int), seg["segment_name"])) if seg is not None else {}
    seg_desc: dict[str, str] = {}
    if PROFILES_PATH.exists():
        prof = _read_csv(PROFILES_PATH)
        seg_desc = dict(zip(prof["segment_name"], prof["description"]))

    if DEMO_PATH.exists():
        demo = pd.read_parquet(DEMO_PATH)
    else:
        demo = pd.DataFrame(columns=["household_key"])
    demo_map = {int(k): v for k, v in demo.set_index("household_key").to_dict("index").items()}

    cur_seg = [seg_map.get(int(k)) for k in cur_keys]
    promo_avg = (
        pd.DataFrame({"segment": cur_seg, "promo": cur["discounted_basket_rate"].to_numpy(dtype=float)})
        .dropna(subset=["segment"]).groupby("segment")["promo"].mean().to_dict()
    )

    spend_sorted = np.sort(cur["customer_spend_total"].fillna(0).to_numpy(dtype=float))
    coupon = cur["coupon_basket_rate"].fillna(0).to_numpy(dtype=float)
    pos_coupon = coupon[coupon > 0]
    coupon_q = tuple(float(q) for q in np.quantile(pos_coupon, [1 / 3, 2 / 3])) if len(pos_coupon) else (0.0, 0.0)

    dept_pop = cur[DEPT_COLS].fillna(0).mean().sort_values(ascending=False)
    dept_popularity = [c.replace("dept_revenue_share_", "") for c in dept_pop.index]

    n_cur = len(cur_keys)
    n_demo_cur = int(sum(1 for k in cur_keys if int(k) in demo_map))
    demo_missing_share = (1 - n_demo_cur / n_cur) if n_cur else 0.0

    rows = {int(k): np.asarray(v) for k, v in panel.groupby("household_key", sort=False).indices.items()}
    arrays = {c: panel[c].to_numpy() for c in panel.columns}

    return {
        "bundle": bundle,
        "threshold": float(bundle["decision_threshold"]),
        "feature_names": fn,
        "coef": coef,
        "intercept": intercept,
        "background_mean": pd.Series(mu, index=fn),
        "panel": panel,
        "current": cur,
        "predictions": pred,
        "segments": seg,
        "segment_descriptions": seg_desc,
        "segment_promo_avg": promo_avg,
        "demographic": demo,
        "contributions": pd.DataFrame(contrib, index=pd.Index(cur_keys, name="household_key"), columns=fn),
        "spend_sorted": spend_sorted,
        "coupon_terciles": coupon_q,
        "dept_popularity": dept_popularity,
        "demographic_missing_share": demo_missing_share,
        # hizli erisim
        "_rows": rows,
        "_arr": arrays,
        "_cur_pos": {int(k): int(p) for k, p in zip(cur_keys, cur_pos)},
        "_contrib_arr": contrib,
        "_contrib_row": {int(k): i for i, k in enumerate(cur_keys)},
        "_pred_row": {int(k): i for i, k in enumerate(pred["household_key"].astype(int).to_numpy())},
        "_pred_arr": {c: pred[c].to_numpy() for c in pred.columns},
        "_seg_map": seg_map,
        "_demo_map": demo_map,
    }


# ---------------------------------------------------------------------------
# Profil parcalari
# ---------------------------------------------------------------------------
def _departments(arr: dict, pos: int) -> list[dict]:
    shares = []
    for key in DEPT_LABELS:
        v = _num(arr[f"dept_revenue_share_{key}"][pos]) or 0.0
        shares.append((key, min(max(v, 0.0), 1.0)))
    top = [s for s in sorted(shares, key=lambda s: -s[1]) if s[1] > 0][:3]
    out = [{"key": k, "name": DEPT_LABELS[k], "share": float(v)} for k, v in top]
    rest = min(max(1.0 - sum(v for _, v in top), 0.0), 1.0)
    out.append({"key": "other", "name": "Diğer", "share": float(rest)})
    return out


def _drivers(ctx: dict, key: int, pos: int, above: bool) -> list[dict]:
    fn = ctx["feature_names"]
    c = ctx["_contrib_arr"][ctx["_contrib_row"][key]]
    order_pos = [int(i) for i in np.argsort(-c) if c[i] > 0]
    order_neg = [int(i) for i in np.argsort(c) if c[i] < 0]
    n_pos, n_neg = (2, 1) if above else (1, 2)
    chosen = order_pos[:n_pos] + order_neg[:n_neg]
    if len(chosen) < 3:
        for i in np.argsort(-np.abs(c)):
            if len(chosen) >= 3:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
    chosen = sorted(chosen, key=lambda i: -abs(c[i]))
    mx = max((abs(c[i]) for i in chosen), default=0.0)
    arr = ctx["_arr"]
    out = []
    for i in chosen:
        f = fn[i]
        val = float(c[i])
        out.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "raw_value": _num(arr[f][pos]),
            "contribution": val,
            "direction": "up" if val > 0 else "down",
            "strength": float(abs(val) / mx) if mx > 0 else 0.0,
        })
    return out


def _whatif_best(ctx: dict, pos: int, base: float) -> dict | None:
    try:
        row = ctx["panel"].iloc[pos]
        sc = whatif.scenarios(row)
        if not sc:
            return None
        fn = ctx["feature_names"]
        Xs = pd.DataFrame([r[fn].to_numpy(dtype=float) for _, _, r in sc], columns=fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = ctx["bundle"]["pipeline"].predict_proba(Xs)[:, 1]
    except Exception:
        return None
    i = int(np.argmin(ps))
    return {"scenario": sc[i][0], "from": float(base), "to": float(ps[i]),
            "below_threshold": bool(ps[i] < ctx["threshold"])}


def _actions(ctx: dict, level: str, segment: str | None, top_k: bool, above: bool,
             spend_pct: float, promo_vs: str | None, depts: list[dict], whatif_best: dict | None) -> list[dict]:
    names = [d["name"] for d in depts if d["key"] != "other"][:2]
    if level == "Yüksek" and segment == "Uzaklaşanlar":
        title = "Geri kazanım kampanyası"
        if len(names) == 2:
            text = f"Sık satın aldığı {names[0]} ve {names[1]} ürünlerinde kişiselleştirilmiş teklif."
        elif names:
            text = f"Sık satın aldığı {names[0]} ürünlerinde kişiselleştirilmiş teklif."
        else:
            text = "Sık satın aldığı ürünlerde kişiselleştirilmiş teklif."
        tone = "warning"
    elif level in ("Yüksek", "Orta"):
        title = "Erken uyarı: kişisel hatırlatma"
        if len(names) == 2:
            text = f"{names[0]} ve {names[1]} odaklı hatırlatma; alışveriş aralığı açılmadan temas."
        elif names:
            text = f"{names[0]} odaklı hatırlatma; alışveriş aralığı açılmadan temas."
        else:
            text = "Kişisel hatırlatma; alışveriş aralığı açılmadan temas."
        tone = "warning"
    elif segment == "Şampiyonlar":
        title = "Sadakat ödülü"
        text = "İndirim gerekmez; teşekkür ve sadakat puanı yeterli."
        tone = "positive"
    else:
        title = "Sepet büyütme"
        cross = list(names)
        for k in ctx["dept_popularity"]:
            if len(cross) >= 2:
                break
            if DEPT_LABELS[k] not in cross:
                cross.append(DEPT_LABELS[k])
        text = f"{cross[0]} alışverişine {cross[1]} kategorisinden çapraz öneri."
        tone = "positive"
    if promo_vs == "üstünde":
        text += " Promosyona duyarlı: kupon işe yarayabilir."
    elif promo_vs == "altında":
        text += " Promosyona düşük ilgi: indirim yerine kişisel öneri."
    actions = [{"title": title, "text": text, "tone": tone}]

    high_value = spend_pct >= 70
    if top_k:
        prio, tone2 = "yüksek", "warning"
        text2 = ("Yüksek risk ve yüksek geçmiş değer birlikte değerlendirildi." if high_value
                 else "En riskli %10 içinde; ilk temas listesine alınmalı.")
    elif above:
        prio, tone2 = "orta", "neutral"
        text2 = ("Risk karar eşiğinin üzerinde ve geçmiş değer yüksek." if high_value
                 else "Risk karar eşiğinin üzerinde ama en riskli %10 dışında.")
    elif high_value:
        prio, tone2 = "orta", "neutral"
        text2 = "Risk düşük ama geçmiş harcama portföyün üst %30'unda; planlı temas yeterli."
    else:
        prio, tone2 = "düşük", "positive"
        text2 = "Risk düşük; rutin iletişim yeterli."
    actions.append({"title": f"Temas önceliği: {prio}", "text": text2, "tone": tone2})

    if whatif_best is not None:
        actions.append({
            "title": f"Hedef davranış: {whatif_best['scenario']}",
            "text": f"Model simülasyonu: olasılık {_pct_short(whatif_best['from'])} → "
                    f"{_pct_short(whatif_best['to'])}. Nedensel değildir.",
            "tone": "neutral",
        })
    return actions


def _clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _profile_rows(ctx: dict, key: int, name: str, segment: str | None, pos: int, actual: int) -> list[list[str]]:
    arr = ctx["_arr"]
    rows: list[list[str]] = []
    demo = ctx["_demo_map"].get(key)
    if demo is not None:
        ho = _clean(demo.get("homeowner_desc"))
        kid = _clean(demo.get("kid_category_desc"))
        rows.append(["Ev sahipliği", "Bilinmiyor" if ho in UNKNOWN_VALUES else HOMEOWNER_TR.get(ho, ho)])
        rows.append(["Çocuk durumu", "Bilinmiyor" if kid in UNKNOWN_VALUES else KID_TR.get(kid, kid)])
        codes = [c for c in (_clean(demo.get(f"classification_{i}")) for i in range(1, 6)) if c]
        if codes:
            rows.append(["Demografi kodları (anonim)", " · ".join(codes)])
    else:
        share = int(round(ctx["demographic_missing_share"] * 100))
        rows.append(["Demografi", f"Kayıt yok (hanelerin %{share}'{_tr_suffix(share)})"])

    coupon = _num(arr["coupon_basket_rate"][pos]) or 0.0
    q1, q2 = ctx["coupon_terciles"]
    coupon_lvl = "Yok" if coupon <= 0 else ("Düşük" if coupon <= q1 else ("Orta" if coupon <= q2 else "Yüksek"))
    rows.append(["Kupon kullanımı", coupon_lvl])
    rows.append(["Kuponlu sepet oranı", f"%{coupon * 100:.0f}"])
    return rows


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------
def household_profile(ctx: dict, household_key: int, include_whatif: bool = True) -> dict:
    key = int(household_key)
    if key not in ctx["_pred_row"] or key not in ctx["_cur_pos"]:
        raise KeyError(household_key)
    thr = ctx["threshold"]
    arr = ctx["_arr"]
    pa = ctx["_pred_arr"]
    pr = ctx["_pred_row"][key]
    pos = ctx["_cur_pos"][key]
    name = name_for(key)
    segment = ctx["_seg_map"].get(key)

    prob = float(pa["lapse_probability"][pr])
    above = prob >= thr
    top_k = bool(int(pa["selected_top_k"][pr]) == 1)
    level = "Yüksek" if top_k else ("Orta" if above else "Düşük")
    actual = int(pa["target_inactive"][pr])

    hh_pos = ctx["_rows"][key]
    recency = _num(arr["recency_days"][pos])
    prev = [int(p) for p in hh_pos if int(arr["snapshot_day"][p]) < CURRENT]
    recency_delta = None
    if prev and recency is not None:
        prev_rec = _num(arr["recency_days"][prev[-1]])
        if prev_rec is not None:
            recency_delta = int(round(recency - prev_rec))
    spend = _num(arr["customer_spend_total"][pos]) or 0.0
    ss = ctx["spend_sorted"]
    spend_pct = float(np.searchsorted(ss, spend, side="right") / len(ss) * 100) if len(ss) else 0.0
    promo = _num(arr["discounted_basket_rate"][pos]) or 0.0
    promo_avg = ctx["segment_promo_avg"].get(segment) if segment else None
    promo_vs = None
    if promo_avg is not None:
        diff = promo - promo_avg
        promo_vs = "benzer" if abs(diff) < 0.02 else ("üstünde" if diff > 0 else "altında")

    trend = [{
        "snapshot_day": int(arr["snapshot_day"][p]),
        "spend": float(_num(arr["customer_spend_total"][p]) or 0.0),
        "visits": int(_num(arr["frequency_baskets"][p]) or 0),
        "active_weeks": int(_num(arr["active_weeks"][p]) or 0),
        "probability": float(arr["probability"][p]),
    } for p in hh_pos]

    depts = _departments(arr, pos)
    drivers = _drivers(ctx, key, pos, above)
    wb = _whatif_best(ctx, pos, prob) if include_whatif else None
    actions = _actions(ctx, level, segment, top_k, above, spend_pct, promo_vs, depts, wb)

    return {
        "household_key": key,
        "customer_name": name,
        "initials": f"H{str(key)[:2]}",
        "snapshot_day": CURRENT,
        "horizon_days": HORIZON_DAYS,
        "segment_name": segment,
        "segment_description": ctx["segment_descriptions"].get(segment) if segment else None,
        "risk": {
            "probability": prob,
            "threshold": float(thr),
            "above_threshold": bool(above),
            "level": level,
            "percentile": float(pa["risk_percentile"][pr]) * 100,
            "rank": int(pa["risk_rank"][pr]),
            "n": int(len(pa["household_key"])),
            "top_k": top_k,
            "actual_inactive": actual,
        },
        "has_demographic": key in ctx["_demo_map"],
        "kpis": {
            "recency_days": int(round(recency)) if recency is not None else 0,
            "recency_delta": recency_delta,
            "spend_182d": float(spend),
            "spend_percentile": spend_pct,
            "visits_182d": int(_num(arr["frequency_baskets"][pos]) or 0),
            "active_weeks": int(_num(arr["active_weeks"][pos]) or 0),
            "promo_share": float(promo),
            "promo_segment_avg": float(promo_avg) if promo_avg is not None else None,
            "promo_vs_segment": promo_vs,
        },
        "trend": trend,
        "departments": depts,
        "drivers": drivers,
        "actions": actions,
        "profile_rows": _profile_rows(ctx, key, name, segment, pos, actual),
        "whatif_best": wb,
    }
