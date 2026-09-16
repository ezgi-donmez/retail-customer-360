"""Chatbot veri araçları (OpenAI function calling).

Model, sistem promptundaki seçili hane bağlamının dışındaki verilere bu araçlarla erişir.
Araçlar SALT OKUNURDUR: kod çalıştırma, dosya okuma veya yazma yoktur. Her parametre
doğrulanır, liste sonuçları sınırlandırılır ve çıktı JSON olarak modele geri verilir.
Arayüzde gösterilen sayılarla tutarlılık için tüm hane metrikleri src.profile'dan gelir.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src import profile as P
from src import whatif as W

MAX_ROWS = 50
MAX_RESULT_CHARS = 14000
RISK_LEVELS = ["Yüksek", "Orta", "Düşük"]
SORT_FIELDS = ["probability", "spend_182d", "recency_days", "visits_182d", "active_weeks", "promo_share"]
GROUP_FIELDS = ["none", "segment", "risk_level", "has_demographic", "top_department", "coupon_level"]
LIMITATIONS = [
    "Pozitif sınıf oranı yalnızca %6-7; metrikler güven aralığıyla okunmalı.",
    "Aynı haneler farklı snapshot'larda tekrar eder; yeni hanelere genelleme test edilmedi.",
    "Test snapshot'ı (655) önceki revizyonlarda görüldü; tamamen bağımsız bir holdout değildir.",
    "Demografi hanelerin yalnızca bir kısmında var ve daha aktif hanelerde yoğunlaşır.",
    "Model etkenleri ve senaryolar tahmin açıklamasıdır; nedensel etki değildir.",
    "Kampanya/aksiyon önerileri kural tabanlı hipotezdir; A/B testiyle doğrulanmadı.",
]


# --------------------------------------------------------------------------- yardımcılar
def _py(v: Any) -> Any:
    """numpy/pandas değerlerini JSON uyumlu Python tiplerine çevirir; float'ları yuvarlar."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _int(v: Any, default: int | None = None, lo: int | None = None, hi: int | None = None) -> int | None:
    try:
        x = int(v)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _norm(s: Any) -> str:
    return str(s or "").strip().casefold()


# --------------------------------------------------------------------------- araç kutusu
class ToolBox:
    def __init__(self, ctx: dict, seg_profiles: pd.DataFrame | None, metrics: dict,
                 importance: pd.DataFrame | None, bundle: dict | None = None,
                 features: pd.DataFrame | None = None):
        self.ctx = ctx
        self.seg_profiles = seg_profiles
        self.metrics = metrics or {}
        self.importance = importance
        self.bundle = bundle
        self.features = features
        self.threshold = float(ctx.get("threshold", P.THRESHOLD))
        self.table = self._build_table()
        self.segments = sorted(self.table["segment"].dropna().unique().tolist())

    # -- tüm haneler için arayüzle aynı metrikleri içeren tablo
    def _build_table(self) -> pd.DataFrame:
        keys = sorted(set(self.ctx["_pred_row"]) & set(self.ctx["_cur_pos"]))
        rows = []
        for k in keys:
            p = P.household_profile(self.ctx, k, include_whatif=False)
            r, kp = p["risk"], p["kpis"]
            coupon = next((v for lab, v in p["profile_rows"] if lab == "Kupon kullanımı"), None)
            rows.append({
                "household_key": int(k),
                "segment": p.get("segment_name"),
                "probability": r["probability"],
                "risk_rank": r["rank"],
                "risk_level": r["level"],
                "top_k": r["top_k"],
                "actual_inactive": r["actual_inactive"],
                "recency_days": kp["recency_days"],
                "recency_delta": kp["recency_delta"],
                "spend_182d": kp["spend_182d"],
                "spend_percentile": kp["spend_percentile"],
                "visits_182d": kp["visits_182d"],
                "active_weeks": kp["active_weeks"],
                "promo_share": kp["promo_share"],
                "has_demographic": p["has_demographic"],
                "top_department": p["departments"][0]["name"] if p.get("departments") else None,
                "coupon_level": coupon,
            })
        return pd.DataFrame(rows)

    # -- OpenAI araç şemaları
    def schemas(self) -> list[dict]:
        seg_enum = self.segments or ["Şampiyonlar", "Sadık Müşteriler", "Uzaklaşanlar"]
        filters = {
            "segment": {"type": "string", "enum": seg_enum, "description": "Segment adı"},
            "risk_level": {"type": "string", "enum": RISK_LEVELS,
                           "description": "Yüksek = en riskli %10; Orta = eşik üstü ama ilk %10 dışı; Düşük = eşik altı"},
            "top_k_only": {"type": "boolean", "description": "Yalnızca en riskli %10 (kampanya hedefi) haneler"},
            "has_demographic": {"type": "boolean", "description": "Demografi kaydı olan/olmayan haneler"},
            "actual_inactive": {"type": "boolean", "description": "Test döneminde gerçekten pasifleşen (true) / aktif kalan (false)"},
            "min_probability": {"type": "number", "description": "En düşük inaktif olma olasılığı (0-1)"},
            "max_probability": {"type": "number", "description": "En yüksek inaktif olma olasılığı (0-1)"},
            "min_recency_days": {"type": "integer", "description": "Son alışverişten beri en az gün"},
            "max_recency_days": {"type": "integer", "description": "Son alışverişten beri en fazla gün"},
            "min_spend": {"type": "number", "description": "182 günlük harcama alt sınırı ($)"},
            "max_spend": {"type": "number", "description": "182 günlük harcama üst sınırı ($)"},
        }
        return [
            {"type": "function", "function": {
                "name": "get_household",
                "description": "Tek bir hanenin tam profilini getirir: risk olasılığı/seviyesi/sırası, KPI'lar (son alışveriş, 182 günlük harcama, ziyaret, aktif hafta, promosyon payı), snapshot trendi, departman payları, modelin en etkili 3 etkeni, kural tabanlı aksiyonlar, kupon ve demografi, en iyi senaryo.",
                "parameters": {"type": "object", "properties": {
                    "household_key": {"type": "integer", "description": "Hane numarası, ör. 1108"}},
                    "required": ["household_key"]}}},
            {"type": "function", "function": {
                "name": "find_households",
                "description": "Haneleri filtreleyip sıralar. Ör. 'Uzaklaşanlar segmentinde riski en yüksek 5 hane', 'harcaması 500$ üstü olup riski yüksek haneler'. Sonuç en fazla 50 satırdır; total_matches toplam eşleşmeyi verir.",
                "parameters": {"type": "object", "properties": {
                    **filters,
                    "sort_by": {"type": "string", "enum": SORT_FIELDS, "description": "Sıralama alanı (varsayılan probability)"},
                    "descending": {"type": "boolean", "description": "Büyükten küçüğe (varsayılan true)"},
                    "limit": {"type": "integer", "description": "Döndürülecek satır sayısı, 1-50 (varsayılan 10)"}}}}},
            {"type": "function", "function": {
                "name": "summarize_portfolio",
                "description": "Portföyü (isteğe bağlı filtrelerle) gruplayıp özetler: hane sayısı, pay, ortalama risk, en riskli %10'daki hane sayısı, gerçek pasifleşme oranı, medyan son alışveriş/harcama/ziyaret, ortalama promosyon payı.",
                "parameters": {"type": "object", "properties": {
                    "group_by": {"type": "string", "enum": GROUP_FIELDS, "description": "Gruplama alanı (varsayılan none)"},
                    **filters}}}},
            {"type": "function", "function": {
                "name": "compare_households",
                "description": "2-6 haneyi temel metrikleriyle yan yana karşılaştırır.",
                "parameters": {"type": "object", "properties": {
                    "household_keys": {"type": "array", "items": {"type": "integer"}, "description": "Hane numaraları (2-6 adet)"}},
                    "required": ["household_keys"]}}},
            {"type": "function", "function": {
                "name": "run_whatif",
                "description": "Bir hane için senaryo analizi: 'bu hafta 1 alışveriş', '4 hafta haftalık ritim' gibi davranışlarda modelin olasılığı nasıl değişir. Model simülasyonudur, nedensel değildir.",
                "parameters": {"type": "object", "properties": {
                    "household_key": {"type": "integer", "description": "Hane numarası"}},
                    "required": ["household_key"]}}},
            {"type": "function", "function": {
                "name": "get_model_info",
                "description": "Model bilgisi: test metrikleri (ROC-AUC, AP, recall, precision, lift), karar eşiği, en önemli özellikler ve sınırlılıklar.",
                "parameters": {"type": "object", "properties": {
                    "top_n_features": {"type": "integer", "description": "Gösterilecek özellik sayısı, 1-15 (varsayılan 8)"}}}}},
            {"type": "function", "function": {
                "name": "get_segment_profiles",
                "description": "Segment profilleri tablosu: hane sayısı, pay, medyan son alışveriş/sepet/harcama, ortalama risk, en riskli %10 payı, gerçek pasifleşme oranı ve segment açıklamaları.",
                "parameters": {"type": "object", "properties": {}}}},
        ]

    # -- çalıştırma
    def execute(self, name: str, args: dict | None) -> dict:
        args = args if isinstance(args, dict) else {}
        fn = {
            "get_household": self.get_household,
            "find_households": self.find_households,
            "summarize_portfolio": self.summarize_portfolio,
            "compare_households": self.compare_households,
            "run_whatif": self.run_whatif,
            "get_model_info": self.get_model_info,
            "get_segment_profiles": self.get_segment_profiles,
        }.get(name)
        if fn is None:
            return {"error": f"Bilinmeyen araç: {name}"}
        try:
            return _py(fn(**args))
        except TypeError as exc:
            return {"error": f"Geçersiz parametre: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Araç hatası: {exc}"}

    @staticmethod
    def to_json(result: dict) -> str:
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + '..."} [KESİLDİ: sonuç çok uzun, daha dar filtre veya küçük limit kullan]'
        return text

    @staticmethod
    def summarize(result: dict) -> str:
        if "error" in result:
            return f"hata: {result['error']}"
        if "rows" in result:
            return f"{result.get('returned', len(result['rows']))}/{result.get('total_matches', '?')} satır"
        if "groups" in result:
            return f"{len(result['groups'])} grup"
        if "household_key" in result:
            return f"hane #{result['household_key']}"
        if "households" in result:
            return f"{len(result['households'])} hane"
        if "scenarios" in result:
            return f"{len(result['scenarios'])} senaryo"
        return "tamam"

    # -- filtre
    def _filter(self, segment=None, risk_level=None, top_k_only=None, has_demographic=None, actual_inactive=None,
                min_probability=None, max_probability=None, min_recency_days=None, max_recency_days=None,
                min_spend=None, max_spend=None) -> tuple[pd.DataFrame, dict]:
        t = self.table
        used: dict = {}
        if segment:
            match = [s for s in self.segments if _norm(s) == _norm(segment)]
            if not match:
                raise ValueError(f"Segment bulunamadı: {segment}. Geçerli: {', '.join(self.segments)}")
            t = t[t["segment"] == match[0]]
            used["segment"] = match[0]
        if risk_level:
            match = [s for s in RISK_LEVELS if _norm(s) == _norm(risk_level)]
            if not match:
                raise ValueError(f"Risk seviyesi geçersiz: {risk_level}. Geçerli: {', '.join(RISK_LEVELS)}")
            t = t[t["risk_level"] == match[0]]
            used["risk_level"] = match[0]
        for col, val in (("top_k", top_k_only), ("has_demographic", has_demographic)):
            if val is not None:
                t = t[t[col] == bool(val)] if (col != "top_k" or val) else t
                used[col] = bool(val)
        if actual_inactive is not None:
            t = t[t["actual_inactive"] == (1 if actual_inactive else 0)]
            used["actual_inactive"] = bool(actual_inactive)
        for col, lo, hi in (("probability", min_probability, max_probability),
                            ("recency_days", min_recency_days, max_recency_days),
                            ("spend_182d", min_spend, max_spend)):
            lo, hi = _float(lo), _float(hi)
            if lo is not None:
                t = t[t[col] >= lo]
                used[f"min_{col}"] = lo
            if hi is not None:
                t = t[t[col] <= hi]
                used[f"max_{col}"] = hi
        return t, used

    @staticmethod
    def _row(r: pd.Series) -> dict:
        return {
            "household_key": int(r["household_key"]), "segment": r["segment"],
            "probability": r["probability"], "risk_rank": int(r["risk_rank"]), "risk_level": r["risk_level"],
            "top_k": bool(r["top_k"]), "recency_days": r["recency_days"], "spend_182d": r["spend_182d"],
            "visits_182d": r["visits_182d"], "active_weeks": r["active_weeks"], "promo_share": r["promo_share"],
            "top_department": r["top_department"], "has_demographic": bool(r["has_demographic"]),
            "actual_inactive": int(r["actual_inactive"]),
        }

    # -- araçlar
    def get_household(self, household_key) -> dict:
        key = _int(household_key)
        if key is None or key not in self.ctx["_pred_row"] or key not in self.ctx["_cur_pos"]:
            return {"error": f"Hane bulunamadı: {household_key}. Test snapshot'ında 2.284 hane var."}
        p = P.household_profile(self.ctx, key)
        return {
            "household_key": key,
            "segment": p.get("segment_name"),
            "segment_description": p.get("segment_description"),
            "risk": p["risk"],
            "kpis": p["kpis"],
            "trend": p["trend"],
            "departments": [{"name": d["name"], "share": d["share"]} for d in p["departments"]],
            "drivers": [{"label": d["label"], "raw_value": d["raw_value"], "direction": d["direction"],
                         "contribution_log_odds": d["contribution"]} for d in p["drivers"]],
            "actions": p["actions"],
            "coupon_and_demographics": {lab: val for lab, val in p["profile_rows"]},
            "whatif_best": p.get("whatif_best"),
            "note": "Etkenler doğrusal model katkısıdır (nedensel değil); aksiyonlar kural tabanlı hipotezdir.",
        }

    def find_households(self, sort_by: str = "probability", descending: bool = True, limit: int = 10, **filters) -> dict:
        t, used = self._filter(**filters)
        sort_by = sort_by if sort_by in SORT_FIELDS else "probability"
        limit = _int(limit, 10, 1, MAX_ROWS)
        t = t.sort_values(sort_by, ascending=not bool(descending), kind="mergesort")
        return {"filters": used, "sort_by": sort_by, "descending": bool(descending),
                "total_matches": int(len(t)), "returned": int(min(limit, len(t))),
                "rows": [self._row(r) for _, r in t.head(limit).iterrows()]}

    def summarize_portfolio(self, group_by: str = "none", **filters) -> dict:
        t, used = self._filter(**filters)
        group_by = group_by if group_by in GROUP_FIELDS else "none"
        n_all = len(self.table)

        def agg(g: pd.DataFrame) -> dict:
            return {"households": int(len(g)), "share_of_portfolio_pct": 100 * len(g) / n_all if n_all else None,
                    "mean_probability": g["probability"].mean(), "top_k_households": int(g["top_k"].sum()),
                    "actual_inactive_rate_pct": 100 * g["actual_inactive"].mean() if len(g) else None,
                    "median_recency_days": g["recency_days"].median(), "median_spend_182d": g["spend_182d"].median(),
                    "median_visits_182d": g["visits_182d"].median(), "mean_promo_share": g["promo_share"].mean()}

        if group_by == "none" or t.empty:
            groups = [{"group": "tümü", **agg(t)}]
        else:
            col = {"segment": "segment", "risk_level": "risk_level", "has_demographic": "has_demographic",
                   "top_department": "top_department", "coupon_level": "coupon_level"}[group_by]
            def label(k):
                if col == "has_demographic":
                    return "Demografi var" if bool(k) else "Demografi yok"
                return "Bilinmiyor" if k is None or (isinstance(k, float) and math.isnan(k)) else str(k)
            groups = [{"group": label(k), **agg(g)} for k, g in t.groupby(col, dropna=False)]
            groups.sort(key=lambda x: -x["households"])
        return {"filters": used, "group_by": group_by, "total_households": int(len(t)), "groups": groups}

    def compare_households(self, household_keys) -> dict:
        keys = [k for k in (_int(x) for x in (household_keys or [])) if k is not None][:6]
        if len(keys) < 1:
            return {"error": "En az bir hane numarası verin."}
        by_key = {int(r["household_key"]): r for _, r in self.table.iterrows() if int(r["household_key"]) in keys}
        found = [self._row(by_key[k]) for k in keys if k in by_key]
        missing = [k for k in keys if k not in by_key]
        return {"households": found, "missing": missing}

    def run_whatif(self, household_key) -> dict:
        key = _int(household_key)
        if self.bundle is None or self.features is None:
            return {"error": "Senaryo analizi için özellik tablosu yüklenemedi."}
        df = W.run(self.bundle, self.features, key, self.threshold) if key is not None else None
        if df is None:
            return {"error": f"Hane bulunamadı: {household_key}"}
        return {"household_key": key, "threshold": self.threshold,
                "scenarios": [{"scenario": r["Senaryo"], "description": r["Açıklama"], "probability": r["Olasılık"],
                               "change": r["Değişim"], "below_threshold": bool(r["Eşik altı"])}
                              for _, r in df.iterrows()],
                "note": "Model simülasyonu; nedensel etki değildir, A/B testiyle doğrulanmalıdır."}

    def get_model_info(self, top_n_features: int = 8) -> dict:
        n = _int(top_n_features, 8, 1, 15)
        keep = ["roc_auc", "average_precision", "brier_score", "positive_rate", "recall", "precision", "f2",
                "top_k_rate", "top_k_count", "recall_at_top_k", "precision_at_top_k", "lift_at_top_k"]
        feats = []
        if self.importance is not None:
            for _, r in self.importance.head(n).iterrows():
                feats.append({"feature": r["feature"], "label": P.FEATURE_LABELS.get(r["feature"], r["feature"]),
                              "importance_mean": r["importance_mean"]})
        return {"model": "Logistic Regression (37 davranışsal özellik)", "target": "Snapshot sonrası 56 günde hiç geçerli alışveriş yok",
                "test_snapshot_day": P.CURRENT, "decision_threshold": self.threshold,
                "test_metrics": {k: self.metrics.get(k) for k in keep if k in self.metrics},
                "top_features_permutation_importance": feats, "limitations": LIMITATIONS}

    def get_segment_profiles(self) -> dict:
        if self.seg_profiles is None:
            return {"error": "Segment profilleri bulunamadı."}
        return {"segments": self.seg_profiles.drop(columns=["segment_id"], errors="ignore").to_dict(orient="records")}
