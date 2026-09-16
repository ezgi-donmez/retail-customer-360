"""Customer 360 uygulamasi icin veri yukleme fonksiyonlari (st.cache_data ile)."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PRED_PATH = DATA / "processed" / "test_inactivity_risk_predictions.parquet"
SEGMENTS_PATH = DATA / "processed" / "customer_segments.parquet"
PROFILES_PATH = DATA / "processed" / "segment_profiles.csv"
SHAP_PATH = DATA / "interim" / "shap_local_drivers.csv"
FEATURE_DICT_PATH = DATA / "interim" / "feature_dictionary.csv"
METRICS_PATH = DATA / "interim" / "test_metrics.csv"
IMPORTANCE_PATH = DATA / "interim" / "selected_feature_importance.csv"


def _read_csv_utf8(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1254"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    cols = [
        "household_key", "snapshot_day", "target_inactive", "lapse_probability",
        "predicted_inactive", "risk_rank", "risk_percentile", "selected_top_k",
    ]
    df = pd.read_parquet(PRED_PATH)
    return df[[c for c in cols if c in df.columns]].sort_values("risk_rank").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame | None:
    """Segment atamalari; dosya henuz uretilmediyse None."""
    if not SEGMENTS_PATH.exists():
        return None
    return pd.read_parquet(SEGMENTS_PATH)


@st.cache_data(show_spinner=False)
def load_segment_profiles() -> pd.DataFrame | None:
    if not PROFILES_PATH.exists():
        return None
    return _read_csv_utf8(PROFILES_PATH)


@st.cache_data(show_spinner=False)
def load_shap_drivers() -> pd.DataFrame:
    df = _read_csv_utf8(SHAP_PATH)
    cols = ["household_key", "driver_rank", "raw_feature", "raw_feature_value",
            "shap_contribution_model_output", "direction"]
    return df[[c for c in cols if c in df.columns]].sort_values(["household_key", "driver_rank"])


@st.cache_data(show_spinner=False)
def load_feature_dictionary() -> dict:
    """feature -> Turkce aciklama."""
    df = _read_csv_utf8(FEATURE_DICT_PATH)
    return dict(zip(df["feature"], df["description"]))


@st.cache_data(show_spinner=False)
def load_test_metrics() -> dict:
    df = _read_csv_utf8(METRICS_PATH)
    return dict(zip(df["metric"], df["test_value"]))


@st.cache_data(show_spinner=False)
def load_feature_importance(top_n: int = 8) -> pd.DataFrame:
    df = _read_csv_utf8(IMPORTANCE_PATH)
    return df.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_household_table() -> pd.DataFrame:
    """Tahminler + (varsa) segment/RFM sutunlari, risk sirasina gore."""
    from src.names import add_names
    pred = load_predictions()
    seg = load_segments()
    if seg is None:
        return add_names(pred)
    extra = [c for c in seg.columns if c not in pred.columns or c == "household_key"]
    return add_names(pred.merge(seg[extra], on="household_key", how="left"))


@st.cache_resource(show_spinner=False)
def load_bundle():
    from src.whatif import load_bundle as _lb
    return _lb()


@st.cache_data(show_spinner=False)
def load_model_features():
    from src.whatif import load_features
    return load_features(655)


@st.cache_resource(show_spinner=False)
def load_profile_context():
    """Profil sayfası için tüm hanelerin hesaplarını bir kez yapar (src/profile.py)."""
    from src.profile import build_context
    return build_context()


# ---------------------------------------------------------------- chatbot ek baglami
@st.cache_data(show_spinner=False)
def load_extra_features():
    """Model tablosundaki, arayuzde gosterilmeyen davranis kolonlari (household_key index)."""
    from src.context import extra_features
    return extra_features()


@st.cache_data(show_spinner=False)
def load_demographics() -> dict:
    from src.context import demographics
    return demographics()


@st.cache_data(show_spinner=False)
def load_campaign_history(household_key: int) -> dict:
    from src.context import campaign_history
    return campaign_history(int(household_key))


@st.cache_data(show_spinner=False)
def load_top_commodities(household_key: int):
    from src.context import top_commodities
    return top_commodities(int(household_key))


@st.cache_data(show_spinner=False)
def load_last_baskets(household_key: int) -> list:
    from src.context import last_baskets
    return last_baskets(int(household_key))


@st.cache_data(show_spinner=False)
def load_portfolio_summary(_hh_table, threshold: float) -> dict:
    """_hh_table hash'lenmez (alt cizgi ile); threshold cache anahtaridir."""
    from src.context import portfolio_summary
    return portfolio_summary(_hh_table, threshold)
