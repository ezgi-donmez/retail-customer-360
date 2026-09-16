"""Müşteri 360 arayüz bileşenleri: koyu tema CSS'i ve HTML kart üreticileri.

Streamlit import ETMEZ; yalnızca HTML/CSS string üretir. Kullanım:
    st.html(ui.theme_css())          # bir kez
    st.html(ui.header_html(profile)) # her kart için

Önemli: st.html içeriği DOMPurify ile yalnızca HTML profiliyle temizlenir;
satır içi <svg> etiketleri silinir. Bu yüzden grafik şekilleri (çizgi, alan,
halka) base64 data-URI <img> içindeki SVG olarak, metinler/etiketler/noktalar
ve ipuçları (title) ise HTML katmanı olarak üretilir. JavaScript kullanılmaz.
Tüm dinamik metinler html.escape ile kaçırılır.
"""
from __future__ import annotations

import base64
import html
import math
from typing import Any, Iterable

PALETTE: dict[str, str] = {
    "bg": "#111318",
    "card": "#1A1C23",
    "border": "#2A2D36",
    "text": "#E8E8EE",
    "muted": "#9CA0AA",
    "faint": "#858995",
    "purple": "#8B7CF6",
    "pink": "#F47C8C",
    "green": "#5CC9A0",
    "amber": "#F2B45A",
    "rail": "#272A33",
    "inner": "#20232C",
    "sidebar": "#15171D",
}

# Rozet/vurgu yazıları için açık tonlar (koyu zemin üzerinde okunurluk)
_LIGHT: dict[str, str] = {
    PALETTE["purple"]: "#BBB2FB",
    PALETTE["pink"]: "#F8A9B4",
    PALETTE["green"]: "#8EDDBF",
    PALETTE["amber"]: "#F6CB8C",
    PALETTE["muted"]: "#B3B7C0",
}

SEGMENT_COLORS: dict[str, str] = {
    "Şampiyonlar": PALETTE["green"],
    "Sadık Müşteriler": PALETTE["purple"],
    "Uzaklaşanlar": PALETTE["pink"],
}

RISK_COLORS: dict[str, str] = {
    "Yüksek": PALETTE["pink"],
    "Orta": PALETTE["amber"],
    "Düşük": PALETTE["green"],
}

_TONE_COLORS = {"warning": PALETTE["pink"], "positive": PALETTE["green"], "neutral": PALETTE["purple"]}

_FONT = "Inter,'Segoe UI',system-ui,-apple-system,Roboto,'Helvetica Neue',Arial,sans-serif"


# --------------------------------------------------------------------------- yardımcılar
def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _round_half_up(v: float, digits: int = 0) -> float:
    q = 10 ** digits
    return math.copysign(math.floor(abs(v) * q + 0.5) / q, v)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _light(color: str) -> str:
    return _LIGHT.get(color, color)


def _dec(v: float, digits: int) -> str:
    s = f"{abs(_round_half_up(v, digits)):,.{digits}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"-{s}" if v < 0 and s.strip("0,.") else s


def _svg_img(svg: str, cls: str, alt: str = "") -> str:
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="{cls}" src="data:image/svg+xml;base64,{data}" alt="{_e(alt)}">'


def _pct_css(v: float) -> str:
    return f"{max(0.0, min(100.0, v)):.2f}%"


# --------------------------------------------------------------------------- biçimlendirme
def fmt_int(x: Any) -> str:
    """1284 -> '1.284'."""
    v = _num(x)
    if v is None:
        return "—"
    n = int(_round_half_up(v))
    s = f"{abs(n):,}".replace(",", ".")
    return f"-{s}" if n < 0 else s


def fmt_money(x: Any) -> str:
    """1284 -> '$1.284'."""
    v = _num(x)
    if v is None:
        return "—"
    n = int(_round_half_up(v))
    return f"-${fmt_int(-n)}" if n < 0 else f"${fmt_int(n)}"


def fmt_pct(x01: Any, digits: int = 0) -> str:
    """0.42 -> '%42' (girdi 0-1)."""
    v = _num(x01)
    if v is None:
        return "—"
    return fmt_pct100(v * 100, digits)


def fmt_pct100(x: Any, digits: int = 0) -> str:
    """91 -> '%91' (girdi 0-100)."""
    v = _num(x)
    if v is None:
        return "—"
    body = fmt_int(abs(v)) if digits <= 0 else _dec(abs(v), digits)
    neg = _round_half_up(v, max(digits, 0)) < 0
    return f"-%{body}" if neg else f"%{body}"


def fmt_delta_days(d: Any) -> str:
    """18 -> '+18 gün', -3 -> '-3 gün', 0 -> 'değişmedi'."""
    v = _num(d)
    if v is None:
        return "—"
    n = int(_round_half_up(v))
    if n == 0:
        return "değişmedi"
    return f"+{fmt_int(n)} gün" if n > 0 else f"-{fmt_int(-n)} gün"


_MONEY_FEATURES = {
    "customer_spend_total", "spend_per_active_week", "median_basket_value",
    "customer_spend_last_4w", "customer_spend_last_8w",
}


def _short_value(feature: str, x: Any) -> str | None:
    """Model özelliğinin ham değerini kısa gösterir."""
    v = _num(x)
    if v is None:
        return None
    f = (feature or "").lower()
    if f in _MONEY_FEATURES:
        return fmt_money(v)
    if any(k in f for k in ("rate", "share", "dependency")):
        return fmt_pct(v)
    if any(k in f for k in ("change", "slope", "_cv", "acceleration", "trend", "typical_gap")):
        return _dec(v, 2)
    if f == "recency_days" or f.endswith("_gap") or "purchase_gap" in f:
        return f"{fmt_int(v)} gün"
    if abs(v - round(v)) < 1e-9:
        return fmt_int(v)
    return _dec(v, 1)


# --------------------------------------------------------------------------- tema
def theme_css() -> str:
    P = PALETTE
    app = '[data-testid="stApp"]'
    css = f"""
:root{{color-scheme:dark}}
.stApp[data-testid="stApp"],{app} [data-testid="stAppViewContainer"],{app} [data-testid="stMain"]{{background:{P['bg']};color:{P['text']}}}
{app} [data-testid="stHeader"]{{background:{_rgba(P['bg'], .92)};border-bottom:1px solid {_rgba(P['border'], .6)}}}
{app} [data-testid="stHeader"] button,{app} [data-testid="stToolbar"] button{{color:{P['muted']}}}
{app} [data-testid="stSidebar"]{{background:{P['sidebar']};border-right:1px solid {P['border']}}}
{app} [data-testid="stSidebarContent"]{{background:{P['sidebar']};color:{P['text']}}}
{app} [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{{color:{P['text']}}}
{app} [data-testid="stSidebarNavLink"] span{{color:{P['muted']}}}
{app} [data-testid="stMainBlockContainer"]{{padding-top:4.75rem;padding-bottom:3rem}}
{app} [data-testid="stVerticalBlock"]{{gap:.75rem}}
{app} [data-testid="stHorizontalBlock"]{{gap:.75rem}}
{app} h1,{app} h2,{app} h3,{app} h4,{app} h5,{app} h6{{color:{P['text']}}}
{app} [data-testid="stMarkdownContainer"],{app} [data-testid="stMarkdownContainer"] p,{app} [data-testid="stMarkdownContainer"] li{{color:{P['text']}}}
{app} [data-testid="stMarkdownContainer"] code{{background:{P['inner']};color:{_light(P['purple'])}}}
{app} [data-testid="stCaptionContainer"],{app} [data-testid="stCaptionContainer"] p{{color:{P['muted']}}}
{app} [data-testid="stWidgetLabel"],{app} [data-testid="stWidgetLabel"] p{{color:{P['muted']}}}
{app} a{{color:{_light(P['purple'])}}}
{app} hr{{border-color:{P['border']}}}
{app} [data-testid="stBaseButton-secondary"],{app} [data-testid="stBaseButton-tertiary"],{app} [data-testid="stBaseButton-secondaryFormSubmit"],{app} [data-testid="stBaseButton-pills"]{{background:{P['card']};color:{P['text']};border:1px solid {P['border']};border-radius:8px}}
{app} [data-testid="stBaseButton-secondary"]:hover,{app} [data-testid="stBaseButton-tertiary"]:hover,{app} [data-testid="stBaseButton-secondaryFormSubmit"]:hover,{app} [data-testid="stBaseButton-pills"]:hover{{background:{P['inner']};border-color:{P['purple']};color:#FFFFFF}}
{app} [data-testid="stBaseButton-primary"],{app} [data-testid="stBaseButton-primaryFormSubmit"],{app} [data-testid="stBaseButton-pillsActive"]{{background:{P['purple']};border:1px solid {P['purple']};color:#FFFFFF;border-radius:8px}}
{app} [data-testid="stBaseButton-primary"]:hover,{app} [data-testid="stBaseButton-primaryFormSubmit"]:hover{{background:#7A6AF0;border-color:#7A6AF0}}
{app} [data-testid^="stBaseButton"] p{{color:inherit}}
{app} div[data-baseweb="select"]>div{{background:{P['card']};border-color:{P['border']};color:{P['text']}}}
{app} div[data-baseweb="select"] svg{{fill:{P['muted']}}}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li,div[data-baseweb="menu"]{{background:{P['card']};color:{P['text']}}}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{{background:{P['inner']}}}
{app} div[data-baseweb="input"],{app} div[data-baseweb="base-input"],{app} div[data-baseweb="textarea"]{{background:{P['card']};border-color:{P['border']}}}
{app} input,{app} textarea{{color:{P['text']};caret-color:{P['purple']}}}
{app} input::placeholder,{app} textarea::placeholder{{color:{P['faint']}}}
{app} [data-testid="stBottom"],{app} [data-testid="stBottom"]>div,{app} [data-testid="stBottomBlockContainer"]{{background:{P['bg']}}}
{app} [data-testid="stChatInput"]{{background:{P['card']};border:1px solid {P['border']};border-radius:12px}}
{app} [data-testid="stChatInput"]:focus-within{{border-color:{P['purple']}}}
{app} [data-testid="stChatInput"]>div{{background:transparent}}
{app} [data-testid="stChatInputTextArea"]{{background:transparent;color:{P['text']}}}
{app} [data-testid="stChatInputSubmitButton"]{{color:{P['purple']}}}
{app} [data-testid="stChatMessage"]{{background:{P['card']};border:1px solid {P['border']};border-radius:10px;padding:.75rem 1rem}}
{app} [data-testid="stExpander"] details{{background:{P['card']};border:1px solid {P['border']};border-radius:10px}}
{app} [data-testid="stExpander"] summary{{color:{P['text']}}}
{app} [data-testid="stExpander"] summary:hover{{color:{_light(P['purple'])}}}
{app} [data-testid="stDataFrame"],{app} [data-testid="stTable"]{{border:1px solid {P['border']};border-radius:10px;overflow:hidden}}
{app} [data-testid="stTable"] table{{color:{P['text']}}}
{app} [data-testid="stTable"] th,{app} [data-testid="stTable"] td{{border-color:{P['border']}}}
{app} [data-testid="stMetric"]{{background:{P['card']};border:1px solid {P['border']};border-radius:10px;padding:12px 14px}}
{app} [data-testid="stMetricLabel"],{app} [data-testid="stMetricLabel"] p{{color:{P['muted']}}}
{app} [data-testid="stMetricValue"]{{color:{P['text']}}}
{app} [data-testid="stTabs"] button p{{color:{P['muted']}}}
{app} [data-testid="stTabs"] button[aria-selected="true"] p{{color:{P['text']}}}
{app} [data-testid="stRadio"] label p,{app} [data-testid="stCheckbox"] label p{{color:{P['text']}}}
{app} [data-testid="stPlotlyChart"]{{background:{P['card']};border:1px solid {P['border']};border-radius:10px;padding:6px}}
{app} [data-testid="stHtml"]{{width:100%}}
{app} *{{scrollbar-color:#3A3D48 transparent}}
.c360{{font-family:{_FONT};color:{P['text']};font-size:13px;line-height:1.4;-webkit-font-smoothing:antialiased;min-width:0}}
.c360 *,.c360 *::before,.c360 *::after{{box-sizing:border-box}}
.c360-card{{background:{P['card']};border:1px solid {P['border']};border-radius:10px;padding:14px 16px;min-width:0}}
.c360-card+.c360-card{{margin-top:12px}}
.c360-card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:12px}}
.c360-card-head>div:first-child{{min-width:0}}
.c360-card-title{{font-size:15px;font-weight:700;color:{P['text']};line-height:1.3;letter-spacing:-.005em}}
.c360-card-sub{{font-size:12px;color:{P['muted']};margin-top:2px}}
.c360-card-right{{font-size:12px;color:{P['muted']};white-space:nowrap;padding-top:2px}}
.c360-muted{{color:{P['muted']}}}
.c360-empty{{color:{P['muted']};font-size:12.5px;padding:6px 0}}
.c360-badge{{display:inline-flex;align-items:center;white-space:nowrap;font-size:12px;font-weight:600;line-height:1;padding:6px 10px;border-radius:999px;border:1px solid transparent}}
.c360-header{{display:flex;align-items:center;flex-wrap:wrap;gap:12px 14px}}
.c360-avatar{{flex:0 0 auto;width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:{_rgba(P['purple'], .2)};color:{_light(P['purple'])};font-weight:700;font-size:15px;letter-spacing:.02em}}
.c360-id{{flex:1 1 200px;min-width:0}}
.c360-name{{font-size:19px;font-weight:700;line-height:1.25;color:{P['text']};letter-spacing:-.01em}}
.c360-name-sep{{color:{P['faint']};font-weight:500;padding:0 2px}}
.c360-name-key{{white-space:nowrap}}
.c360-meta{{font-size:12.5px;color:{P['muted']};margin-top:3px}}
.c360-badges{{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end;margin-left:auto}}
.c360-kpi-wrap{{container-type:inline-size;container-name:c360kpi}}
.c360-kpi-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.c360-kpi{{background:{P['card']};border:1px solid {P['border']};border-radius:10px;padding:12px 14px;min-width:0;display:flex;flex-direction:column}}
.c360-kpi-label{{font-size:12px;color:{P['muted']};line-height:1.3;cursor:help}}
.c360-kpi-value{{font-size:23px;font-weight:700;color:{P['text']};line-height:1.15;margin:6px 0 5px;letter-spacing:-.015em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.c360-kpi-unit{{font-size:.66em;font-weight:600;color:{P['text']};margin-left:3px;letter-spacing:0}}
.c360-kpi-sub{{font-size:11.5px;color:{P['muted']};line-height:1.35;margin-top:auto}}
@container c360kpi (max-width:700px){{.c360-kpi-grid{{gap:8px}}.c360-kpi{{padding:10px 11px}}.c360-kpi-label{{font-size:11px;min-height:2.6em}}.c360-kpi-value{{font-size:19px;margin:4px 0}}.c360-kpi-sub{{font-size:10.5px}}}}
@container c360kpi (max-width:430px){{.c360-kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.c360-kpi-label{{min-height:0}}}}
.c360-up{{color:{P['pink']}}}
.c360-down{{color:{P['green']}}}
.c360-chart{{position:relative;height:150px;margin:4px 0 0}}
.c360-plot{{position:absolute;left:44px;right:12px;top:8px;bottom:24px}}
.c360-plot-svg{{position:absolute;inset:0;width:100%;height:100%;display:block}}
.c360-ylab{{position:absolute;left:-44px;width:36px;text-align:right;transform:translateY(-50%);font-size:10.5px;color:{P['faint']};line-height:1;white-space:nowrap}}
.c360-xlab{{position:absolute;top:100%;margin-top:8px;transform:translateX(-50%);font-size:11px;color:{P['faint']};line-height:1;white-space:nowrap}}
.c360-dot{{position:absolute;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;border-radius:50%;background:{P['purple']};box-shadow:0 0 0 2px {P['card']};cursor:help}}
.c360-dot-last{{width:11px;height:11px;margin:-5.5px 0 0 -5.5px;background:#A396FA}}
.c360-chart-note{{position:absolute;right:0;top:0;font-size:11px;color:{P['muted']}}}
.c360-bars{{display:flex;flex-direction:column;gap:11px}}
.c360-bar-row{{display:grid;grid-template-columns:minmax(64px,34%) minmax(0,1fr) 40px;align-items:center;gap:10px}}
.c360-bar-name{{font-size:13px;color:{P['text']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.c360-bar-val{{font-size:12.5px;color:{P['muted']};text-align:right;font-variant-numeric:tabular-nums}}
.c360-rail{{position:relative;height:8px;border-radius:999px;background:{P['rail']};overflow:hidden}}
.c360-rail-thin{{height:6px}}
.c360-fill{{position:absolute;left:0;top:0;bottom:0;border-radius:999px}}
.c360-model{{display:flex;align-items:center;gap:14px}}
.c360-donut{{position:relative;flex:0 0 auto;width:78px;height:78px}}
.c360-donut-svg{{width:100%;height:100%;display:block}}
.c360-donut-val{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:{P['text']};letter-spacing:-.01em}}
.c360-model-txt{{min-width:0}}
.c360-model-title{{font-size:14.5px;font-weight:700;color:{P['text']};line-height:1.3}}
.c360-model-sub{{font-size:12px;color:{P['muted']};margin-top:3px;line-height:1.4}}
.c360-model-note{{font-size:11px;color:{P['faint']};margin-top:3px}}
.c360-sep{{height:1px;background:{P['border']};margin:14px 0 12px}}
.c360-caption{{font-size:11.5px;color:{P['faint']};margin-bottom:10px;letter-spacing:.01em}}
.c360-drivers{{display:flex;flex-direction:column;gap:12px}}
.c360-driver-top{{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:6px}}
.c360-driver-label{{font-size:13px;color:{P['text']};min-width:0;line-height:1.3}}
.c360-driver-raw{{color:{P['muted']};font-size:12px;white-space:nowrap}}
.c360-driver-dir{{font-size:12px;font-weight:600;white-space:nowrap}}
.c360-actions{{display:flex;flex-direction:column;gap:8px}}
.c360-action{{background:{P['inner']};border-radius:8px;border-left:3px solid {P['green']};padding:10px 12px}}
.c360-action-title{{font-size:13.5px;font-weight:700;color:#F2F2F6;line-height:1.3}}
.c360-action-text{{font-size:12.5px;color:{P['muted']};margin-top:3px;line-height:1.4}}
.c360-kv{{display:flex;flex-direction:column}}
.c360-kv-row{{display:flex;justify-content:space-between;align-items:baseline;gap:14px;padding:8px 0;border-bottom:1px solid {P['border']}}}
.c360-kv-row:first-child{{padding-top:0}}
.c360-kv-row:last-child{{border-bottom:0;padding-bottom:0}}
.c360-kv-k{{font-size:12.5px;color:{P['muted']};flex:0 0 auto;max-width:48%}}
.c360-kv-v{{font-size:13px;color:{P['text']};text-align:right;min-width:0;overflow-wrap:anywhere}}
.c360-section{{margin:6px 0 2px}}
.c360-section-title{{font-size:20px;font-weight:700;color:{P['text']};letter-spacing:-.01em;line-height:1.3}}
.c360-section-sub{{font-size:13px;color:{P['muted']};margin-top:2px}}
.c360-stat .c360-kpi-value{{font-size:24px}}
.c360-wi{{container-type:inline-size;container-name:c360wi}}
.c360-wi-row{{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1.3fr) 78px 86px;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid {P['border']}}}
.c360-wi-row:last-child{{border-bottom:0;padding-bottom:0}}
.c360-wi-head{{padding-top:0;font-size:11.5px;color:{P['faint']}}}
.c360-wi-base{{background:{_rgba(P['purple'], .06)};border-radius:8px;padding:10px;margin:0 -10px;border-bottom-color:transparent}}
.c360-wi-name{{font-size:13px;font-weight:600;color:{P['text']};line-height:1.3}}
.c360-wi-desc{{font-size:11.5px;color:{P['muted']};margin-top:2px;line-height:1.35}}
.c360-wi-prob{{display:flex;align-items:center;gap:8px;min-width:0}}
.c360-wi-prob .c360-rail{{flex:1 1 auto}}
.c360-wi-pv{{font-size:12.5px;font-weight:600;color:{P['text']};width:44px;text-align:right;font-variant-numeric:tabular-nums}}
.c360-wi-delta{{font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.c360-wi-status{{text-align:right}}
.c360-thr{{position:absolute;top:-3px;bottom:-3px;width:2px;margin-left:-1px;background:{P['text']};opacity:.55;border-radius:1px}}
.c360-rail-wrap{{position:relative;flex:1 1 auto;padding:3px 0}}
.c360-wi-note{{font-size:11.5px;color:{P['faint']};margin-top:10px}}
@container c360wi (max-width:560px){{.c360-wi-row{{grid-template-columns:minmax(0,1fr) auto;gap:6px 12px}}.c360-wi-row>.c360-wi-sc{{grid-column:1/-1}}.c360-wi-head{{display:none}}.c360-wi-status{{grid-column:1/-1;text-align:left}}}}
{app} [data-testid="stSelectbox"] [role="group"]{{background:{P['card']} !important;border:1px solid {P['border']} !important;border-radius:8px;transition:border-color .15s ease,box-shadow .15s ease}}
{app} [data-testid="stSelectbox"] [role="group"]:hover{{border-color:#3A3D48 !important}}
{app} [data-testid="stSelectbox"] [role="group"]:focus-within{{border-color:{P['purple']} !important;box-shadow:0 0 0 3px {_rgba(P['purple'], .18)}}}
{app} [data-testid="stSelectbox"] input{{background:transparent !important;color:{P['text']} !important;-webkit-text-fill-color:{P['text']} !important;caret-color:{P['purple']}}}
{app} [data-testid="stSelectbox"] svg{{color:{P['muted']} !important;fill:currentColor}}
div:has(> [role="listbox"]){{background:{P['inner']} !important;border:1px solid {P['border']} !important;border-radius:10px !important;box-shadow:0 12px 32px rgba(0,0,0,.45) !important}}
[role="listbox"]{{background:{P['inner']} !important;color:{P['text']} !important}}
[role="option"],[role="option"]>div{{background:transparent !important;color:{P['text']} !important;-webkit-text-fill-color:{P['text']} !important}}
[role="option"]:hover,[role="option"]:hover>div,[role="option"][aria-selected="true"],[role="option"][aria-selected="true"]>div{{background:{_rgba(P['purple'], .22)} !important;color:#FFFFFF !important;-webkit-text-fill-color:#FFFFFF !important}}
[role="option"] *{{color:inherit !important}}
{app} [data-testid="stButtonGroup"] button[role="radio"]{{background:{P['card']} !important;color:{P['muted']} !important;border-color:{P['border']} !important;transition:background .15s ease,color .15s ease,border-color .15s ease}}
{app} [data-testid="stButtonGroup"] button[role="radio"]:hover{{background:{P['inner']} !important;color:{P['text']} !important;border-color:#4A4D5A !important}}
{app} [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"]{{background:{_rgba(P['purple'], .28)} !important;color:#FFFFFF !important;border-color:{P['purple']} !important}}
{app} [data-testid="stButtonGroup"] button[role="radio"] *{{color:inherit !important}}
[data-testid="stTooltipContent"],div:has(> [data-testid="stTooltipContent"]){{background:{P['inner']} !important;color:{P['text']} !important;border-radius:8px}}
.c360-table-wrap{{overflow-x:auto}}
.c360-table{{width:100%;border-collapse:collapse;font-size:13px}}
.c360-table th{{color:{P['muted']};font-weight:600;font-size:12px;padding:8px 10px;border-bottom:1px solid {P['border']};white-space:nowrap}}
.c360-table td{{padding:9px 10px;border-bottom:1px solid {P['border']};color:{P['text']};white-space:nowrap}}
.c360-table tr:last-child td{{border-bottom:0}}
.c360-table td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.c360-table tbody tr:hover td{{background:{_rgba(P['purple'], .07)}}}
[class*="st-key-c360-col-"]{{height:100%}}
[class*="st-key-c360-col-"] [data-testid="stVerticalBlock"]{{height:100%;display:flex;flex-direction:column}}
[class*="st-key-c360-col-"] [data-testid="stElementContainer"]:last-child{{flex:1 1 auto;display:flex}}
[class*="st-key-c360-col-"] [data-testid="stElementContainer"]:last-child>div{{flex:1 1 auto;display:flex;width:100%}}
[class*="st-key-c360-col-"] [data-testid="stElementContainer"]:last-child .c360-card{{flex:1 1 auto;display:flex;flex-direction:column;width:100%}}
"""
    return "<style>" + "".join(line.strip() for line in css.splitlines()) + "</style>"


# --------------------------------------------------------------------------- genel kartlar
def _badge(text: str, color: str, title: str | None = None, gray: bool = False) -> str:
    if gray:
        style = f"background:{_rgba(PALETTE['muted'], .12)};color:{PALETTE['muted']};border-color:{_rgba(PALETTE['muted'], .2)}"
    else:
        style = f"background:{_rgba(color, .15)};color:{_light(color)};border-color:{_rgba(color, .3)}"
    t = f' title="{_e(title)}"' if title else ""
    return f'<span class="c360-badge" style="{style}"{t}>{_e(text)}</span>'


def card_html(title: str | None, body_html: str, right: str | None = None, sub: str | None = None) -> str:
    """Genel kart. body_html güvenilir HTML olmalıdır (bu modülün ürettiği)."""
    head = ""
    if title or right or sub:
        left = ""
        if title:
            left += f'<div class="c360-card-title">{_e(title)}</div>'
        if sub:
            left += f'<div class="c360-card-sub">{_e(sub)}</div>'
        r = f'<div class="c360-card-right">{_e(right)}</div>' if right else ""
        head = f'<div class="c360-card-head"><div>{left}</div>{r}</div>'
    return f'<div class="c360 c360-card">{head}{body_html}</div>'


def section_title_html(text: str, sub: str | None = None) -> str:
    s = f'<div class="c360-section-sub">{_e(sub)}</div>' if sub else ""
    return f'<div class="c360 c360-section"><div class="c360-section-title">{_e(text)}</div>{s}</div>'


def stat_card_html(label: str, value: Any, sub: str | None = None, accent: str | None = None) -> str:
    """Portföy özet kartı. accent: değer rengi (ör. PALETTE['pink'])."""
    style = f' style="color:{_e(accent)}"' if accent else ""
    s = f'<div class="c360-kpi-sub">{_e(sub)}</div>' if sub else ""
    return (f'<div class="c360 c360-kpi c360-stat"><div class="c360-kpi-label" style="cursor:default">{_e(label)}</div>'
            f'<div class="c360-kpi-value"{style}>{_e(value)}</div>{s}</div>')


# --------------------------------------------------------------------------- başlık + KPI
def header_html(p: dict) -> str:
    key = p.get("household_key")
    initials = (p.get("initials") or (f"H{str(key)[:2]}" if key is not None else "H")).upper()
    risk = p.get("risk") or {}
    level = risk.get("level")
    seg = p.get("segment_name")

    badges = []
    if seg:
        badges.append(_badge(f"Segment: {seg}", SEGMENT_COLORS.get(seg, PALETTE["purple"]),
                             title=p.get("segment_description") or None))
    if level:
        tip = f"İnaktif olma olasılığı {fmt_pct(risk.get('probability'))} · karar eşiği {fmt_pct(risk.get('threshold'), 1)}"
        badges.append(_badge(f"{level} inaktivite riski", RISK_COLORS.get(level, PALETTE["muted"]), title=tip))
    if p.get("has_demographic"):
        badges.append(_badge("Demografi mevcut", PALETTE["purple"], title="Hane için demografi kaydı var (anonim kodlar)"))
    else:
        badges.append(_badge("Demografi yok", PALETTE["muted"], title="Hane için demografi kaydı yok", gray=True))

    day = p.get("snapshot_day", 655)
    horizon = p.get("horizon_days", 56)
    key_txt = f"Hane #{key}" if key is not None else "Hane"
    return (
        '<div class="c360 c360-card c360-header">'
        f'<div class="c360-avatar" aria-hidden="true">{_e(initials[:3])}</div>'
        '<div class="c360-id">'
        f'<div class="c360-name"><span class="c360-name-key">{_e(key_txt)}</span></div>'
        f'<div class="c360-meta">Son gözlem: Gün {_e(day)} · Tahmin ufku: {_e(horizon)} gün</div>'
        '</div>'
        f'<div class="c360-badges">{"".join(badges)}</div>'
        '</div>'
    )


def _kpi(label: str, value_html: str, sub_html: str, tip: str) -> str:
    return (
        '<div class="c360-kpi">'
        f'<div class="c360-kpi-label" title="{_e(tip)}">{_e(label)}</div>'
        f'<div class="c360-kpi-value">{value_html}</div>'
        f'<div class="c360-kpi-sub">{sub_html}</div>'
        '</div>'
    )


def _val_unit(value: str, unit: str | None = None) -> str:
    u = f'<span class="c360-kpi-unit">{_e(unit)}</span>' if unit else ""
    return f"{_e(value)}{u}"


def kpi_row_html(p: dict) -> str:
    k = p.get("kpis") or {}

    # 1) Son alışverişten beri
    rec = k.get("recency_days")
    delta = _num(k.get("recency_delta"))
    if delta is None:
        rec_sub = "Önceki snapshot yok"
    else:
        cls = "c360-up" if delta > 0 else ("c360-down" if delta < 0 else "")
        d = _e(fmt_delta_days(delta))
        rec_sub = f"Önceki snapshot'a göre <span class=\"{cls}\">{d}</span>" if cls else f"Önceki snapshot'a göre {d}"
    rec_val = _val_unit(fmt_int(rec), "gün" if _num(rec) is not None else None)

    # 2) Harcama
    spend_sub = f"Hane yüzdelik dilimi: {_e(fmt_pct100(k.get('spend_percentile')))}"

    # 3) Sıklık
    visits = k.get("visits_182d")
    visit_val = _val_unit(fmt_int(visits), "ziyaret" if _num(visits) is not None else None)
    visit_sub = f"Aktif hafta: {_e(fmt_int(k.get('active_weeks')))}"

    # 4) Promosyon
    vs = k.get("promo_vs_segment")
    avg = _num(k.get("promo_segment_avg"))
    if vs == "üstünde":
        promo_sub = "Segment ortalamasının üstünde"
    elif vs == "altında":
        promo_sub = "Segment ortalamasının altında"
    elif vs == "benzer":
        promo_sub = "Segment ortalamasına benzer"
    else:
        promo_sub = "Segment bilgisi yok"
    promo_tip = "İndirimli ürün içeren sepetlerin oranı (son 182 gün)"
    if avg is not None:
        promo_tip += f" · segment ortalaması {fmt_pct(avg)}"

    cards = [
        _kpi("Son alışverişten beri", rec_val, rec_sub,
             f"Son alışverişten gözlem gününe (Gün {p.get('snapshot_day', 655)}) kadar geçen gün sayısı"),
        _kpi("182 günlük harcama", _e(fmt_money(k.get("spend_182d"))), spend_sub,
             "Son 182 gündeki toplam harcama; yüzdelik dilim aynı gündeki tüm haneler içinde"),
        _kpi("Alışveriş sıklığı", visit_val, visit_sub,
             "Son 182 gündeki alışveriş sepeti (ziyaret) sayısı ve alışveriş yapılan hafta sayısı"),
        _kpi("Promosyon payı", _e(fmt_pct(k.get("promo_share"))), _e(promo_sub), promo_tip),
    ]
    return f'<div class="c360 c360-kpi-wrap"><div class="c360-kpi-grid">{"".join(cards)}</div></div>'


# --------------------------------------------------------------------------- trend
_TREND_METRICS = {
    "spend": ("spend", "182 günlük harcama ($)", fmt_money),
    "visits": ("visits", "182 günlük ziyaret sayısı", fmt_int),
    "active_weeks": ("active_weeks", "Aktif hafta sayısı", fmt_int),
    "probability": ("probability", "İnaktif olma olasılığı (model)", fmt_pct),
}


def _nice_top(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    f = v / exp
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if f <= m + 1e-9:
            return m * exp
    return 10 * exp


def trend_card_html(p: dict, metric: str = "spend") -> str:
    field, sub, fmt = _TREND_METRICS.get(metric, _TREND_METRICS["spend"])
    pts = []
    for t in p.get("trend") or []:
        d, v = _num(t.get("snapshot_day")), _num(t.get(field))
        if d is not None and v is not None:
            pts.append((int(d), v))
    pts.sort()
    n = len(pts)
    title = "Davranış değişimi"
    if n == 0:
        return card_html(title, '<div class="c360-empty">Trend verisi yok.</div>', sub=sub)

    right = "Tek snapshot" if n == 1 else f"Son {n} snapshot"
    top = _nice_top(max(v for _, v in pts) * 1.08)
    if metric == "probability":
        top = min(1.0, top) if max(v for _, v in pts) <= 1 else top
    ticks = [0.0, top / 2, top]

    d0, d1 = pts[0][0], pts[-1][0]
    pad = 4.0

    def xp(d: int) -> float:
        if n == 1 or d1 == d0:
            return 50.0
        return pad + (d - d0) / (d1 - d0) * (100 - 2 * pad)

    def yp(v: float) -> float:
        return 100.0 - max(0.0, min(1.0, v / top)) * 100.0

    purple = PALETTE["purple"]
    grid = "".join(
        f'<line x1="0" y1="{yp(t):.2f}" x2="100" y2="{yp(t):.2f}" stroke="#FFFFFF" stroke-opacity="{0.10 if t == 0 else 0.06}" '
        f'stroke-width="1" vector-effect="non-scaling-stroke"/>'
        for t in ticks
    )
    shapes = ""
    if n >= 2:
        coords = [(xp(d), yp(v)) for d, v in pts]
        line = " ".join(f"{'M' if i == 0 else 'L'}{x:.2f} {y:.2f}" for i, (x, y) in enumerate(coords))
        area = f"{line} L{coords[-1][0]:.2f} 100 L{coords[0][0]:.2f} 100 Z"
        shapes = (
            '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{purple}" stop-opacity="0.38"/>'
            f'<stop offset="1" stop-color="{purple}" stop-opacity="0.03"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#g)"/>'
            f'<path d="{line}" fill="none" stroke="{purple}" stroke-width="2.2" stroke-linejoin="round" '
            'stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none">'
        f"{grid}{shapes}</svg>"
    )

    ylabs = "".join(f'<div class="c360-ylab" style="top:{yp(t):.2f}%">{_e(fmt(t))}</div>' for t in ticks)
    xlabs = "".join(f'<div class="c360-xlab" style="left:{xp(d):.2f}%">{_e(d)}</div>' for d, _ in pts)
    dots = "".join(
        f'<span class="c360-dot{" c360-dot-last" if i == n - 1 else ""}" style="left:{xp(d):.2f}%;top:{yp(v):.2f}%" '
        f'title="Gün {_e(d)}: {_e(fmt(v))}"></span>'
        for i, (d, v) in enumerate(pts)
    )
    alt = "; ".join(f"Gün {d}: {fmt(v)}" for d, v in pts)
    body = (
        '<div class="c360-chart">'
        f'<div class="c360-plot">{_svg_img(svg, "c360-plot-svg", alt)}{ylabs}{xlabs}{dots}</div>'
        '</div>'
    )
    return card_html(title, body, right=right, sub=sub)


# --------------------------------------------------------------------------- departmanlar
def departments_card_html(p: dict) -> str:
    rows = []
    for d in p.get("departments") or []:
        share = _num(d.get("share"))
        if share is None:
            continue
        share = max(0.0, min(1.0, share))
        is_other = d.get("key") == "other"
        color = _rgba(PALETTE["purple"], .45) if is_other else PALETTE["purple"]
        rows.append(
            '<div class="c360-bar-row">'
            f'<div class="c360-bar-name" title="{_e(d.get("name"))}">{_e(d.get("name"))}</div>'
            f'<div class="c360-rail"><div class="c360-fill" style="width:{_pct_css(share * 100)};background:{color}"></div></div>'
            f'<div class="c360-bar-val">{_e(fmt_pct(share))}</div>'
            '</div>'
        )
    body = f'<div class="c360-bars">{"".join(rows)}</div>' if rows else '<div class="c360-empty">Departman verisi yok.</div>'
    return card_html("Tercih edilen departmanlar", body, right="Harcama payı")


# --------------------------------------------------------------------------- model
def _donut_svg(prob: float, color: str) -> str:
    r, sw = 32.0, 9.0
    c = 2 * math.pi * r
    filled = max(0.0, min(1.0, prob)) * c
    arc = ""
    if filled > 0.01:
        arc = (f'<circle cx="40" cy="40" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}" '
               f'stroke-dasharray="{filled:.3f} {c:.3f}" transform="rotate(-90 40 40)"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">'
        f'<circle cx="40" cy="40" r="{r}" fill="none" stroke="{PALETTE["rail"]}" stroke-width="{sw}"/>'
        f"{arc}</svg>"
    )


def model_card_html(p: dict) -> str:
    risk = p.get("risk") or {}
    prob = _num(risk.get("probability")) or 0.0
    thr = _num(risk.get("threshold"))
    level = risk.get("level")
    color = RISK_COLORS.get(level, PALETTE["purple"])
    above = risk.get("above_threshold")
    if above is None and thr is not None:
        above = prob >= thr
    pos = "Karar eşiğinin üzerinde" if above else "Karar eşiğinin altında"
    sub = f"{pos} · Risk yüzdelik dilimi: {fmt_pct100(risk.get('percentile'))}"
    note_parts = []
    if thr is not None:
        note_parts.append(f"Eşik: {fmt_pct(thr, 1)}")
    if _num(risk.get("rank")) is not None and _num(risk.get("n")) is not None:
        note_parts.append(f"Sıra: {fmt_int(risk.get('rank'))} / {fmt_int(risk.get('n'))}")
    note = f'<div class="c360-model-note">{_e(" · ".join(note_parts))}</div>' if note_parts else ""
    tip = f"İnaktif olma olasılığı {fmt_pct(prob, 1)}" + (f" · karar eşiği {fmt_pct(thr, 1)}" if thr is not None else "")

    top = (
        '<div class="c360-model">'
        f'<div class="c360-donut" title="{_e(tip)}">{_svg_img(_donut_svg(prob, color), "c360-donut-svg", tip)}'
        f'<div class="c360-donut-val">{_e(fmt_pct(prob))}</div></div>'
        '<div class="c360-model-txt">'
        '<div class="c360-model-title">İnaktif olma olasılığı</div>'
        f'<div class="c360-model-sub">{_e(sub)}</div>{note}'
        '</div></div>'
    )

    drivers = []
    for d in p.get("drivers") or []:
        up = d.get("direction") == "up"
        col = PALETTE["pink"] if up else PALETTE["green"]
        strength = _num(d.get("strength")) or 0.0
        raw = _short_value(d.get("feature", ""), d.get("raw_value"))
        raw_html = f' <span class="c360-driver-raw">· {_e(raw)}</span>' if raw else ""
        contrib = _num(d.get("contribution"))
        dtip = f"Log-odds katkısı: {_dec(contrib, 2)}" if contrib is not None else ""
        drivers.append(
            f'<div class="c360-driver" title="{_e(dtip)}">'
            '<div class="c360-driver-top">'
            f'<div class="c360-driver-label">{_e(d.get("label") or d.get("feature"))}{raw_html}</div>'
            f'<div class="c360-driver-dir" style="color:{col}">{"riski artırıyor" if up else "riski azaltıyor"}</div>'
            '</div>'
            f'<div class="c360-rail c360-rail-thin"><div class="c360-fill" style="width:{_pct_css(max(strength, 0.04) * 100)};background:{col}"></div></div>'
            '</div>'
        )
    drv = ""
    if drivers:
        drv = ('<div class="c360-sep"></div><div class="c360-caption">Tahmini en çok etkileyen etkenler</div>'
               f'<div class="c360-drivers">{"".join(drivers)}</div>')
    return card_html("Model değerlendirmesi", top + drv)


# --------------------------------------------------------------------------- aksiyon + profil
def actions_card_html(p: dict) -> str:
    items = []
    for a in p.get("actions") or []:
        col = _TONE_COLORS.get(a.get("tone"), PALETTE["purple"])
        items.append(
            f'<div class="c360-action" style="border-left-color:{col}">'
            f'<div class="c360-action-title">{_e(a.get("title"))}</div>'
            f'<div class="c360-action-text">{_e(a.get("text"))}</div>'
            '</div>'
        )
    body = f'<div class="c360-actions">{"".join(items)}</div>' if items else '<div class="c360-empty">Önerilen aksiyon yok.</div>'
    return card_html("Önerilen aksiyon", body, right="Kural tabanlı")


def profile_card_html(p: dict) -> str:
    rows = []
    for r in p.get("profile_rows") or []:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            continue
        rows.append(f'<div class="c360-kv-row"><div class="c360-kv-k">{_e(r[0])}</div><div class="c360-kv-v">{_e(r[1])}</div></div>')
    body = f'<div class="c360-kv">{"".join(rows)}</div>' if rows else '<div class="c360-empty">Profil bilgisi yok.</div>'
    return card_html("Kupon ve demografi", body)


# --------------------------------------------------------------------------- senaryo analizi
def _records(df: Any) -> list[dict]:
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        try:
            return list(df.to_dict("records"))
        except TypeError:
            pass
    if isinstance(df, Iterable):
        return [dict(r) for r in df]
    return []


def whatif_card_html(df: Any, threshold: float) -> str:
    """whatif.run çıktısı (Senaryo, Açıklama, Olasılık, Değişim, Eşik altı) için koyu tablo kartı."""
    recs = _records(df)
    title, right = "Senaryo analizi", "Model simülasyonu"
    thr = _num(threshold)
    if not recs:
        return card_html(title, '<div class="c360-empty">Bu hane için senaryo hesaplanamadı.</div>', right=right)

    probs = [v for v in (_num(r.get("Olasılık")) for r in recs) if v is not None]
    scale = max(probs + ([thr] if thr is not None else []) + [0.01]) * 1.15
    scale = min(1.0, scale)

    rows = [
        '<div class="c360-wi-row c360-wi-head"><div class="c360-wi-sc">Senaryo</div><div>Olasılık</div>'
        '<div style="text-align:right">Değişim</div><div style="text-align:right">Durum</div></div>'
    ]
    thr_mark = ""
    if thr is not None:
        thr_mark = f'<div class="c360-thr" style="left:{_pct_css(thr / scale * 100)}" title="Karar eşiği {_e(fmt_pct(thr, 1))}"></div>'
    for i, r in enumerate(recs):
        prob = _num(r.get("Olasılık"))
        delta = _num(r.get("Değişim")) or 0.0
        below = bool(r.get("Eşik altı")) if r.get("Eşik altı") is not None else (thr is not None and prob is not None and prob < thr)
        base = i == 0
        if prob is None:
            bar_col = PALETTE["muted"]
            width = 0.0
        else:
            bar_col = PALETTE["green"] if below else PALETTE["pink"]
            width = prob / scale * 100
        if base or abs(delta) < 0.0005:
            delta_html = f'<span class="c360-muted">{"—" if base else "0,0 puan"}</span>'
        else:
            cls = "c360-down" if delta < 0 else "c360-up"
            sign = "+" if delta > 0 else "-"
            delta_html = f'<span class="{cls}">{sign}{_e(_dec(abs(delta) * 100, 1))} puan</span>'
        badge = _badge("Eşik altı", PALETTE["green"]) if below else _badge("Eşik üstü", PALETTE["pink"])
        rows.append(
            f'<div class="c360-wi-row{" c360-wi-base" if base else ""}">'
            f'<div class="c360-wi-sc"><div class="c360-wi-name">{_e(r.get("Senaryo"))}</div>'
            f'<div class="c360-wi-desc">{_e(r.get("Açıklama"))}</div></div>'
            '<div class="c360-wi-prob"><div class="c360-rail-wrap">'
            f'<div class="c360-rail c360-rail-thin"><div class="c360-fill" style="width:{_pct_css(width)};background:{bar_col}"></div></div>'
            f'{thr_mark}</div><div class="c360-wi-pv">{_e(fmt_pct(prob, 1))}</div></div>'
            f'<div class="c360-wi-delta">{delta_html}</div>'
            f'<div class="c360-wi-status">{badge}</div>'
            '</div>'
        )
    note = ('<div class="c360-wi-note">Beyaz çizgi karar eşiğini gösterir'
            f'{" (" + _e(fmt_pct(thr, 1)) + ")" if thr is not None else ""}. '
            'Senaryolar modelin tepkisini gösterir; nedensel etki değildir.</div>')
    body = f'<div class="c360-wi">{"".join(rows)}</div>{note}'
    return card_html(title, body, right=right)


def table_html(df: Any, title: str | None = None, right: str | None = None, numeric: tuple = ()) -> str:
    """Koyu, tema bağımsız HTML tablo kartı. Değerler önceden biçimlendirilmiş olmalı."""
    cols = list(df.columns)
    head = "".join(f'<th style="text-align:{"right" if c in numeric else "left"}">{_e(c)}</th>' for c in cols)
    body_rows = []
    for r in df.itertuples(index=False):
        cells = "".join(f'<td class="{"num" if c in numeric else ""}">{_e(v)}</td>' for c, v in zip(cols, r))
        body_rows.append(f"<tr>{cells}</tr>")
    body = (f'<div class="c360-table-wrap"><table class="c360-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table></div>')
    return card_html(title, body, right=right)
