"""Customer 360 - müşteri profili, portföy ve LLM chatbot. Çalıştır: py -3 -m streamlit run app.py"""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import data as D  # noqa: E402
from src import llm as L  # noqa: E402
from src import names as N  # noqa: E402
from src import profile as P  # noqa: E402
from src import ui as U  # noqa: E402
from src import whatif as W  # noqa: E402

st.set_page_config(page_title="Customer 360", page_icon="📊", layout="wide")
st.html(U.theme_css())

PAL = U.PALETTE
THRESHOLD = P.THRESHOLD
PAGES = ["Müşteri Profili", "Portföy", "Chatbot"]
SLUG_TO_PAGE = {"profil": 0, "portfoy": 1, "dashboard": 1, "chatbot": 2}
PAGE_TO_SLUG = {0: "profil", 1: "portfoy", 2: "chatbot"}
TREND_METRICS = {"Harcama": "spend", "Ziyaret": "visits", "Aktif hafta": "active_weeks", "Risk olasılığı": "probability"}
RISK_QUESTION = "Bu müşteriyi riskliden risksize nasıl geçiririz? Senaryolara dayanarak somut öneri ver."

# ---------------------------------------------------------------- veri
try:
    hh_table = N.add_names(D.build_household_table())
    seg_profiles = D.load_segment_profiles()
    shap_df = D.load_shap_drivers()
    feature_dict = D.load_feature_dictionary()
    metrics = D.load_test_metrics()
    importance = D.load_feature_importance(8)
    ctx = D.load_profile_context()
except Exception as exc:  # noqa: BLE001
    st.error(f"Veri yüklenemedi: {exc}")
    st.stop()

keys = hh_table["household_key"].astype(int).tolist()  # risk sırasına göre; ilk eleman en riskli
labels = dict(zip(keys, hh_table.apply(N.label, axis=1)))
n_households = len(keys)

# ---------------------------------------------------------------- durum + query param
qp = st.query_params
if "page" not in st.session_state:
    st.session_state.page = PAGES[SLUG_TO_PAGE.get(qp.get("page", "profil"), 0)]
if "hane" not in st.session_state:
    try:
        requested = int(qp.get("hane", keys[0]))
    except ValueError:
        requested = keys[0]
    st.session_state.hane = requested if requested in labels else keys[0]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = qp.get("q") or None


def open_profile(key: int) -> None:
    st.session_state.hane = int(key)
    st.session_state.page = PAGES[0]


def ask_chatbot(question: str) -> None:
    st.session_state.pending_prompt = question
    st.session_state.page = PAGES[2]


def set_prompt(question: str) -> None:
    st.session_state.pending_prompt = question


def clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.pending_prompt = None


def dark(fig, title: str, height: int):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=12, r=12, t=52, b=12),
        title=dict(text=title, font=dict(size=15, color=PAL["text"])),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PAL["muted"], family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PAL["text"])),
        hoverlabel=dict(bgcolor=PAL["inner"], font_color=PAL["text"]),
    )
    fig.update_xaxes(gridcolor=PAL["border"], zerolinecolor=PAL["border"], linecolor=PAL["border"])
    fig.update_yaxes(gridcolor=PAL["border"], zerolinecolor=PAL["border"], linecolor=PAL["border"])
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Customer 360")
    st.caption("Hane pasiflik riski · test snapshot'ı 655. gün")
    page = st.radio("Sayfa", PAGES, key="page")
    selected_key = st.selectbox(
        "Müşteri", keys, key="hane", format_func=lambda k: labels[k],
        help="Liste risk sırasına göre; en üstte en riskli müşteri. Adlar sentetiktir, hane numarası gerçektir.")
    st.divider()
    st.caption(f"Karar eşiği: %{THRESHOLD * 100:.1f} (validation F2)")
    st.caption(f"Hedefleme: en riskli %10 ({int(hh_table['selected_top_k'].sum())} hane)")
    st.caption("Tahmin ufku: 56 gün · Gözlem penceresi: 182 gün")

page_idx = PAGES.index(page)
st.query_params.update({"page": PAGE_TO_SLUG[page_idx], "hane": str(selected_key)})
p = P.household_profile(ctx, int(selected_key))

# ---------------------------------------------------------------- müşteri profili
if page_idx == 0:
    st.html(U.header_html(p))
    st.html(U.kpi_row_html(p))

    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        with st.container(key="c360-col-left"):
            metric = st.segmented_control("Davranış metriği", list(TREND_METRICS), default="Harcama",
                                          key="trend_metric", label_visibility="collapsed")
            st.html(U.trend_card_html(p, TREND_METRICS.get(metric or "Harcama", "spend")))
            if metric == "Risk olasılığı":
                st.caption(
                    "431, 487 ve 543. gün snapshot'ları modelin eğitiminde görüldü; bu noktalar iyimser olabilir.")
            st.html(U.departments_card_html(p))
            st.html(U.profile_card_html(p))
    with right:
        with st.container(key="c360-col-right"):
            st.html(U.model_card_html(p))
            st.html(U.actions_card_html(p))

    st.html(U.section_title_html(
        "Riski düşürmek için ne yapmalı?",
        "Model simülasyonu: bu müşteri şu davranışı gösterseydi olasılık ne olurdu? Nedensel kanıt değildir; "
        "kampanya etkisi A/B testiyle ölçülmelidir."))
    feats = D.load_model_features()
    wf = W.run(D.load_bundle(), feats, int(selected_key), THRESHOLD) if feats is not None else None
    st.html(U.whatif_card_html(wf, THRESHOLD))
    st.button("💬 Bu müşteriyi Chatbot'a sor", type="primary", on_click=ask_chatbot, args=(RISK_QUESTION,))

# ---------------------------------------------------------------- portföy
elif page_idx == 1:
    st.html(U.section_title_html("Portföy görünümü", f"{n_households:,} hane · snapshot 655".replace(",", ".")))
    c1, c2, c3, c4 = st.columns(4)
    c1.html(U.stat_card_html("Toplam hane", U.fmt_int(n_households), "Test snapshot'ı"))
    c2.html(U.stat_card_html("En riskli %10", U.fmt_int(hh_table["selected_top_k"].sum()), "Kampanya hedefi",
                             accent=PAL["pink"]))
    c3.html(U.stat_card_html("Ortalama risk", U.fmt_pct(hh_table["lapse_probability"].mean(), 1),
                             "İnaktif olma olasılığı"))
    c4.html(U.stat_card_html("ROC-AUC (test)", f"{metrics.get('roc_auc', float('nan')):.3f}".replace(".", ","),
                             "1'e yakın = iyi sıralama", accent=PAL["green"]))

    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(hh_table, x="lapse_probability", nbins=50, color_discrete_sequence=[PAL["purple"]],
                           labels={"lapse_probability": "İnaktif olma olasılığı"})
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=PAL["pink"],
                      annotation_text="eşik", annotation_font_color=PAL["pink"], annotation_position="top right")
        fig.update_layout(yaxis_title="Hane sayısı", bargap=0.05)
        st.plotly_chart(dark(fig, "Risk dağılımı", 340), width="stretch")
    with g2:
        imp = importance.assign(etiket=importance["feature"].map(lambda f: P.FEATURE_LABELS.get(f, f)))
        fig = px.bar(imp.sort_values("importance_mean"), x="importance_mean", y="etiket", orientation="h",
                     error_x="importance_std", color_discrete_sequence=[PAL["purple"]],
                     labels={"importance_mean": "Permütasyon önemi", "etiket": ""})
        st.plotly_chart(dark(fig, "Modelin en çok baktığı 8 özellik", 340), width="stretch")

    if "segment_name" in hh_table.columns and seg_profiles is not None:
        order = seg_profiles.sort_values("segment_id")["segment_name"].tolist()
        s1, s2 = st.columns(2)
        with s1:
            fig = px.bar(seg_profiles, x="segment_name", y="n_households", color="segment_name", text="n_households",
                         category_orders={"segment_name": order}, color_discrete_map=U.SEGMENT_COLORS,
                         labels={"segment_name": "", "n_households": "Hane sayısı"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(dark(fig, "Segment büyüklüğü", 320), width="stretch")
        with s2:
            fig = px.bar(seg_profiles, x="segment_name", y="mean_lapse_probability", color="segment_name",
                         text=seg_profiles["mean_lapse_probability"].map(lambda v: f"%{v * 100:.1f}".replace(".", ",")),
                         category_orders={"segment_name": order}, color_discrete_map=U.SEGMENT_COLORS,
                         labels={"segment_name": "", "mean_lapse_probability": "Ortalama olasılık"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(dark(fig, "Segment başına ortalama risk", 320), width="stretch")

        fig = px.scatter(hh_table, x="recency_days", y="monetary", color="segment_name", size="lapse_probability",
                         size_max=18, opacity=0.7, category_orders={"segment_name": order},
                         color_discrete_map=U.SEGMENT_COLORS,
                         hover_data={"household_key": True, "lapse_probability": ":.3f"},
                         labels={"recency_days": "Son alışverişten beri (gün)", "monetary": "182 günlük harcama ($)",
                                 "segment_name": "Segment", "lapse_probability": "Risk",
                                 "household_key": "Hane"})
        fig.update_yaxes(type="log")
        st.plotly_chart(dark(fig, "RFM haritası (nokta boyutu = risk)", 420), width="stretch")

        with st.expander("Segment profilleri (tablo)"):
            sp = seg_profiles.sort_values("segment_id")
            seg_show = pd.DataFrame({
                "Segment": sp["segment_name"],
                "Hane": sp["n_households"].map(U.fmt_int),
                "Pay": sp["share_pct"].map(lambda v: f"%{v:.1f}".replace(".", ",")),
                "Son alışveriş (medyan gün)": sp["median_recency_days"].map(U.fmt_int),
                "Sepet (medyan)": sp["median_frequency_baskets"].map(U.fmt_int),
                "Harcama (medyan)": sp["median_monetary"].map(U.fmt_money),
                "Ortalama risk": sp["mean_lapse_probability"].map(lambda v: f"%{v * 100:.1f}".replace(".", ",")),
                "Gerçek pasif oranı": sp["actual_inactive_rate_pct"].map(lambda v: f"%{v:.1f}".replace(".", ",")),
            })
            st.html(U.table_html(seg_show, numeric=tuple(seg_show.columns[1:])))

    st.html(U.section_title_html("En riskli 20 müşteri", "Bir müşteriyi seçip profilini açabilirsiniz."))
    top20 = hh_table.head(20)
    t1, t2 = st.columns([3, 1], gap="medium")
    with t1:
        show = top20[["risk_rank", "household_key", "segment_name", "lapse_probability",
                      "target_inactive"]].rename(columns={
            "risk_rank": "Sıra", "household_key": "Hane", "segment_name": "Segment",
            "lapse_probability": "Olasılık", "target_inactive": "Gerçekte pasifleşti"})
        show["Olasılık"] = show["Olasılık"].map(lambda v: f"%{v * 100:.1f}".replace(".", ","))
        show["Gerçekte pasifleşti"] = show["Gerçekte pasifleşti"].map({1: "Evet", 0: "Hayır"})
        show["Hane"] = show["Hane"].map(lambda k: f"#{int(k)}")
        st.html(U.table_html(show, numeric=("Sıra",)))
    with t2:
        top_keys = top20["household_key"].astype(int).tolist()
        pick = st.selectbox("Müşteri seç", top_keys, format_func=lambda k: labels[k], key="portfoy_pick")
        st.button("Profili aç", type="primary", on_click=open_profile, args=(pick,), width="stretch")

# ---------------------------------------------------------------- chatbot
else:
    st.html(U.header_html(p))
    st.caption(f"Model: {L.DEFAULT_MODEL} · Cevaplar seçili müşterinin verisine dayanır; öneriler test edilmemiş hipotezdir.")

    if not L.has_api_key():
        st.warning("OPENAI_API_KEY bulunamadı. Chatbot için anahtarı `.env` dosyasına yazın; "
                   "profil ve portföy sayfaları anahtar olmadan da çalışır.")
    else:
        feats = D.load_model_features()
        extras = D.load_extra_features()
        extras_row = extras.loc[int(selected_key)].to_dict() if int(selected_key) in extras.index else None
        context = {
            "household": hh_table.loc[hh_table["household_key"] == selected_key].iloc[0].to_dict(),
            "drivers": shap_df[shap_df["household_key"] == selected_key],
            "profiles": seg_profiles,
            "feature_dict": feature_dict,
            "n_households": n_households,
            "profile": p,
            "whatif": W.run(D.load_bundle(), feats, int(selected_key), THRESHOLD) if feats is not None else None,
            # --- yeni: kullanilmayan tablolardan gelen ek baglam
            "portfolio": D.load_portfolio_summary(hh_table, THRESHOLD),
            "extras": extras_row,
            "demographic": D.load_demographics().get(int(selected_key)),
            "demographic_missing_share": ctx.get("demographic_missing_share"),
            "commodities": D.load_top_commodities(int(selected_key)),
            "campaigns": D.load_campaign_history(int(selected_key)),
            "last_baskets": D.load_last_baskets(int(selected_key)),
        }
        system_prompt = L.build_system_prompt(context)

        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 0.7])
        b1.button("Bu müşteri neden riskli?", width="stretch", on_click=set_prompt,
                  args=("Bu müşteri neden riskli?",))
        b2.button("Riski nasıl düşürürüz?", width="stretch", on_click=set_prompt, args=(RISK_QUESTION,))
        b3.button("Hangi segmente hangi kampanya?", width="stretch", on_click=set_prompt,
                  args=("Hangi segmente hangi kampanya uygun?",))
        b4.button("Modelin sınırlılıkları", width="stretch", on_click=set_prompt,
                  args=("Modelin sınırlılıkları neler?",))
        b5.button("Temizle", width="stretch", on_click=clear_chat)

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_input = st.chat_input("Sorunuzu yazın...")
        prompt = user_input or st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        if "q" in st.query_params:
            del st.query_params["q"]

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor..."):
                    try:
                        messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history[-10:]
                        answer = L.chat(messages, L.DEFAULT_MODEL)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"LLM çağrısı başarısız: {exc}")
                        answer = None
                if answer:
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.expander("Sistem promptu (bağlam)"):
            st.code(system_prompt, language="markdown")
