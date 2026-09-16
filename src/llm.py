"""LLM chatbot yardimcilari: sistem promptu ve OpenAI cagrisi."""
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

PROJECT_SUMMARY = """Sen "Customer 360" uygulamasinin Turkce konusan asistanisin. Sadece Turkce ve kisa yanit ver.

Proje ozeti:
- Veri: dunnhumby "The Complete Journey" (2.500 hane, ~2 yil perakende islem verisi).
- Hedef: Bir hanenin snapshot gununden sonraki 56 gun icinde hic gecerli alisveris yapmama riski ("pasiflik"); sozlesmeli churn degildir.
- Gozlem penceresi son 182 gun, tahmin penceresi sonraki 56 gun. Analiz birimi household_key x snapshot_day.
- Modele alinma kosullari: gozlem doneminde en az 2 gecerli sepet, ilk alisveristen bu yana en az 90 gun, son 84 gunde en az 1 sepet.
- Secilen model: Logistic Regression (C=0.2, class_weight=None), behavior_core_linear feature seti (37 feature). Secim metrigi validation Average Precision.
- Test snapshot'i 655. gun, 2.284 hane. Test metrikleri: ROC-AUC 0.851, Average Precision 0.273, Brier 0.058, pozitif oran %7.2.
- Karar esigi validation F2 ile 0.10205; operasyonel senaryo en riskli %10 hanenin hedeflenmesi (229 hane, recall@top10 0.436, precision@top10 0.314).
- Ek kalibrasyon Brier'i iyilestirmedigi icin modelin dogal olasiligi korunmustur.
- Sinirliliklar: pozitif sinif orani sadece %6-7; ayni haneler farkli snapshot'larda tekrar eder, yeni hanelere genelleme test edilmedi; test snapshot'i onceki revizyonlarda gorulmustur, tam bagimsiz holdout degildir; demografi yalnizca hanelerin %32'sini kapsar; kampanya analizleri nedensel degildir.
- SHAP katkilari yalnizca tahmin aciklamasidir; nedensel etki veya kesin musteri davranisi olarak sunulmamalidir. Kullaniciya donusturulmus deger degil ham feature degeri (raw_feature_value) gosterilir.

Kurallar:
- Sayilari asagidaki baglamdaki gibi kullan, uydurma. Baglamda olmayan bir bilgi sorulursa bilmedigini soyle.
- Kampanya onerilerinde segment profillerine dayan ve bunlarin veri destekli hipotez oldugunu belirt.
- Portfoy sorularinda (kac hane riskli, segmentler nasil, en riskli kimler) yukaridaki portfoy ozetini kullan; listede olmayan bir hane sorulursa kullanicidan o haneyi soldaki listeden secmesini iste.
- Kategori, magaza, saat, kampanya ve kupon bilgileri yalnizca gozlem penceresine aittir; tahmin donemine ait bilgi yoktur.
- Demografi modele girmedi; riskin nedeni olarak sunma, yalnizca iletisim onerisinde renk ver.
"""


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _profiles_markdown(profiles: pd.DataFrame | None) -> str:
    if profiles is None or profiles.empty:
        return "Segment profilleri: segmentasyon henuz calistirilmadi.\n"
    cols = [
        ("segment_name", "Segment"), ("n_households", "Hane"), ("share_pct", "Pay %"),
        ("median_recency_days", "Med. recency (gun)"), ("median_frequency_baskets", "Med. sepet"),
        ("median_monetary", "Med. harcama"), ("median_avg_basket_value", "Med. sepet degeri"),
        ("mean_lapse_probability", "Ort. risk"), ("top10_risk_share_pct", "Top10 payi %"),
        ("actual_inactive_rate_pct", "Gercek pasif %"),
    ]
    cols = [(c, h) for c, h in cols if c in profiles.columns]
    lines = ["Segment profilleri (test snapshot'i):", "| " + " | ".join(h for _, h in cols) + " |",
             "|" + "---|" * len(cols)]
    for _, r in profiles.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c, _ in cols) + " |")
    if "description" in profiles.columns:
        lines.append("")
        for _, r in profiles.iterrows():
            lines.append(f"- {r['segment_name']}: {r['description']}")
    return "\n".join(lines) + "\n"


def _household_markdown(context: dict) -> str:
    hh = context.get("household")
    if hh is None:
        return "Secili hane: yok.\n"
    lines = [f"Secili musteri: Hane #{hh.get('household_key')} (veri seti anonim; musteriye asla isimle degil hane numarasiyla hitap et)",
             f"- Pasiflik olasiligi: {_fmt(hh.get('lapse_probability'))}",
             f"- Risk sirasi: {_fmt(hh.get('risk_rank'))} / {context.get('n_households', '-')}",
             f"- Risk yuzdeligi: {_fmt(hh.get('risk_percentile'))}",
             f"- En riskli %10 icinde mi: {'evet' if hh.get('selected_top_k') else 'hayir'}",
             f"- Gercek sonuc (test): {'pasif' if hh.get('target_inactive') else 'aktif kaldi'}"]
    if hh.get("segment_name") is not None and not pd.isna(hh.get("segment_name")):
        lines.append(f"- Segment: {hh['segment_name']}")
        rfm = [("recency_days", "recency (gun)"), ("frequency_baskets", "sepet sayisi"),
               ("monetary", "toplam harcama"), ("avg_basket_value", "ort. sepet degeri"),
               ("active_weeks", "aktif hafta"), ("unique_products", "benzersiz urun")]
        lines.append("- RFM: " + ", ".join(f"{h}={_fmt(hh.get(c), 2)}" for c, h in rfm if hh.get(c) is not None))
    else:
        lines.append("- Segment: segmentasyon henuz calistirilmadi")
    drivers = context.get("drivers")
    fdict = context.get("feature_dict") or {}
    if drivers is not None and len(drivers):
        lines.append("- SHAP surucileri (log-odds katkisi, nedensel degil):")
        for _, d in drivers.iterrows():
            desc = fdict.get(d["raw_feature"], "")
            lines.append(f"  {int(d['driver_rank'])}. {d['raw_feature']} = {_fmt(d['raw_feature_value'], 2)} "
                         f"({d['direction']}, katki {_fmt(d['shap_contribution_model_output'])})"
                         + (f" - {desc}" if desc else ""))
    else:
        lines.append("- SHAP surucileri: bu hane icin mevcut degil (yalnizca en riskli 25 hane icin hesaplandi).")
    prof = context.get("profile")
    if prof:
        k = prof.get("kpis", {})
        r = prof.get("risk", {})
        lines.append(f"- Risk seviyesi: {r.get('level')} (olasilik {r.get('probability', 0):.3f}, esik {r.get('threshold', 0):.3f}, yuzdelik {r.get('percentile', 0):.0f})")
        lines.append(f"- KPI: son alisveristen beri {k.get('recency_days')} gun (onceki snapshot'a gore degisim: {k.get('recency_delta')}), "
                     f"182 gunluk harcama ${k.get('spend_182d', 0):.2f} (hane yuzdelik {k.get('spend_percentile', 0):.0f}), "
                     f"{k.get('visits_182d')} ziyaret, {k.get('active_weeks')} aktif hafta, promosyonlu sepet payi {k.get('promo_share', 0):.2f} (segment ortalamasinin {k.get('promo_vs_segment')})")
        if prof.get("trend"):
            lines.append("- Snapshot trendi (gun: harcama / ziyaret / aktif hafta): " + "; ".join(
                f"{t['snapshot_day']}: ${t['spend']:.0f} / {t['visits']} / {t['active_weeks']}" for t in prof["trend"]))
        if prof.get("departments"):
            lines.append("- Departman payi: " + ", ".join(f"{d['name']} %{d['share'] * 100:.0f}" for d in prof["departments"]))
        if prof.get("drivers"):
            lines.append("- Model suruculeri (dogrusal log-odds katkisi, tum haneler icin, nedensel degil): " + "; ".join(
                f"{d['label']} ({'riski artiriyor' if d['direction'] == 'up' else 'riski azaltiyor'}, {d['contribution']:+.2f})" for d in prof["drivers"]))
        if prof.get("actions"):
            lines.append("- Kural tabanli aksiyon onerileri (test edilmemis hipotez): " + " | ".join(
                f"{a['title']}: {a['text']}" for a in prof["actions"]))
    wf = context.get("whatif")
    if wf is not None and len(wf) > 1:
        lines.append("- Senaryo analizi (model simulasyonu, nedensel degil: 'bu davranis olsaydi model ne derdi?'):")
        for _, r in wf.iterrows():
            lines.append(f"  * {r['Senaryo']}: olasilik {r['Olasılık']:.3f} "
                         f"({'esik alti' if r['Eşik altı'] else 'esik ustu'}) - {r['Açıklama']}")
        lines.append("  Kullanici 'riskliyi nasil risksize ceviririz' diye sorarsa bu senaryolara dayan; "
                     "en dusuk olasilikli senaryoyu one cikar ve bunun simulasyon oldugunu belirt.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- yeni baglam bloklari
UNKNOWN_DEMO = {"Unknown", "None/Unknown", "", "nan", "None"}

DEMO_LABELS = {
    "classification_1": "Yas grubu (anonim kod)",
    "classification_2": "classification_2",
    "classification_3": "classification_3",
    "classification_4": "classification_4",
    "classification_5": "classification_5",
    "homeowner_desc": "Ev sahipligi",
    "kid_category_desc": "Cocuk durumu",
}


def _portfolio_markdown(ps: dict | None) -> str:
    if not ps:
        return ""
    lines = [
        "Portfoy ozeti (test snapshot'i, 655. gun):",
        f"- Toplam hane: {ps['n_households']}; ortalama risk {ps['mean_probability']:.3f}, "
        f"medyan risk {ps['median_probability']:.3f}",
        f"- Karar esiginin ustunde: {ps['above_threshold']} hane; en riskli %10 listesi: {ps['top_k_count']} hane",
        f"- Tahmin doneminde gercekten pasiflesen: {ps['actual_inactive']} hane",
    ]
    for s in ps.get("segments", []):
        lines.append(f"- {s['segment_name']}: {s['hane']} hane, ort. risk {s['ort_risk']:.3f}, "
                     f"top10 listesinde {int(s['top10'])}, gercekte pasiflesen {int(s['gercek_pasif'])}")
    if ps.get("top10_riskli"):
        lines.append("- En riskli 10 hane: " + "; ".join(
            f"#{int(r['household_key'])} ({r['lapse_probability']:.2f})"
            for r in ps["top10_riskli"]))
    return "\n".join(lines) + "\n"


def _extras_markdown(e: dict | None) -> str:
    if not e:
        return ""
    return "\n".join([
        "Ek davranis ozellikleri (ayni 182 gunluk gozlem penceresi, model tablosundan):",
        f"- Kategori genisligi: {_fmt(e.get('unique_departments'))} departman, "
        f"{_fmt(e.get('unique_commodities'))} kategori, {_fmt(e.get('unique_subcommodities'))} alt kategori; "
        f"tekrar alinan urun orani {_fmt(e.get('repeat_product_rate'), 2)}",
        f"- Yogunlasma: en buyuk departman payi {_fmt(e.get('top_department_revenue_share'), 2)}, "
        f"ilk 3 departman payi {_fmt(e.get('top3_department_revenue_share'), 2)}, "
        f"ozel marka payi {_fmt(e.get('private_brand_revenue_share'), 2)}",
        f"- Son 8 hafta: birakilan departman {_fmt(e.get('lost_departments_8w'))}, "
        f"yeni denenen {_fmt(e.get('new_departments_8w'))}, "
        f"devamlilik orani {_fmt(e.get('department_retention_rate'), 2)}",
        f"- Magaza: {_fmt(e.get('unique_stores'))} farkli magaza, "
        f"ana magaza payi {_fmt(e.get('dominant_store_share'), 2)}",
        f"- Alisveris saati: ortalama {_fmt(e.get('avg_transaction_hour'))}, "
        f"sabah {_fmt(e.get('morning_basket_rate'), 2)} / ogleden sonra {_fmt(e.get('afternoon_basket_rate'), 2)} / "
        f"aksam {_fmt(e.get('evening_basket_rate'), 2)}",
        f"- Kampanya (model tablosu): tamamlanmis {_fmt(e.get('completed_campaigns_received'))} kampanya "
        f"(TypeA {_fmt(e.get('completed_campaigns_received_TypeA'))}, "
        f"TypeB {_fmt(e.get('completed_campaigns_received_TypeB'))}, "
        f"TypeC {_fmt(e.get('completed_campaigns_received_TypeC'))}); "
        f"snapshot aninda aktif {_fmt(e.get('active_campaigns_at_snapshot'))}",
        f"- Kupon: {_fmt(e.get('coupon_redemption_events_obs'))} kullanim, "
        f"{_fmt(e.get('unique_coupons_redeemed_obs'))} farkli kupon; son kullanimdan beri "
        f"{_fmt(e.get('days_since_last_redemption'))} gun, son kampanyadan beri "
        f"{_fmt(e.get('days_since_last_campaign'))} gun",
    ]) + "\n"


def _demo_markdown(demo: dict | None, missing_share: float | None = None) -> str:
    if not demo:
        share = f" (hanelerin yaklasik %{(missing_share or 0) * 100:.0f}'inde demografi yok)" if missing_share else ""
        return f"Demografi: bu hane icin kayit yok{share}; demografiye dayali yorum yapma.\n"
    parts = [f"{label}: {demo.get(col)}" for col, label in DEMO_LABELS.items()
             if demo.get(col) is not None and str(demo.get(col)).strip() not in UNKNOWN_DEMO]
    if not parts:
        return "Demografi: kayit var ama tum alanlar bilinmiyor.\n"
    return ("Demografi (dunnhumby anonim kodlari; modele girmedi, yalnizca yorum icin): "
            + "; ".join(parts) + "\n")


def _commodity_markdown(tc) -> str:
    if tc is None or len(tc) == 0:
        return "Kategori kirilimi: gozlem penceresinde islem bulunamadi.\n"
    lines = ["En cok harcanan kategoriler (182 gunluk pencere, transaction_data + product):"]
    for r in tc.itertuples():
        lines.append(f"- {r.commodity_desc}: ${r.spend:.2f} (harcamanin %{r.share * 100:.0f}'i), "
                     f"{int(r.baskets)} sepette")
    return "\n".join(lines) + "\n"


def _campaign_markdown(ch: dict | None) -> str:
    if not ch or not ch.get("campaigns"):
        return "Kampanya gecmisi: snapshot gunune kadar bu haneye kampanya atanmamis.\n"
    lines = [f"Kampanya gecmisi (snapshot gunune kadar, {len(ch['campaigns'])} kampanya):"]
    for c in ch["campaigns"][-6:]:
        aktif = " [snapshot aninda aktif]" if c["active_at_snapshot"] else ""
        lines.append(f"- Kampanya {c['campaign']} ({c['type']}), gun {c['start_day']}-{c['end_day']}{aktif}")
    if ch.get("redemptions"):
        days = sorted({r["day"] for r in ch["redemptions"]})
        camps = sorted({r["campaign"] for r in ch["redemptions"]})
        lines.append(f"- Kupon kullanimi: {len(ch['redemptions'])} kupon, {len(camps)} kampanyadan "
                     f"({', '.join(str(c) for c in camps)}); son kullanim gunu {days[-1]}")
    else:
        lines.append("- Kupon kullanimi: kampanya aldi ama hic kupon kullanmadi.")
    return "\n".join(lines) + "\n"


def _baskets_markdown(lb: list[dict] | None) -> str:
    if not lb:
        return ""
    lines = ["Son sepetler (transaction_data, snapshot oncesi):"]
    for b in lb:
        lines.append(f"- Gun {b['day']} ({b['days_ago']} gun once): ${b['spend']:.2f}, {b['lines']} satir, "
                     f"magaza {b['store']}, indirim ${b['discount']:.2f}")
    return "\n".join(lines) + "\n"


def build_system_prompt(context: dict) -> str:
    """Beklenen anahtarlar: household, drivers, profiles, feature_dict, n_households,
    profile, whatif + (yeni) portfolio, extras, demographic, demographic_missing_share,
    commodities, campaigns, last_baskets."""
    parts = [
        PROJECT_SUMMARY,
        _portfolio_markdown(context.get("portfolio")),
        _profiles_markdown(context.get("profiles")),
        _household_markdown(context),
        _extras_markdown(context.get("extras")),
        _demo_markdown(context.get("demographic"), context.get("demographic_missing_share")),
        _commodity_markdown(context.get("commodities")),
        _campaign_markdown(context.get("campaigns")),
        _baskets_markdown(context.get("last_baskets")),
    ]
    return "\n".join(p for p in parts if p)


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def chat(messages: list[dict], model: str | None = None) -> str:
    """OpenAI chat.completions cagrisi. Hata durumunda exception firlatir."""
    from openai import OpenAI

    client = OpenAI()
    model = model or DEFAULT_MODEL
    # GPT-5 ailesi max_tokens ve temperature kabul etmez; ortak parametre kullan.
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 700}
    if not model.startswith(("gpt-5", "o")):
        kwargs["temperature"] = 0.2
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
