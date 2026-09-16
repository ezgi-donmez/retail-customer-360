"""Sentetik müşteri adları.

Veri seti anonimdir (household_key). Demo ve sunum için her haneye SABİT ve BENZERSİZ
bir Türkçe ad atanır. Gerçek kişilerle ilgisi yoktur. Arayüzde gösterilmez; hane numarası
kullanılır, ad yalnızca arka planda customer_name sütununda tutulur.

Yöntem: 60 ad x 60 soyad = 3.600 kombinasyon. Hane numaraları 1..2500 artan sırayla
dolaşılır; her hane, numarasının md5 özetinden bir başlangıç kombinasyonu alır, o
kombinasyon doluysa sıradaki boş kombinasyona geçer. Böylece iki haneye aynı ad düşmez
ve sonuç her çalıştırmada aynıdır.
"""
from functools import lru_cache
import hashlib

import pandas as pd

FIRST = [
    "Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Şerife", "Sultan", "Hanife",
    "Mehmet", "Mustafa", "Ahmet", "Ali", "Hüseyin", "Hasan", "İbrahim", "İsmail", "Osman", "Yusuf",
    "Selin", "Deniz", "Ece", "Melis", "Cem", "Burak", "Emre", "Kerem", "Tuğba", "Gizem",
    "Merve", "Esra", "Büşra", "Seda", "Derya", "Pınar", "Aslı", "Nazlı", "Ceren", "Damla",
    "Murat", "Serkan", "Onur", "Barış", "Kaan", "Mert", "Can", "Oğuz", "Tolga", "Volkan",
    "Hakan", "Levent", "Sinan", "Umut", "Berk", "Arda", "Furkan", "Gökhan", "Özge", "İrem",
]
LAST = [
    "Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir",
    "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek",
    "Polat", "Korkmaz", "Erdoğan", "Güneş", "Aksoy", "Bulut", "Taş", "Acar", "Turan", "Aktaş",
    "Keskin", "Ünal", "Tekin", "Karaca", "Uçar", "Duman", "Sarı", "Çakır", "Işık", "Ateş",
    "Tunç", "Erdem", "Bozkurt", "Avcı", "Kaplan", "Özer", "Yavuz", "Uysal", "Ekinci", "Önal",
    "Karakaya", "Sezer", "Gündoğdu", "Coşkun", "Başaran", "Altun", "Güler", "Oral", "Aydemir", "Tosun",
]
MAX_KEY = 2500  # veri setindeki hane numaraları 1..2500


def _hash(key: int) -> int:
    return int(hashlib.md5(f"cj-{int(key)}".encode()).hexdigest()[:12], 16)


@lru_cache(maxsize=1)
def _mapping() -> dict[int, str]:
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    n = len(combos)
    used: set[int] = set()
    out: dict[int, str] = {}
    for key in range(1, MAX_KEY + 1):
        idx = _hash(key) % n
        while idx in used:
            idx = (idx + 1) % n
        used.add(idx)
        out[key] = combos[idx]
    return out


def name_for(household_key: int) -> str:
    """Hane numarasına karşılık gelen benzersiz sentetik ad."""
    key = int(household_key)
    return _mapping().get(key, f"Müşteri {key}")


def add_names(df: pd.DataFrame) -> pd.DataFrame:
    """household_key sütununa göre 'customer_name' ekler (varsa dokunmaz)."""
    if "customer_name" in df.columns:
        return df
    out = df.copy()
    out["customer_name"] = out["household_key"].map(name_for)
    return out


def label(row) -> str:
    """Seçici etiketi: 'Hane #1108 · risk %83'."""
    return f"Hane #{int(row['household_key'])} · risk %{row['lapse_probability'] * 100:.0f}"
