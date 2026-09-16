Bu proje, MIUUL AI Data Scientist Bootcamp 21. Dönem final projesi kapsamında geliştirilmiştir. Proje sürecindeki değerli yönlendirmeleri ve destekleri için mentörümüz **Mustafa Gürkan Çanakçı’ya** ve **MIUUL ekibine** teşekkür ederiz.

## Proje Ekibi

- Selda Kırkkanat
- Ezgi Dönmez
- Tuğçe Güven

# The Complete Journey — Perakendede Müşteri Kaybı Tahmini ve Akıllı Karar Destek Sistemi

Bu proje, dunnhumby **The Complete Journey** veri setindeki geçmiş alışveriş davranışlarını kullanarak bir hanenin izleyen **56 gün içinde hiç geçerli alışveriş yapmama riskini** tahmin eder.

Buradaki hedef sözleşmeli hizmetlerdeki kesin müşteri kaybı (`churn`) değildir. Perakende bağlamında **yaklaşan alışveriş pasifliği** olarak yorumlanır. Model; riskli haneleri önceliklendirmek, sınırlı kampanya kapasitesini daha verimli kullanmak ve ileride geliştirilecek Customer 360/chatbot uygulamasına açıklanabilir çıktılar sağlamak üzere tasarlanmıştır.

## Projenin kısa özeti

| Başlık | Mevcut tasarım |
|---|---|
| Analiz birimi | `household_key × snapshot_day` |
| Gözlem penceresi | Son 182 gün |
| Tahmin penceresi | Sonraki 56 gün |
| Pozitif sınıf | Tahmin döneminde hiç geçerli sepet oluşturmayan hane |
| Train snapshot'ları | 431, 487 ve 543. günler |
| Validation snapshot'ı | 599. gün |
| Test snapshot'ı | 655. gün |
| Seçilen model | Logistic Regression |
| Seçilen feature seti | `behavior_core_linear` — 37 feature |
| Model seçim metriği | Validation Average Precision |
| Karar eşiği | Validation F2 ile seçilen `0.10205` |
| Operasyonel senaryo | En riskli %10 hanenin hedeflenmesi |

## Veri seti

The Complete Journey, yaklaşık iki yıllık hane düzeyinde perakende davranışını ilişkisel tablolar halinde sunar.

| Veri bileşeni | Projedeki kapsam | Kullanım amacı |
|---|---:|---|
| İşlem verisi | 2.595.732 satır | Harcama, sıklık, sepet, ritim ve trend feature'ları |
| Sepet | 276.484 benzersiz sepet | Alışveriş olayı ve sepet davranışı |
| Hane | 2.500 hane | Tahmin ve hedefleme evreni |
| Ürün | 92.353 ürün | Departman, marka, çeşitlilik ve yakıt ayrımı |
| Kampanya | 30 kampanya | Kampanya geçmişi ve aktiflik göstergeleri |
| Kupon kullanımı | 2.318 kayıt | Gerçek kupon kullanımı ve fırsat oranları |
| Demografi | 801 hane (%32,0) | Kontrollü ek deney; ana modele zorunlu değil |

Ham veri dosyaları büyük olduğu ve üçüncü taraf kullanım koşullarına tabi olabileceği için repoda tutulmayabilir. Analizi baştan çalıştırmak için aşağıdaki dosyalar `data/raw/` klasörüne yerleştirilmelidir:

```text
data/raw/
├── campaign_desc.csv
├── campaign_table.csv
├── coupon.csv
├── coupon_redempt.csv
├── hh_demographic.csv
├── product.csv
└── transaction_data.csv
```

## İş problemi

Model her snapshot gününde yakın zamanda aktif olan haneleri değerlendirir ve şu soruyu yanıtlar:

> Bu hane, snapshot gününden sonraki 56 gün boyunca hiç alışveriş yapmayacak mı?

Bir hanenin modele alınabilmesi için:

- Snapshot öncesindeki 182 günlük gözlem döneminde en az iki geçerli sepeti bulunmalıdır.
- İlk gözlenen alışverişinden itibaren en az 90 gün geçmiş olmalıdır.
- Snapshot'tan önceki son 84 günde en az bir geçerli sepeti bulunmalıdır.

Bu kurallar, henüz yeterli geçmişi olmayan veya zaten uzun süredir kayıp olan hanelerin hedef tanımını bozmasını azaltır.

## Proje akışı

Notebook'lar birbirinin çıktısını kullanır ve aşağıdaki sırayla çalıştırılmalıdır:

| Notebook | Amaç | Temel çıktı |
|---|---|---|
| [`01_data_preparation.ipynb`](01_data_preparation.ipynb) | Veri yükleme, kalite ve anahtar kontrolleri | Temiz Parquet tabloları |
| [`02_eda_transactions_and_products.ipynb`](02_eda_transactions_and_products.ipynb) | İşlem, ürün, sepet ve RFM EDA'sı | Feature mühendisliği kararları |
| [`03_eda_customers_and_demographics.ipynb`](03_eda_customers_and_demographics.ipynb) | Demografik kapsama ve temsil farkı | Demografi deney stratejisi |
| [`04_eda_campaigns_and_coupons.ipynb`](04_eda_campaigns_and_coupons.ipynb) | Kampanya ve kupon iş kuralları | Güvenilir kampanya/kupon feature tanımları |
| [`05_target_and_feature_engineering.ipynb`](05_target_and_feature_engineering.ipynb) | Hedef, snapshot paneli ve feature üretimi | Model tablosu ve feature setleri |
| [`06_modeling_and_evaluation.ipynb`](06_modeling_and_evaluation.ipynb) | Model seçimi, kalibrasyon, test ve açıklanabilirlik | Model bundle, tahminler ve metrikler |

## Veri hazırlama ve EDA'dan alınan temel kararlar

- Müşteri harcaması `customer_paid_value = sales_value + coupon_disc` olarak hesaplanmıştır. Bu değer perakendecinin işlem gelirinden farklıdır.
- `quantity` medyanı 1 iken maksimumu 89.638'dir. Yüksek değerlerin büyük bölümü benzin ürünlerinin farklı ölçü sistemiyle kaydedilmesinden kaynaklandığı için tüm veriye IQR kırpması uygulanmamıştır.
- Ham toplam `quantity` genel model feature'ı yapılmamış; yakıt davranışı harcama payı, sepet oranı ve son yakıt alışverişinden geçen gün ile temsil edilmiştir.
- Demografi yalnızca hanelerin %32'sini kapsar ve daha aktif hanelerde yoğunlaşır. Bu nedenle ana model tüm hanelerde davranışsal feature'larla kurulmuş, demografi yalnızca ablasyon deneyinde denenmiştir.
- Type A kampanyalarında haneye gönderilen 16 kuponun kimliği bilinmediği için gerçek kupon fırsatı paydası hesaplanamaz. Type B/C kupon fırsat oranları ayrı ele alınmıştır.
- Kampanya öncesi ve kampanya dönemi karşılaştırmaları tanımlayıcıdır; kontrol grubu bulunmadığından nedensel kampanya etkisi olarak yorumlanmamıştır.

## Hedef ve feature mühendisliği

Notebook 05 toplam **11.356 hane-snapshot satırı**, **2.430 benzersiz hane** ve **185 aday feature** üretmiştir.

| Split | Snapshot | Satır | Pasif hane-snapshot | Pasif oranı |
|---|---|---:|---:|---:|
| Train | 431, 487, 543 | 6.804 | 423 | %6,22 |
| Validation | 599 | 2.268 | 150 | %6,61 |
| Test | 655 | 2.284 | 165 | %7,22 |
| **Toplam** | 5 snapshot | **11.356** | **738** | **%6,50** |

### Feature setleri

| Feature seti | Sayı | İçerik |
|---|---:|---|
| `behavior_core_linear` | 37 | Logistic Regression için davranışsal çekirdek |
| `behavior_core` | 38 | Ağaç modelleri için çekirdek + `active_days` |
| `behavior_plus_context` | 76 | Davranış + kategori/marka/yakıt + kampanya/kupon + `has_demographic` |
| `demographic_experiment` | 85 | Context seti + anonim demografi kategorileri ve belirsizlik göstergeleri |

`active_days`, train döneminde `frequency_baskets` ile yaklaşık 0,99 ve `active_weeks` ile yaklaşık 0,97 Spearman korelasyonuna sahip olduğu için lineer feature setinden çıkarılmıştır. `frequency_baskets` alışveriş sayısını, `active_weeks` ise alışverişlerin zamana yayılımını temsil ettiği için ikisi korunmuştur.

<details>
<summary>Seçilen Logistic Regression modelindeki 37 feature'ı göster</summary>

**RFM ve aktivite**

`recency_days`, `frequency_baskets`, `customer_spend_total`, `active_weeks`, `spend_per_active_week`, `baskets_per_active_week`, `unique_products`, `unique_stores`

**İndirim ve kupon**

`discount_dependency`, `discounted_basket_rate`, `coupon_basket_rate`

**Sepet ve mağaza davranışı**

`median_basket_value`, `basket_value_cv`, `avg_products_per_basket`, `single_product_basket_rate`, `dominant_store_share`

**Alışveriş ritmi**

`median_purchase_gap`, `last_purchase_gap`, `recent_mean_purchase_gap_3`, `gap_acceleration`, `recency_to_typical_gap`, `gap_trend_last_5`

**Son 4 hafta**

`customer_spend_last_4w`, `baskets_last_4w`, `active_weeks_last_4w`

**Son 8 hafta**

`customer_spend_last_8w`, `baskets_last_8w`, `active_weeks_last_8w`, `discounted_basket_rate_last_8w`, `coupon_basket_rate_last_8w`

**Değişim ve 12 haftalık trend**

`spend_change_8w`, `basket_change_8w`, `avg_basket_value_change_8w`, `spend_slope_12w`, `basket_slope_12w`, `weekly_spend_cv_12w`, `weekly_basket_cv_12w`

</details>

### Veri sızıntısını önleme

- Bütün işlem tabanlı feature'lar yalnızca ilgili snapshot gününe kadar olan 182 günlük gözlemden üretilmiştir.
- Hedef dönemindeki alışveriş veya harcama bilgisi model girdisine alınmamıştır.
- Kaynak tablolar önce hane-snapshot düzeyinde özetlenmiş, ardından `one_to_one` kontrolüyle birleştirilmiştir.
- Departman şeması yalnızca train döneminin görebildiği veriden belirlenmiştir.
- Eksik değer doldurma ve ölçekleme, model pipeline'ı içinde yalnızca ilgili train fold'undan öğrenilmiştir.
- `household_key`, `snapshot_day`, split bilgisi ve hedef yardımcıları feature olarak kullanılmamıştır.

## Modelleme yaklaşımı

Aynı hane birden fazla snapshot'ta bulunabildiği için rastgele satır bazlı train/test ayrımı veya random KFold kullanılmamıştır.

Hiperparametre seçimi yalnızca train snapshot'larında iki ileri-zincirleme fold ile yapılmıştır:

1. `431 → 487`
2. `431 + 487 → 543`

Karşılaştırılan ana modeller:

- Dummy prior baseline
- Logistic Regression
- Random Forest
- Random Forest + context feature'ları
- Random Forest + context ve demografi feature'ları

| Model | Feature seti | Feature | Validation AP | Validation Brier | Top-%10 recall |
|---|---|---:|---:|---:|---:|
| **Logistic Regression** | `behavior_core_linear` | **37** | **0,314** | **0,0522** | 0,440 |
| Random Forest | `behavior_core` | 38 | 0,296 | 0,0529 | 0,467 |
| Random Forest | `behavior_plus_context` | 76 | 0,285 | 0,0532 | 0,460 |
| Random Forest | `demographic_experiment` | 85 | 0,280 | 0,0531 | 0,480 |
| Dummy prior | — | 0 | 0,066 | 0,0618 | 0,080 |

Validation Average Precision değeri en yüksek olduğu için `C=0.2`, `class_weight=None` ayarlı Logistic Regression seçilmiştir. Context ve demografi feature'ları Random Forest'ın genel sıralama performansını artırmamıştır.

## Olasılık kalibrasyonu ve karar kuralları

Seçilen modelin doğal olasılığı ile train out-of-fold tahminlerinden öğrenilen Platt scaling adayı validation döneminde karşılaştırılmıştır.

| Yöntem | Validation Brier | Validation ECE | Karar |
|---|---:|---:|---|
| `native_raw` | **0,05223** | **0,01104** | Seçildi |
| `platt_oof` | 0,05252 | 0,01388 | İyileştirme sağlamadı |

Ek kalibrasyon Brier skorunu iyileştirmediği için modelin doğal olasılığı korunmuştur. Isotonic regression, mevcut örneklemde aşırı uyum ve farklı skorları aynı olasılığa bağlama riski nedeniyle adaylara eklenmemiştir.

İki farklı karar mekanizması raporlanır:

- **F2 threshold:** Validation döneminde recall'a daha fazla ağırlık vererek seçilen `0.10205` eşiği.
- **Top-k:** Kampanya kapasitesi sabitse doğrudan en riskli %10 hanenin seçilmesi.

## Nihai test sonuçları

| Metrik | Test sonucu | %95 bootstrap güven aralığı |
|---|---:|---:|
| Average Precision | 0,273 | 0,227–0,337 |
| ROC-AUC | 0,851 | — |
| Brier score | 0,0584 | 0,0512–0,0657 |
| Precision | 0,259 | 0,219–0,297 |
| Recall | 0,703 | 0,636–0,769 |
| F2 | 0,523 | 0,469–0,575 |
| Top-%10 recall | 0,436 | 0,371–0,509 |
| Top-%10 precision | 0,314 | 0,258–0,376 |
| Top-%10 lift | 4,35× | 3,70–5,08× |

Testte en riskli **229 hane** seçildiğinde gerçekten pasifleşen **72 hane** yakalanmıştır. Başka bir ifadeyle yalnızca hanelerin %10'u hedeflenerek pasif hanelerin yaklaşık **%44'üne** ulaşılmıştır.

F2 eşiği hanelerin yaklaşık %19,6'sını riskli işaretlemiş ve pasif hanelerin yaklaşık %70'ini yakalamıştır. Daha yüksek recall'ın karşılığı, daha fazla yanlış pozitif ve daha geniş kampanya maliyetidir.

## En etkili feature'lar

Permutation importance sonuçlarına göre modelin en fazla yararlandığı feature'lar:

| Feature | Ortalama önem | Yorum |
|---|---:|---|
| `active_weeks` | 0,0940 | Aktivitenin zamana ne kadar yayıldığı |
| `recent_mean_purchase_gap_3` | 0,0575 | Son alışveriş aralıklarının seyrekleşmesi |
| `unique_products` | 0,0185 | Ürün çeşitliliği ve ilişki derinliği |
| `weekly_spend_cv_12w` | 0,0175 | Yakın dönem harcama oynaklığı |
| `basket_slope_12w` | 0,0117 | Sepet sıklığındaki yönlü trend |
| `discounted_basket_rate` | 0,0092 | İndirimli alışveriş davranışı |
| `active_weeks_last_4w` | 0,0091 | En yakın dönemde devam eden aktivite |
| `customer_spend_last_4w` | 0,0089 | En yakın dönem harcama seviyesi |

Sıfıra yakın veya negatif permutation importance, bir feature'ın kesinlikle gereksiz olduğunu tek başına göstermez. Korelasyonlu feature'lar birbirinin bilgisini taşıyabildiği için silme kararları yeni validation sonuçlarıyla test edilmelidir.


## Kurulum ve çalıştırma

### 1. Repoyu klonlayın

```bash
git clone https://github.com/ezgi-donmez/complete-journey-project.git
cd complete-journey-project
```

### 2. Sanal ortam oluşturun

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Gerekli paketleri kurun

```bash
pip install jupyter pandas numpy scipy scikit-learn matplotlib seaborn pyarrow joblib shap
```

Çalıştırılmış model metadata'sındaki ortam:

- Python 3.11.15
- pandas 3.0.3
- NumPy 2.4.6
- scikit-learn 1.8.0

### 4. Notebook'ları başlatın

```bash
jupyter lab
```

Notebook'ları `01` → `06` sırasıyla **Restart Kernel and Run All Cells** kullanarak çalıştırın.

# Customer 360 — Kullanım Kılavuzu

## 1. Bu uygulama ne yapar?

Bir marketin 2.284 müşterisini (veri setinde "hane" deniyor) inceler ve üç soruya cevap verir:

| Soru | Uygulamada nerede? |
|---|---|
| Hangi müşteri önümüzdeki 56 günde alışverişi bırakabilir? | Dashboard → risk puanı ve "En riskli 20" tablosu |
| Bu müşteri nasıl bir müşteri? | Dashboard → segment (Şampiyonlar / Sadık Müşteriler / Uzaklaşanlar) |
| Riski düşürmek için ne yapmalıyız? | Dashboard → senaryo analizi ve Chatbot |

> Müşteri adları sentetiktir. Veri seti anonim olduğu için adlar hane numarasından üretilir;
> gerçek kişilerle ilgisi yoktur. Hane numarası (#1108 gibi) gerçektir.

---

## 2. Kurulum (bir kez yapılır)

### 2.1 Python'u kurun

1. https://www.python.org/downloads/ adresinden **Python 3.11 veya 3.12** indirin.
2. Kurulumda **"Add python.exe to PATH"** kutusunu işaretleyin. Bu kutu işaretlenmezse
   aşağıdaki komutlar çalışmaz.
3. Kurulumu doğrulamak için Başlat menüsüne **PowerShell** yazıp açın ve şunu yazın:

```powershell
py -3 --version
```

`Python 3.12.x` gibi bir satır görüyorsanız tamamdır.

### 2.2 Projeyi bilgisayarınıza alın

- GitHub sayfasında yeşil **Code → Download ZIP** düğmesine basın, ZIP'i açın.

Klasörü örneğin `C:\Projeler\datagambit-g03-main` gibi bir yere koyun. Bundan sonraki tüm
komutlar bu klasörün içinde çalıştırılır. PowerShell'de klasöre girmek için:

```powershell
cd C:\Projeler\datagambit-g03-main
```

### 2.3 Gerekli paketleri kurun

```powershell
py -3 -m pip install -r requirements.txt
```

Bu komut internetten yaklaşık 300 MB paket indirir; birkaç dakika sürebilir. Sonunda hata
görmüyorsanız devam edin.

### 2.4 OpenAI anahtarını tanımlayın (chatbot için)

Chatbot, OpenAI'nin yapay zeka servisini kullanır. Bunun için bir **API anahtarı** gerekir.

1. https://platform.openai.com/api-keys adresinde anahtar oluşturun. `sk-` ile başlar.
2. Proje klasöründeki `.env.example` dosyasını kopyalayıp adını **`.env`** yapın.
3. `.env` dosyasını Not Defteri ile açın, iki satırı doldurun:

```
OPENAI_API_KEY=sk-buraya-kendi-anahtariniz
OPENAI_MODEL=gpt-5.4-mini
```

4. Anahtarın çalıştığını test edin:

```powershell
py -3 src/test_openai.py
```

`Model gpt-5.4-mini cevap verdi` satırını görüyorsanız hazır.

**Önemli kurallar**
- Anahtarı kimseyle paylaşmayın; WhatsApp, e-posta veya sohbete yapıştırmayın.
- `.env` dosyası repoya gönderilmez (`.gitignore` bunu engeller). `.env.example` ise gönderilir;
  içine gerçek anahtar yazmayın.
- Anahtar yanlışlıkla paylaşıldıysa OpenAI panelinden silip yenisini oluşturun.
- Anahtar olmadan da uygulama açılır; sadece Chatbot sekmesi uyarı gösterir.

### 2.5 Veri dosyalarını hazırlayın

Uygulama, notebook'ların ürettiği dosyaları okur. Repoda hazır geliyorlarsa bu adımı atlayın.
Eksikse (özellikle `data/processed/household_snapshot_model_table.parquet` boşsa) şunları çalıştırın:

```powershell
py -3 -m jupyter nbconvert --to notebook --execute 05_target_and_feature_engineering.ipynb --ExecutePreprocessor.kernel_name=python3 --output _calisti_05.ipynb
```

```powershell
py -3 src/segmentation.py
```

İlki özellik tablosunu (5–10 dakika), ikincisi müşteri segmentlerini (birkaç saniye) üretir.

---

## 3. Uygulamayı açma

Her seferinde tek komut:

```powershell
py -3 -m streamlit run app.py
```

Tarayıcı kendiliğinden açılır. Açılmazsa adres çubuğuna `http://localhost:8501` yazın.
Kapatmak için PowerShell penceresinde **Ctrl + C** basın.

---

## 4. Ekranı tanıyalım

**Sol menü**
- **Sayfa:** Dashboard ile Chatbot arasında geçiş.
- **Müşteri:** Liste risk sırasına göredir; en üstte en riskli müşteri. "Ad · #hane · risk %"
  biçimindedir. Bir müşteri seçince tüm sayfa ona göre güncellenir.
- **Risk eşiği:** Model 0,102 üstünü "riskli" sayar. En riskli %10 (229 müşteri) kampanya
  hedefidir.

**Dashboard, yukarıdan aşağıya**
1. **Dört özet kart:** toplam müşteri, en riskli %10'daki müşteri sayısı, ortalama risk,
   modelin başarısı (ROC-AUC 0,851; 1'e ne kadar yakınsa o kadar iyi).
2. **Risk dağılımı:** Çoğu müşteri sıfıra yakın; sağdaki uzun kuyruk riskli grup.
3. **En önemli 8 özellik:** Modelin en çok neye baktığı. İlk ikisi: kaç farklı haftada
   alışveriş yapıldığı ve son alışveriş aralıklarının uzayıp uzamadığı.
4. **Segmentler:** Üç grup ve her grubun ortalama riski.
5. **RFM grafiği:** Her nokta bir müşteri. Sağa gittikçe son alışveriş eskiyor, yukarı çıktıkça
   harcama artıyor. Büyük noktalar daha riskli.
6. **En riskli 20 müşteri:** Kampanya listesi için başlangıç noktası.
7. **Müşteri kartı:** Seçili müşterinin risk, sıra, segment ve RFM değerleri.


**SHAP sürücüleri** tablosu "model bu müşteride neye takıldı?" sorusunu cevaplar. "riski artıran"
satırlar puanı yukarı çeken davranışlardır. Yalnızca en riskli 25 müşteri için hesaplanmıştır.

---

## 5. Segmentler ne anlama geliyor?

| Segment | Kim? | Ne yapmalı? |
|---|---|---|
| **Şampiyonlar** (%42) | Neredeyse her gün gelen, en çok harcayan | Rahatsız etmeyin; teşekkür ve sadakat programı yeter |
| **Sadık Müşteriler** (%39) | Düzenli ama daha seyrek, sepeti güçlü | Sepet büyütme: çapraz satış, kişisel öneri |
| **Uzaklaşanlar** (%18) | Son alışverişi eski, sepet sayısı düşük | Geri kazanım: kupon, hatırlatma. Kampanya bütçesinin odağı |

Sayılar, dashboard'daki "Segment profilleri (tablo)" bölümünde açılabilir.

---

## 6. "Riski nasıl düşürürüz?" — senaryo analizi

Müşteri kartının altındaki tablo, modele "bu müşteri şu davranışı gösterseydi puan ne olurdu?"
diye sorar. Örnek: en riskli müşteri bu hafta bir kez alışveriş yapsa puanı 0,83'ten 0,27'ye iner.

Nasıl okunur?
- **Olasılık:** senaryodaki yeni risk puanı.
- **Değişim:** mevcut duruma göre fark (eksi = iyileşme).
- **Eşik altı:** ✅ varsa müşteri artık "riskli" sayılmıyor.

**Dikkat:** Bu bir simülasyondur. Modelin ne dediğini gösterir, kampanyanın gerçekten işe
yarayacağını kanıtlamaz. Gerçek etki, bir grup müşteriye kampanya uygulayıp diğerine
uygulamayarak (A/B testi) ölçülür.

---

## 7. Chatbot nasıl kullanılır?

1. Sol menüden **Chatbot** sayfasını seçin.
2. Hazır sorulardan birine tıklayın veya alttaki kutuya kendi sorunuzu yazıp Enter'a basın.
3. Cevap, seçili müşterinin verisine dayanır. Müşteriyi değiştirip aynı soruyu sorarsanız cevap
   değişir.

İyi sorular:
- "Bu müşteri neden riskli?"
- "Bu müşteriyi riskliden risksize nasıl geçiririz?"
- "Uzaklaşanlar segmentine hangi kampanya uygun?"
- "Modelin sınırlılıkları neler?"

Bilmeniz gerekenler:
- Chatbot hesap yapmaz; sayıları uygulama ona hazır verir, o açıklar. Verdiği kampanya önerileri
  **hipotezdir**, test edilmemiştir.
- "Sistem promptu (bağlam)" bölümünü açarsanız yapay zekaya tam olarak ne gönderildiğini görürsünüz.
- "Sohbeti temizle" geçmişi siler.
- Her soru küçük bir ücret oluşturur (mini modellerde bir soru genellikle bir kuruştan azdır).

---

## 8. Sık karşılaşılan sorunlar

| Belirti | Sebep | Çözüm |
|---|---|---|
| `py : The term 'py' is not recognized` | Python PATH'e eklenmemiş | Python'u kaldırıp "Add to PATH" işaretli kurun |
| `No module named streamlit` | Paketler kurulmadı | `py -3 -m pip install -r requirements.txt` |
| Chatbot "OPENAI_API_KEY bulunamadı" diyor | `.env` yok veya adı yanlış | Dosya adı tam olarak `.env` olmalı (`.env.txt` değil) |
| Chatbot "LLM çağrısı başarısız" diyor | Anahtar geçersiz, bakiye yok veya model adı yanlış | `py -3 src/test_openai.py` çalıştırıp mesajı okuyun |
| "Segmentasyon henüz çalıştırılmadı" uyarısı | Segment dosyası yok | `py -3 src/segmentation.py` |
| "Özellik tablosu yok" (senaryo analizi) | Model tablosu boş | Bölüm 2.5'teki nbconvert komutu |
| Sayfa boş kalıyor / dönüyor | İlk yükleme 20 MB veri okur | 10–20 saniye bekleyin, sonra sayfayı yenileyin |
| Türkçe karakterler bozuk görünüyor | PowerShell kodlaması | Komutun başına `$env:PYTHONUTF8=1;` ekleyin |

---

Sorunuz olursa proje ekibi: Selda Kırıkkanat, Ezgi Dönmez, Tuğçe Güven.

## Repo yapısı

```text
datagambit-g03/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda_transactions_and_products.ipynb
│   ├── 03_eda_customers_and_demographics.ipynb
│   ├── 04_eda_campaigns_and_coupons.ipynb
│   ├── 05_target_and_feature_engineering.ipynb
│   ├── 06_modeling_and_evaluation.ipynb
│
├── src/
│   ├── app.py
│   ├── context.py
│   ├── data.py
│   ├── llm.py
│   ├── names.py
│   ├── profile.py
│   ├── segmentation.py
│   ├── tools.py
│   ├── ui.py
│   └── whatif.py
│
│── about.md
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt


```

## Sınırlılıklar

- Pozitif sınıf oranı yalnızca %6–7 düzeyindedir; metrikler bootstrap güven aralıklarıyla birlikte değerlendirilmelidir.
- Aynı haneler farklı snapshot'larda tekrar eder. Tasarım mevcut müşterilerin zaman içinde yeniden puanlanmasına uygundur; tamamen yeni hanelere genelleme ayrıca test edilmemiştir.
- Test yalnızca 655. gün snapshot'ından oluşur ve önceki revizyonlarda görülmüştür. Bu nedenle sonuç tamamen bağımsız, hiç görülmemiş bir son holdout olarak sunulmamalıdır.
- Demografik veri yalnızca hanelerin %32'sini kapsar ve daha aktif hanelerde yoğunlaşır.
- Kampanya EDA'sı nedensel etki analizi değildir; Type A kupon fırsatı paydası veri kısıtı nedeniyle bilinmez.

## Lisans

Proje kodu [`LICENSE`](LICENSE) dosyasındaki MIT Lisansı ile sunulmaktadır. The Complete Journey veri setinin kullanım koşulları kendi veri sağlayıcısına aittir.

