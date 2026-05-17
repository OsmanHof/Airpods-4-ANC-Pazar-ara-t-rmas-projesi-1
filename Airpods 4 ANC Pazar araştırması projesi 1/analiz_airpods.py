import pandas as pd
import re

# CSV dosyasını oku
df = pd.read_csv("yorumlar.csv", encoding="cp1254")

# Temizleme fonksiyonu
def temizle(metin):
    metin = str(metin)
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    return metin

# Yeni sütun oluştur
df["temiz"] = df["yorumlar:"].apply(temizle)

# Sonucu yazdır
print(df.head())

print("kod çalıştı")

print(len(df))

print(df.shape)

print(df["yorumlar:"].count())


print(df.head())   # ilk 5
print(df.tail())   # son 5


print(df.sample(5)[["yorumlar:", "temiz"]])



import re

def kontrol(metin):
    return bool(re.search(r'[^\w\s]', metin))

df["hala_karakter_var"] = df["temiz"].apply(kontrol)

print(df["hala_karakter_var"].value_counts())


df[df["hala_karakter_var"] == True]


tum_yorumlar = " ".join(df["temiz"])

print(tum_yorumlar[:500])  # ilk 500 karakteri göster

print("bundan sonra 8 adım")

print("BAŞ:", tum_yorumlar[:100])
print("ORTA:", tum_yorumlar[500:600])
print("SON:", tum_yorumlar[-100:])

print("bundan sonra 9 adım")

from collections import Counter

kelimeler = tum_yorumlar.split()

kelime_sayilari = Counter(kelimeler)

print(kelime_sayilari.most_common(20))

print("bundan sonra 10 adım")

stop_words = ["ve", "bir", "bu", "da", "de", "ama", "için", "çok", "ürün"]

kelimeler = [k for k in kelimeler if k not in stop_words]

from collections import Counter
kelime_sayilari = Counter(kelimeler)

print(kelime_sayilari.most_common(20))


from transformers import pipeline

print("\nYapay zeka yorumları okumaya başlıyor... Bu işlem biraz sürebilir.")

# 1. Adım: Türkçe bilen modelimizi çağırıyoruz
analizci = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

# 2. Adım: Duygu analizi yapacak fonksiyonu tanımlıyoruz
def duygu_analizi(metin):
    if not metin or pd.isna(metin):
        return "Nötr"
    metin = str(metin)[:512] # Modelin kapasitesini aşmamak için
    sonuc = analizci(metin)[0]
    
    # Model LABEL_1 derse Pozitif, LABEL_0 derse Negatif kabul ediyoruz
    if sonuc['label'] == "LABEL_1":
        return "Pozitif"
    elif sonuc['label'] == "LABEL_0":
        return "Negatif"
    else:
        return "Nötr"

# 3. Adım: 115 yorumun hepsine bunu uygula
df["duygu_sonucu"] = df["temiz"].apply(duygu_analizi)

# 4. Adım: Sonuçları ekrana yazdır
print("\n--- ANALİZ ÖZETİ ---")
print(df["duygu_sonucu"].value_counts())

print("\n--- ÖRNEK ANALİZLER ---")
print(df[["temiz", "duygu_sonucu"]].head(10))



from transformers import pipeline

print("\nYapay zeka yorumları okumaya başlıyor...")

# 1. Adım: Modeli yükle
analizci = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

# 2. Adım: Daha akıllı bir fonksiyon yazalım
def duygu_analizi(metin):
    if not metin or pd.isna(metin):
        return "Nötr"
    
    metin = str(metin)[:512]
    sonuc = analizci(metin)[0]
    
    # Modelin ne döndürdüğünü alıyoruz
    etiket = sonuc['label'].upper() 
    eminlik = sonuc['score'] # Yapay zekanın eminlik puanı

    # --- PROFESYONEL EŞİK ---
    # Eğer model %67'den az eminse 'Nötr' diyoruz
    if eminlik < 0.80:
        return "Nötr"
    
    if etiket in ["LABEL_1", "POSITIVE", "POS"]:
        return "Pozitif"
    elif etiket in ["LABEL_0", "NEGATIVE", "NEG"]:
        return "Negatif"
    else:
        return "Nötr"

# 3. Adım: Uygula
df["duygu_sonucu"] = df["temiz"].apply(duygu_analizi)

# 4. Adım: Sonuçları yazdır
print("\n--- ANALİZ ÖZETİ ---")
print(df["duygu_sonucu"].value_counts())

print("\n--- İLK 10 YORUM ANALİZİ ---")
print(df[["temiz", "duygu_sonucu"]].head(10))


# Tüm satırları göster ayarını aktif et (Pandas hepsini yazdırsın diye)
pd.set_option('display.max_rows', None)

print("\n" + "!"*20 + " TÜM YORUMLARIN LİSTESİ " + "!"*20)
print(df[["temiz", "duygu_sonucu"]])


# Analiz sonuçlarını bilgisayarına bir Excel (CSV) dosyası olarak kaydeder
df.to_csv("airpods_analiz_kontrol.csv", index=False, encoding="utf-16")
print("\nAnaliz sonuçları 'airpods_analiz_kontrol.csv' adıyla kaydedildi. Excel ile açabilirsin!")


# Sonuçları masaüstünde açabileceğin bir dosyaya kaydeder
df[["temiz", "duygu_sonucu"]].to_csv("analiz_sonuclari.csv", index=False, encoding="utf-16")
print("\nİşlem tamam! 'analiz_sonuclari.csv' dosyası oluşturuldu. Excel veya Not Defteri ile açıp hepsini okuyabilirsin.")



