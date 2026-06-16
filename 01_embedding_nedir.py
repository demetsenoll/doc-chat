"""
MODÜL 1 — EMBEDDİNG NEDİR?

Soru: Bir bilgisayar "elma" ile "meyve" nin benzer olduğunu nasıl anlar?
Cevap: Her kelimeyi/cümleyi sayı dizisine (vektör) çevirir.
Bu işleme "embedding" denir.

Embedding = anlam bilgiyi taşıyan sayı listesi.
Benzer anlamlar → birbirine yakın vektörler.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Model: cümleleri vektöre çeviren hazır bir sinir ağı
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 boyutlu vektör üretir

# ── Adım 1: Cümleleri vektöre çevir ──────────────────────────────────────────
cumleler = [
    "Elma bir meyvedir.",
    "Muz sarı bir meyvedir.",
    "Python bir programlama dilidir.",
    "JavaScript web geliştirmede kullanılır.",
]

vektorler = model.encode(cumleler)

print("=== EMBEDDING BOYUTU ===")
print(f"Her cümle → {vektorler.shape[1]} sayıdan oluşan bir vektör")
print(f"Toplam {vektorler.shape[0]} cümle encode edildi\n")

print("=== İLK VEKTÖRün İLK 8 SAYISI ===")
print(f"'{cumleler[0]}' → {vektorler[0][:8].round(3)}\n")


# ── Adım 2: Benzerlik hesapla (cosine similarity) ────────────────────────────
def benzerlik(v1, v2):
    """İki vektör ne kadar benzer? 1.0 = aynı, 0.0 = ilgisiz, -1.0 = zıt"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


print("=== BENZERLİK KARŞILAŞTIRMASI ===")
cifler = [
    (0, 1),  # elma ↔ muz
    (0, 2),  # elma ↔ python
    (2, 3),  # python ↔ javascript
]

for i, j in cifler:
    skor = benzerlik(vektorler[i], vektorler[j])
    print(f"'{cumleler[i]}' ↔ '{cumleler[j]}'")
    print(f"  Benzerlik: {skor:.3f}\n")


# ── Adım 3: Sorguya en benzer cümleyi bul ────────────────────────────────────
print("=== SORGU → EN BENZER CÜMLE ===")
sorgu = "Hangi yazılım dilleri vardır?"
sorgu_vektor = model.encode([sorgu])[0]

skorlar = [(cumleler[i], benzerlik(sorgu_vektor, vektorler[i])) for i in range(len(cumleler))]
skorlar.sort(key=lambda x: x[1], reverse=True)

print(f"Sorgu: '{sorgu}'\n")
for cumle, skor in skorlar:
    print(f"  {skor:.3f} — {cumle}")

print("\n✅ Tebrikler! Şimdi RAG'ın kalbini anladın.")
print("   Bir soru geldiğinde → vektöre çevir → en yakın belgeleri bul → LLM'e ver")
