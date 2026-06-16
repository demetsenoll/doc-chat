"""
MODÜL 2 — VECTOR DATABASE NEDİR?

Problem: 01_embedding_nedir.py'de vektörleri bir listede tuttuk.
1000 doküman varsa her sorgu için 1000 vektörü tek tek karşılaştırmak gerekir.
1 milyon dokümanda bu imkansız.

Çözüm: Vector Database (ChromaDB, Pinecone, Weaviate, Qdrant...)
- Vektörleri diske kaydeder
- Yakın komşu aramasını (ANN) çok hızlı yapar
- Metadata filtresi desteği (kategori, tarih, kaynak...)
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Adım 1: ChromaDB client aç ────────────────────────────────────────────────
# persistent_client → veriyi diske kaydeder (kapansa da kalır)
client = chromadb.PersistentClient(path="./chroma_db")

# Collection = bir "tablo" gibi düşün
collection = client.get_or_create_collection(
    name="bilgi_bankasi",
    metadata={"hnsw:space": "cosine"},  # cosine similarity kullan
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Türkçe destekli

# ── Adım 2: Dokümanları ekle ──────────────────────────────────────────────────
dokumanlar = [
    "Python, veri bilimi ve yapay zeka için en popüler programlama dilidir.",
    "JavaScript, web geliştirmede kullanılan temel bir dildir.",
    "RAG, büyük dil modellerini dış bilgi kaynakları ile güçlendiren bir tekniktir.",
    "ChromaDB, vektörleri saklayan ve hızlı arama yapan bir veritabanıdır.",
    "Elma, C vitamini açısından zengin bir meyvedir.",
    "Muz, potasyum kaynağı olan tropikal bir meyvedir.",
]

metadatalar = [
    {"kategori": "programlama"},
    {"kategori": "programlama"},
    {"kategori": "yapay_zeka"},
    {"kategori": "yapay_zeka"},
    {"kategori": "meyve"},
    {"kategori": "meyve"},
]

idler = [f"doc_{i}" for i in range(len(dokumanlar))]

# Eğer zaten eklenmiş ise yeniden ekleme
mevcut = collection.count()
if mevcut == 0:
    vektorler = model.encode(dokumanlar).tolist()
    collection.add(
        ids=idler,
        documents=dokumanlar,
        embeddings=vektorler,
        metadatas=metadatalar,
    )
    print(f"✅ {len(dokumanlar)} doküman eklendi\n")
else:
    print(f"ℹ️  Koleksiyonda zaten {mevcut} doküman var\n")


# ── Adım 3: Sorgu yap ─────────────────────────────────────────────────────────
print("=== SORGU 1: Genel arama ===")
sorgu = "Yapay zekada hangi araçlar kullanılır?"
sorgu_vektor = model.encode([sorgu]).tolist()

sonuclar = collection.query(
    query_embeddings=sorgu_vektor,
    n_results=3,
)

for i, (doc, dist) in enumerate(zip(sonuclar["documents"][0], sonuclar["distances"][0])):
    print(f"  {i+1}. [{1-dist:.3f}] {doc}")

print()

# ── Adım 4: Metadata filtresiyle sorgu ────────────────────────────────────────
print("=== SORGU 2: Sadece 'programlama' kategorisinde ara ===")
sorgu2 = "Hangi dil daha popüler?"
sorgu_vektor2 = model.encode([sorgu2]).tolist()

sonuclar2 = collection.query(
    query_embeddings=sorgu_vektor2,
    n_results=2,
    where={"kategori": "programlama"},  # filtre
)

for i, (doc, dist) in enumerate(zip(sonuclar2["documents"][0], sonuclar2["distances"][0])):
    print(f"  {i+1}. [{1-dist:.3f}] {doc}")

print()
print("=== ÖZET ===")
print(f"Toplam doküman: {collection.count()}")
print("Vector DB, aramayı kategoriye göre filtreleyerek daraltabiliyor.")
print("\n✅ Modül 2 tamam! Artık dokümanları nasıl saklayıp sorgulayacağını biliyorsun.")
print("   Sıradaki: Bu iki parçayı birleştirip LLM ile cevap üretelim → RAG!")
