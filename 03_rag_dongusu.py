"""
MODÜL 3 — TAM RAG DÖNGÜSÜ

Retrieve → Augment → Generate

Senaryo: Şirketimiz hakkında bir bilgi bankası var.
Kullanıcı soru soruyor, sistem doğru cevabı buluyor.
"""

import os
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Kurulum ───────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db_rag")
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

collection = chroma_client.get_or_create_collection(
    name="sirket_bilgileri",
    metadata={"hnsw:space": "cosine"},
)

# ── Bilgi Bankası ─────────────────────────────────────────────────────────────
dokumanlar = [
    "Şirketimizin adı CoilTech'tir. 2018 yılında İstanbul'da kurulmuştur.",
    "CoilTech, endüstriyel otomasyon ve yazılım geliştirme alanında hizmet vermektedir.",
    "Çalışan sayımız 45 kişidir. Ar-Ge departmanında 12 mühendis görev yapmaktadır.",
    "Yıllık izin hakkı her çalışan için 14 iş günüdür. İzin talebi en az 1 hafta önceden yapılmalıdır.",
    "Uzaktan çalışma politikamıza göre haftada 2 gün evden çalışma hakkı tanınmaktadır.",
    "Şirket merkezi Levent, İstanbul'dadır. Ayrıca Ankara'da bir ofisimiz bulunmaktadır.",
    "Maaş ödemeleri her ayın 5'inde yapılmaktadır.",
    "Yeni çalışanlar için 3 aylık deneme süresi uygulanmaktadır.",
]

if collection.count() == 0:
    vektorler = embedding_model.encode(dokumanlar).tolist()
    collection.add(
        ids=[f"doc_{i}" for i in range(len(dokumanlar))],
        documents=dokumanlar,
        embeddings=vektorler,
    )
    print(f"✅ {len(dokumanlar)} doküman bilgi bankasına eklendi\n")


# ── RAG Fonksiyonu ────────────────────────────────────────────────────────────
def rag_cevapla(soru: str, n_sonuc: int = 3) -> str:

    # 1. RETRIEVE — soruya en yakın dokümanları getir
    sorgu_vektor = embedding_model.encode([soru]).tolist()
    sonuclar = collection.query(query_embeddings=sorgu_vektor, n_results=n_sonuc)
    bulunan_dokumanlar = sonuclar["documents"][0]

    print(f"📥 Soru: {soru}")
    print(f"🔍 Bulunan ilgili dokümanlar:")
    for i, doc in enumerate(bulunan_dokumanlar, 1):
        print(f"   {i}. {doc}")
    print()

    # 2. AUGMENT — dokümanları prompt'a ekle
    baglam = "\n".join(f"- {doc}" for doc in bulunan_dokumanlar)

    prompt = f"""Aşağıdaki bilgileri kullanarak soruyu yanıtla.
Sadece verilen bilgilere dayan, bilgi dışına çıkma.
Bilgi bulamazsan "Bu konuda bilgim yok." de.

Bilgiler:
{baglam}

Soru: {soru}
Cevap:"""

    # 3. GENERATE — Claude cevap üretsin
    yanit = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return yanit.content[0].text


# ── Test ──────────────────────────────────────────────────────────────────────
sorular = [
    "Kaç gün yıllık iznim var?",
    "Şirket nerede kurulmuş?",
    "Haftada kaç gün evden çalışabilirim?",
    "Maaşlar ne zaman yatıyor?",
]

print("=" * 60)
for soru in sorular:
    cevap = rag_cevapla(soru)
    print(f"💬 Cevap: {cevap}")
    print("=" * 60)
