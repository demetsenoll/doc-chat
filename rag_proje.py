"""
RAG PROJESİ — Doküman yükle, soru sor, cevap al.

Desteklenen formatlar: .pdf, .txt
Kullanım:
  python rag_proje.py yukle dosya.pdf
  python rag_proje.py sohbet
  python rag_proje.py sor "Sorum nedir?"
  python rag_proje.py sifirla
"""

import os
import re
import sys
import textwrap
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

# ── Sabitler ──────────────────────────────────────────────────────────────────
MAX_CHUNK_BOYUTU = 1000  # Paragraf bu sınırı aşarsa cümle bazlı böl
N_SONUC = 6              # Kaç doküman getirilsin

# ── Başlatma ─────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db_proje")
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
collection = chroma_client.get_or_create_collection(
    name="dokumanlar",
    metadata={"hnsw:space": "cosine"},
)


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def metni_oku(dosya_yolu: str) -> str:
    """PDF veya TXT dosyasından metin çıkar."""
    if dosya_yolu.endswith(".pdf"):
        reader = PdfReader(dosya_yolu)
        return "\n".join(sayfa.extract_text() or "" for sayfa in reader.pages)
    elif dosya_yolu.endswith(".txt"):
        with open(dosya_yolu, encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Sadece .pdf ve .txt destekleniyor.")


def metni_parcala(metin: str) -> list[str]:
    """
    Önce çift satır sonu (boş satır) ile paragraf bloklarına böl.
    Blok MAX_CHUNK_BOYUTU'nu aşarsa satır sınırından kes.
    Böylece bir alarm kaydının başlık + çözüm adımları aynı chunk'ta kalır.
    """
    bloklar = [b.strip() for b in metin.split("\n\n") if b.strip()]
    parcalar = []
    for blok in bloklar:
        if len(blok) <= MAX_CHUNK_BOYUTU:
            parcalar.append(blok)
        else:
            # Çok uzun bloğu satır sınırından böl
            satirlar = blok.split("\n")
            parca = ""
            for satir in satirlar:
                if len(parca) + len(satir) > MAX_CHUNK_BOYUTU and parca:
                    parcalar.append(parca.strip())
                    parca = satir
                else:
                    parca += "\n" + satir
            if parca.strip():
                parcalar.append(parca.strip())
    return [p for p in parcalar if len(p) > 30]


# ── Komutlar ──────────────────────────────────────────────────────────────────
def yukle(dosya_yolu: str):
    """Dokümanı okur, parçalar, vektörleştirir ve ChromaDB'ye kaydeder."""
    if not os.path.exists(dosya_yolu):
        print(f"❌ Dosya bulunamadı: {dosya_yolu}")
        return

    print(f"📄 Dosya okunuyor: {dosya_yolu}")
    metin = metni_oku(dosya_yolu)
    print(f"   Toplam karakter: {len(metin):,}")

    parcalar = metni_parcala(metin)
    print(f"   Parça sayısı: {len(parcalar)}")

    print("🔢 Vektörler oluşturuluyor...")
    vektorler = embedding_model.encode(parcalar, show_progress_bar=True).tolist()

    # Mevcut dokümanları temizle, yenisini ekle
    mevcut_idler = collection.get()["ids"]
    if mevcut_idler:
        collection.delete(ids=mevcut_idler)

    dosya_adi = os.path.basename(dosya_yolu)
    collection.add(
        ids=[f"{dosya_adi}_parca_{i}" for i in range(len(parcalar))],
        documents=parcalar,
        embeddings=vektorler,
        metadatas=[{"kaynak": dosya_adi, "parca": i} for i in range(len(parcalar))],
    )

    print(f"✅ {len(parcalar)} parça başarıyla yüklendi. Artık soru sorabilirsin!\n")


def retrieve(soru: str, gecmis: list) -> list[str]:
    """Soruya ve geçmişe göre ilgili chunk'ları getirir."""
    # Bağlamlı sorularda ("bunu", "şunu") önceki soruyu da ekle
    baglamsiz_kelimeler = {"bu", "bunu", "şunu", "şu", "o", "onu", "peki", "ya"}
    ilk_kelime = soru.strip().split()[0].lower().rstrip("?")
    if ilk_kelime in baglamsiz_kelimeler and gecmis:
        onceki_soru = gecmis[-1]["content"].split("Soru:")[-1].strip()
        sorgu = f"{onceki_soru} {soru}"
    else:
        sorgu = soru

    alarm_kodu = None
    eslesme = re.search(r'\bE\d{3}\b', sorgu.upper())
    if eslesme:
        alarm_kodu = eslesme.group()

    sorgu_vektor = embedding_model.encode([sorgu]).tolist()
    sonuclar = collection.query(query_embeddings=sorgu_vektor, n_results=N_SONUC)
    parcalar = sonuclar["documents"][0]

    if alarm_kodu:
        tum_docs = collection.get()["documents"]
        for d in tum_docs:
            if alarm_kodu in d and d not in parcalar:
                parcalar.insert(0, d)
        parcalar = parcalar[:N_SONUC]

    return parcalar


def sor(soru: str, gecmis: list = None) -> str:
    """RAG döngüsü: soruyu al, ilgili parçaları bul, Claude ile cevapla."""
    if gecmis is None:
        gecmis = []

    if collection.count() == 0:
        print("❌ Henüz doküman yüklenmedi. Önce: python rag_proje.py yukle <dosya>")
        return ""

    parcalar = retrieve(soru, gecmis)
    baglam = "\n\n---\n\n".join(parcalar)

    sistem_prompt = f"""Bir teknik destek asistanısın. Yalnızca aşağıdaki belge parçalarına dayanarak cevap ver.
Belge dışında kendi bilgini kesinlikle kullanma.
Soru belgeyle ilgili değilse veya cevap belgede yoksa sadece "Bu konuda bilgim yok." de, başka bir şey ekleme.

Belge parçaları:
{baglam}"""

    # Geçmiş mesajlar + yeni soru
    mesajlar = gecmis + [{"role": "user", "content": soru}]

    print("\n📝 Cevap:")
    cevap = ""
    with claude_client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=sistem_prompt,
        messages=mesajlar,
    ) as stream:
        for metin in stream.text_stream:
            print(metin, end="", flush=True)
            cevap += metin

    print("\n")
    return cevap


def sohbet():
    """Sohbet modu — konuşma geçmişiyle interaktif sorgu."""
    if collection.count() == 0:
        print("❌ Henüz doküman yüklenmedi. Önce: python rag_proje.py yukle <dosya>")
        return

    print("💬 Sohbet modu başladı. Çıkmak için 'çıkış' yaz.\n")
    gecmis = []

    while True:
        try:
            soru = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz!")
            break

        if not soru:
            continue
        if soru.lower() in ("çıkış", "exit", "quit"):
            print("Görüşürüz!")
            break

        cevap = sor(soru, gecmis)

        # Geçmişe ekle (son 6 tur = 3 soru-cevap)
        gecmis.append({"role": "user", "content": soru})
        gecmis.append({"role": "assistant", "content": cevap})
        gecmis = gecmis[-6:]


def sifirla():
    """Veritabanını temizle."""
    mevcut_idler = collection.get()["ids"]
    if mevcut_idler:
        collection.delete(ids=mevcut_idler)
        print("🗑️  Veritabanı temizlendi.")
    else:
        print("ℹ️  Veritabanı zaten boş.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    komut = sys.argv[1]

    if komut == "yukle" and len(sys.argv) == 3:
        yukle(sys.argv[2])
    elif komut == "sohbet":
        sohbet()
    elif komut == "sor" and len(sys.argv) >= 3:
        sor(" ".join(sys.argv[2:]))
    elif komut == "sifirla":
        sifirla()
    else:
        print(__doc__)
