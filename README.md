# doc-chat

Kendi dokümanlarına doğal dilde soru sorabildiğin RAG (Retrieval-Augmented Generation) asistanı. PDF veya TXT dosyası yükle, terminal üzerinden sohbet et.

## Nasıl Çalışır

```
Doküman yükle → Chunk'lara böl → Vektörleştir → ChromaDB'ye kaydet
                                                        ↓
Soru sor → Soruyu vektörleştir → En yakın chunk'ları getir → Claude'a ver → Cevap al
```

## Özellikler

- PDF ve TXT dosyası desteği
- Türkçe destekli embedding modeli (`paraphrase-multilingual-MiniLM-L12-v2`)
- Paragraf bazlı chunking — cümle ortasında bölünmez
- Hibrit arama — alarm kodu gibi teknik terimler için direkt eşleşme + semantik arama
- Sohbet modu — konuşma geçmişiyle bağlamlı sorular çalışır
- Streaming cevap — kelime kelime anlık gösterim
- Belge dışı sorulara cevap vermez

## Kurulum

```bash
git clone https://github.com/kullanici-adi/doc-chat.git
cd doc-chat

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`.env` dosyası oluştur:

```
ANTHROPIC_API_KEY=sk-ant-...
```

API key almak için: [console.anthropic.com](https://console.anthropic.com)

## Kullanım

```bash
# Doküman yükle
python rag_proje.py yukle belge.pdf

# Sohbet modu (önerilen)
python rag_proje.py sohbet

# Tek seferlik soru
python rag_proje.py sor "E001 alarmı nedir?"

# Veritabanını sıfırla
python rag_proje.py sifirla
```

## Öğrenme Modülleri

Projeyle birlikte gelen 3 modül, RAG'ın temel kavramlarını adım adım açıklar:

| Dosya | Konu |
|-------|------|
| `01_embedding_nedir.py` | Embedding ve benzerlik hesaplama |
| `02_vectordb_nedir.py` | ChromaDB ile vektör depolama ve sorgulama |
| `03_rag_dongusu.py` | Retrieve → Augment → Generate döngüsü |

```bash
python 01_embedding_nedir.py
python 02_vectordb_nedir.py
python 03_rag_dongusu.py
```

## Teknolojiler

- [Anthropic Claude API](https://anthropic.com) — dil modeli
- [ChromaDB](https://www.trychroma.com) — vektör veritabanı
- [Sentence Transformers](https://www.sbert.net) — embedding modeli
- [pypdf](https://pypdf.readthedocs.io) — PDF okuma

## Eksikler / Geliştirme Fikirleri

- [ ] Web arayüzü
- [ ] Çoklu doküman desteği
- [ ] Tamamen local LLM desteği (Ollama)
- [ ] Doküman güncelleme (sıfırlamadan ekleme)
