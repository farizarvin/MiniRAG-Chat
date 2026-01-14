from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time
import requests
from bs4 import BeautifulSoup

# --- IMPORT MODULE APLIKASI ---
# Pastikan file clustering_search.py dan training_logic.py sudah ada
try:
    from .clustering_search import cari_dokumen_relevan, reload_model_otomatis, data_dokumen
except ImportError:
    cari_dokumen_relevan = None
    reload_model_otomatis = None
    data_dokumen = {}

try:
    from .training_logic import latih_model_sekarang
except ImportError:
    latih_model_sekarang = None

from .utils import prediksi_niat_user 

# Import Summarizer (Feature Based)
try:
    from .summarizer import ringkas_dokumen, ringkas_teks_sederhana
except ImportError:
    ringkas_dokumen = None
    ringkas_teks_sederhana = None

# Import Sentiment Analyzer
try:
    from .sentiment_analyzer import predict_sentiment
except ImportError:
    predict_sentiment = None

# Import Groq LLM Service
try:
    from .groq_service import groq_service
except ImportError:
    groq_service = None

# ==============================================================================
# 1. FUNGSI HELPER (SCRAPER & LAINNYA)
# ==============================================================================

def ambil_berita_terbaru():
    """Mengambil berita dari dinus.ac.id (Versi aman di dalam views)"""
    url = "https://dinus.ac.id/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=30)  # Increased timeout
        soup = BeautifulSoup(response.text, 'html.parser')
        berita_list = []
        
        # Logika scraping
        articles = soup.find_all('div', class_='news-item', limit=5)
        if not articles:
             articles = soup.select('div.content-news h3 a')[:5]

        for item in articles:
            judul = item.get_text(strip=True)
            if judul: berita_list.append(judul)
            
        return berita_list if berita_list else ["Tidak ada berita baru ditemukan dari situs"]
    except requests.Timeout:
        return ["Koneksi ke dinus.ac.id terlalu lama (timeout). Coba lagi nanti."]
    except requests.RequestException as e:
        return [f"Tidak dapat terhubung ke website: {str(e)[:50]}"]
    except Exception as e:
        return [f"Error: {str(e)[:50]}"]

def _is_greeting(text):
    """Check if message is a greeting"""
    greetings = ['halo', 'hai', 'hello', 'hi', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'assalamualaikum', 'permisi']
    text_lower = text.lower()
    return any(greeting in text_lower for greeting in greetings) and len(text.split()) <= 5

def _is_thanks(text):
    """Check if message is a thank you"""
    thanks = ['terima kasih', 'terimakasih', 'makasih', 'thank you', 'thanks', 'thx']
    text_lower = text.lower()
    return any(thank in text_lower for thank in thanks)

def _is_profanity(text):
    """Check if message contains profanity"""
    profanity_words = ['anjing', 'bangsat', 'tolol', 'bodoh', 'goblok', 'kampret']
    text_lower = text.lower()
    return any(word in text_lower for word in profanity_words)

# ==============================================================================
# 2. VIEW UTAMA (HALAMAN WEB)
# ==============================================================================

def index(request):
    return render(request, 'bot_app/index.html')

# ==============================================================================
# 3. API ENDPOINTS (DIPANGGIL OLEH JAVASCRIPT)
# ==============================================================================

# --- API SCRAPER (Untuk Sidebar 'Cek Berita') ---
@csrf_exempt
def scrape_api(request):
    if request.method == 'POST':
        berita = ambil_berita_terbaru()
        html_response = "<b>[Hasil Scraping Dinus.ac.id]</b><br>"
        for i, b in enumerate(berita, 1):
            html_response += f"{i}. {b}<br>"
        return JsonResponse({'status': 'success', 'msg': html_response})     
    return JsonResponse({'status': 'error', 'msg': 'Invalid request'})

# --- API TRAINING (Untuk Sidebar 'Latih Ulang Bot') ---
@csrf_exempt
def train_api(request):
    """API untuk memicu proses training ulang"""
    if request.method == 'POST':
        if not latih_model_sekarang:
            return JsonResponse({'status': 'error', 'msg': 'Modul training_logic.py tidak ditemukan!'})

        # 1. Jalankan Training (Membuat file .pkl baru)
        sukses, pesan = latih_model_sekarang()
        
        if sukses:
            # 2. Jika sukses, reload model di RAM agar langsung update
            if reload_model_otomatis:
                reload_model_otomatis()
            return JsonResponse({'status': 'success', 'msg': pesan})
        else:
            return JsonResponse({'status': 'error', 'msg': pesan})
            
    return JsonResponse({'status': 'error', 'msg': 'Invalid request'})

# ==============================================================================
# 4. LOGIKA UTAMA CHATBOT (GET RESPONSE)
# ==============================================================================

@csrf_exempt
def get_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('msg', '')
            use_summarization = data.get('use_summarization', True)  # Default: enabled
            response_text = ""
            
            # --- DETECT USER SENTIMENT EARLY ---
            user_sentiment = ""
            if predict_sentiment:
                try:
                    sentiment, confidence = predict_sentiment(user_input)
                    user_sentiment = sentiment  # positive, negative, neutral
                except:
                    user_sentiment = ""
            
            # --- MAPPING INTENT KE CLUSTER ID ---
            # Sesuaikan angka ini dengan hasil training (output latih_dari_csv.py)
            INTENT_TO_CLUSTER = {
                'ADMINISTRASI_KEUANGAN': 2,
                'AKADEMIK_KRS': 5,
                'INFORMASI_DOSEN': 0,
                'BEASISWA': 6,
                'PENDAFTARAN_PMB': 1,
                'FASILITAS_KAMPUS': 3,
                'TEKNIS_AKUN': 4
            }

            # Helper functions for polite responses
            def _is_greeting(text):
                greetings = ["halo", "hai", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "hi", "p"]
                return any(word in text.lower().split() for word in greetings)

            def _is_thanks(text):
                thanks_keywords = ["terima kasih", "makasih", "thanks", "thank you"]
                return any(word in text.lower() for word in thanks_keywords)

            def _is_profanity(text):
                profane_words = ["anjing", "babi", "kontol", "memek", "bangsat", "asu", "jancok", "fuck", "shit"] # Add more as needed
                return any(word in text.lower() for word in profane_words)

            # --- A. FITUR SCRAPER (Via Chat) ---
            if "berita" in user_input.lower() or "news" in user_input.lower():
                response_text = "<b>[Fitur: Web Mining]</b><br>Sedang mencari berita...<br>"
                berita = ambil_berita_terbaru()
                for i, b in enumerate(berita, 1):
                    response_text += f"{i}. {b}<br>"
                return JsonResponse({'response': response_text})
            
            # --- B. POLITE RESPONSE TEMPLATES (Non-contextual messages) ---
            elif _is_greeting(user_input):
                import random
                greetings = [
                    "Halo! Selamat datang di KampusBot AI 🎓 Ada yang bisa saya bantu?",
                    "Hai! Terima kasih sudah menghubungi kami. Silakan tanyakan apa saja!",
                    "Selamat datang! Saya siap membantu Anda dengan informasi kampus."
                ]
                response_text = random.choice(greetings)
            
            elif _is_thanks(user_input):
                import random
                thanks_responses = [
                    "Sama-sama! Senang bisa membantu 😊",
                    "Terima kasih kembali! Jangan ragu untuk bertanya lagi.",
                    "Dengan senang hati! Semoga informasinya bermanfaat."
                ]
                response_text = random.choice(thanks_responses)
            
            elif _is_profanity(user_input):
                response_text = "Mohon maaf, saya tidak dapat merespons pesan yang tidak sopan. Mari kita berkomunikasi dengan baik-baik 🙏"
            
            # --- C. FITUR SUMMARIZER (Manual Command) ---
            elif user_input.lower().startswith("ringkas"):
                teks = user_input.replace("ringkas", "").strip()
                if ringkas_teks_sederhana:
                    hasil = ringkas_teks_sederhana(teks, max_sentences=3)

            # --- C. FITUR PENCARIAN BEBAS (SEARCH) ---
            # Jika user mengetik "Cari info..."
            if user_input.lower().startswith("cari") or user_input.lower().startswith("search"):
                keyword = user_input.replace("cari", "").replace("search", "").strip()
                
                if cari_dokumen_relevan:
                    cluster_id, docs = cari_dokumen_relevan(keyword)
                    if cluster_id is not None:
                        # Ringkas dokumen yang ditemukan
                        if ringkas_dokumen and len(docs) > 0:
                            ringkasan = ringkas_dokumen(docs, max_sentences=3)
                            response_text = f"<b>[Mode Pencarian - Cluster {cluster_id}]</b><br>Kata kunci: <i>'{keyword}'</i><br><br>"
                            response_text += f"<b>📄 Ringkasan (3 Kalimat Utama):</b><br>{ringkasan}"
                        else:
                            # Fallback jika summarizer tidak aktif
                            response_text = f"<b>[Mode Pencarian]</b><br>Kata kunci '{keyword}' relevan dengan <b>Cluster {cluster_id}</b>.<br><br>"
                            for d in docs[:3]:
                                response_text += f"🔍 {d}<br><br>"
                    else:
                        response_text = f"<b>[Mode Pencarian]</b><br>{docs[0]}"
                else:
                    response_text = "Modul Clustering belum siap."
            
            # --- D. FITUR PERCAKAPAN BIASA (INTENT) ---
            else:
                niat = prediksi_niat_user(user_input)
                # Debugging di terminal
                print(f"DEBUG: Input='{user_input}' -> Niat='{niat}'")

                if niat in INTENT_TO_CLUSTER:
                    target_cluster = INTENT_TO_CLUSTER[niat]
                    
                    # Ambil dokumen dari variabel global yang di-import
                    docs = data_dokumen.get(target_cluster, [])

                    # --- GROQ LLM INTEGRATION ---
                    if groq_service and groq_service.client:
                        # Use Groq to generate human-like response with sentiment awareness
                        groq_response = groq_service.generate_contextual_response(
                            user_query=user_input,
                            clustered_docs=docs,
                            intent=niat,
                            user_sentiment=user_sentiment
                        )
                        
                        response_text = f"<b>[Topik: {niat}]</b><br><br>"
                        response_text += f"🤖 {groq_response}"
                        
                        # Add feature keywords if available
                        if use_summarization and ringkas_dokumen and docs:
                            _, features = ringkas_dokumen(docs, max_sentences=3, return_features=True)
                            if features:
                                top_keywords = ", ".join([f.replace("_", " ") for f, score in features[:5]])
                                response_text += f"<br><br><i>🔑 Kata kunci: {top_keywords}</i>"
                    
                    else:
                        # Fallback to original method if Groq not available
                        response_text = f"<b>[Topik: {niat}]</b><br>"
                        if docs:
                            # Cek apakah user ingin ringkasan atau dokumen penuh
                            if use_summarization and ringkas_dokumen:
                                # Ringkas dokumen otomatis dengan ekstraksi fitur
                                ringkasan, features = ringkas_dokumen(docs, max_sentences=3, return_features=True)
                                
                                # Tampilkan kata kunci yang terdeteksi (Feature Based)
                                if features:
                                    top_keywords = ", ".join([f.replace("_", " ") for f, score in features[:5]])
                                    response_text += f"<i>🔑 Kata kunci terdeteksi: {top_keywords}</i><br><br>"
                                
                                response_text += f"<b>📄 Ringkasan Informasi (3 Kalimat Utama):</b><br>{ringkasan}"
                            else:
                                # Fallback: tampilkan dokumen lengkap tanpa summarize
                                response_text += "<b>📚 Informasi Lengkap:</b><br><br>"
                                for d in docs[:5]:
                                    response_text += f"📘 {d}<br><br>"
                        else:
                            response_text += "Belum ada dokumen untuk topik ini di database."
                else:
                    # Unknown intent - try Groq for general response with sentiment awareness
                    if groq_service and groq_service.client:
                        groq_response = groq_service.generate_response(
                            user_query=user_input,
                            context="",
                            intent="Umum",
                            user_sentiment=user_sentiment
                        )
                        response_text = f"🤖 {groq_response}"
                    else:
                        response_text = f"Maaf, saya kurang paham. (Kategori terdeteksi: {niat})"

            time.sleep(0.5) # Efek mengetik alami
            return JsonResponse({'response': response_text})
        except Exception as e:
            return JsonResponse({'response': f'Error: {str(e)}'})
    return JsonResponse({'response': 'Hanya POST yang diterima.'})


# ==============================================================================
# 6. API SENTIMENT ANALYSIS
# ==============================================================================
@csrf_exempt
def sentiment_api(request):
    """API endpoint for sentiment analysis"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text.strip():
                return JsonResponse({'error': 'Text cannot be empty'}, status=400)
            
            if predict_sentiment is None:
                return JsonResponse({
                    'error': 'Sentiment analyzer not available. Train model first.',
                    'sentiment': 'unknown',
                    'confidence': {}
                })
            
            # Predict sentiment
            sentiment, confidence = predict_sentiment(text)
            
            return JsonResponse({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'status': 'success'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e), 'status': 'error'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)