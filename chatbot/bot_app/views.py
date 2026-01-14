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

# Import Summarizer (Opsional)
try:
    from .nlp_modules import soal3_ringkasan as bot_ringkas
except ImportError:
    bot_ringkas = None

# ==============================================================================
# 1. FUNGSI HELPER (SCRAPER & LAINNYA)
# ==============================================================================

def ambil_berita_terbaru():
    """Mengambil berita dari dinus.ac.id (Versi aman di dalam views)"""
    url = "https://dinus.ac.id/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        berita_list = []
        
        # Logika scraping
        articles = soup.find_all('div', class_='news-item', limit=5)
        if not articles:
             articles = soup.select('div.content-news h3 a')[:5]

        for item in articles:
            judul = item.get_text(strip=True)
            if judul: berita_list.append(judul)
            
        return berita_list if berita_list else ["Selamat Datang di Udinus", "Cek pmb.dinus.ac.id"]
    except Exception as e:
        return [f"Gagal koneksi: {str(e)}"]

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
            response_text = ""
            
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

            # --- A. FITUR SCRAPER (Via Chat) ---
            if "berita" in user_input.lower() or "news" in user_input.lower():
                response_text = "<b>[Fitur: Web Mining]</b><br>Sedang mencari berita...<br>"
                berita = ambil_berita_terbaru()
                for i, b in enumerate(berita, 1):
                    response_text += f"{i}. {b}<br>"
                return JsonResponse({'response': response_text})

            # --- B. FITUR SUMMARIZER ---
            if user_input.lower().startswith("ringkas"):
                teks = user_input.replace("ringkas", "").strip()
                if bot_ringkas:
                    hasil = bot_ringkas.buat_ringkasan(teks)
                    return JsonResponse({'response': f"<b>[Ringkasan]</b>:<br>{hasil}"})
                else:
                    return JsonResponse({'response': "Modul Ringkasan belum aktif."})

            # --- C. FITUR PENCARIAN BEBAS (SEARCH) ---
            # Jika user mengetik "Cari info..."
            if user_input.lower().startswith("cari") or user_input.lower().startswith("search"):
                keyword = user_input.replace("cari", "").replace("search", "").strip()
                
                if cari_dokumen_relevan:
                    cluster_id, docs = cari_dokumen_relevan(keyword)
                    if cluster_id is not None:
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

                    response_text = f"<b>[Topik: {niat}]</b><br>Berikut informasi terkait:<br><br>"
                    if docs:
                        for d in docs[:3]:
                            response_text += f"📘 {d}<br><br>"
                    else:
                        response_text += "Belum ada dokumen untuk topik ini di database."
                else:
                    response_text = f"Maaf, saya kurang paham. (Kategori terdeteksi: {niat})"

            time.sleep(0.5) # Efek mengetik alami
            return JsonResponse({'response': response_text})

        except Exception as e:
            print(f"ERROR SERVER: {e}")
            return JsonResponse({'response': "Terjadi kesalahan pada server."})
            
    return JsonResponse({'response': 'Invalid request'})