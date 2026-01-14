import requests
from bs4 import BeautifulSoup

def ambil_berita_terbaru():
    url = "https://dinus.ac.id/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Mengakses {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        berita_list = []

        articles = soup.find_all('div', class_='news-item', limit=5)
        if not articles:
             articles = soup.select('div.content-news h3 a')[:5]

        for item in articles:
            judul = item.get_text(strip=True)
            if judul:
                berita_list.append(judul)

        if not berita_list:
            return ["Selamat Datang di Universitas Dian Nuswantoro", "Info PMB tersedia di pmb.dinus.ac.id"]

        return berita_list

    except Exception as e:
        print(f"Error Scraper: {e}")
        return [f"Gagal koneksi: {str(e)}"]