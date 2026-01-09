import requests
from bs4 import BeautifulSoup
from pathlib import Path

def scrape_news():
    """Scraping berita dari UDINUS dan simpan ke file"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        site = "https://dinus.ac.id/category/news/page/"
        all_titles = []
        
        print("[START SCRAP]")
        
        for i in range(1, 6):  # Pages 1 - 5
            page = f"{site}{i}/"
            print(f"  📄 Mengambil data dari: {page}")
            
            try:
                reqget = requests.get(page, headers=headers, timeout=10)
                soup = BeautifulSoup(reqget.text, 'html.parser')
                articles = soup.select('h2.entry-title a')
                
                for article in articles:
                    title = article.text.strip()
                    all_titles.append(title)
                    
                print(f"[+] {len(articles)} page {i}")
                
            except Exception as e:
                print(f"error {page}: {e}")
                continue
        
        print(f"\n[+] total news titile: {len(all_titles)}")
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        docs_dir = project_root / "data" / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = docs_dir / "news-udi-multiple.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            book_title = "Berita terbaru udinus"
            formatted_output = f"""Judul: {book_title}
Penulis: hengker
Genre: berita
Penerbit: UDINUS
Kata Kunci: Berita, News, berita terbaru

Isi:
Buku ini berisi tentang berita terbaru yang diambil langsung dari website udinus

Topik utama yang dibahas mencakup:
"""
            
            # Tambahkan SEMUA judul dengan numbering
            for idx, title in enumerate(all_titles, 1):
                formatted_output += f"{idx}. {title}\n"
            
            formatted_output += "\n" + "="*60 + "\n\n"
            f.write(formatted_output)
        
        print(f"saved success: {output_path} with {len(all_titles)} topic")
        
        # RETURN tuple dengan success status dan file path
        return True, str(output_path)
        
    except Exception as e:
        print(f"errror scrape: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def run_preprocess():
    """Jalankan preprocess setelah scraping"""
    try:
        print("🔄 Menjalankan preprocess...")
        
        # Import preprocess dari src folder
        import sys
        from pathlib import Path
        
        # Tambahkan src folder ke sys.path
        current_file = Path(__file__).resolve()
        src_path = current_file.parent.parent.parent.parent / "src"
        
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        try:
            import preprocess
            preprocess.run_all()
            return True, "Preprocess berhasil"
        except ImportError as e:
            print(f"❌ Tidak bisa import preprocess dari {src_path}: {e}")
            return False, f"Import error: {e}"
            
    except Exception as e:
        print(f"❌ Error dalam run_preprocess: {e}")
        return False, str(e)

# Untuk testing standalone
if __name__ == "__main__":
    # Hanya jalankan jika di-run langsung
    success, path = scrape_news()
    if success:
        run_preprocess()