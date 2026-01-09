from django.shortcuts import render, redirect
from django.contrib import messages
from .rag_engine import rag_pipeline
from .scrape import scrape_news, run_preprocess

def index(request):
    """
    View utama untuk halaman Asisten AI Perpustakaan.
    Menangani input pertanyaan dari form, menjalankan pipeline RAG,
    dan menampilkan jawaban serta sumber dokumen.
    """
    answer = ""
    citations = []
    question = ""
    
    if request.method == "POST":
        if 'update_news' in request.POST:
            try:
                print("\n" + "="*50)
                print("🚀 MEMULAI PROSES UPDATE DATA")
                print("="*50)
                
                # STEP 1: Scraping berita
                print("\n📡 STEP 1: Scraping berita dari UDINUS...")
                scrape_success, file_path = scrape_news()
                
                if not scrape_success:
                    messages.error(request, "❌ Gagal scraping data")
                    return redirect('index')
                
                print(f"✅ Scraping selesai. File: {file_path}")
                
                # STEP 2: Jalankan preprocess
                print("\n⚙️  STEP 2: Menjalankan preprocess...")
                preprocess_success, preprocess_msg = run_preprocess()
                
                if preprocess_success:
                    messages.success(request, "✅ Data berhasil di-update dan diproses!")
                    print("🎉 PROSES UPDATE SELESAI")
                else:
                    messages.warning(request, f"⚠️  Data di-update tapi preprocess gagal: {preprocess_msg}")
                    print("⚠️  Preprocess ada masalah")
                
                print("="*50 + "\n")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                messages.error(request, f"❌ Error: {str(e)}")

        else:
            question = request.POST.get("question", "").strip()
            if question:
                answer, citations = rag_pipeline(question, k=3)
            
    citation_text = "\n".join([
        f"- {c['title']} ({c['source']}) • skor: {c['score']}"
        for c in citations
    ]) if citations else "-"

    return render(request, "index.html", {
        "answer": answer or "Belum ada jawaban. Silakan ajukan pertanyaan di atas.",
        "citations": citation_text,
        "question": question
    })
