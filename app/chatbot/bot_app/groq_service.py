"""
Groq LLM Service for RAG (Retrieval-Augmented Generation)
Integrates Groq API to generate human-like responses
"""

from groq import Groq
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class GroqService:
    """Service class to interact with Groq API"""
    
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set in environment variables")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, user_query, context="", intent="", user_sentiment="", max_tokens=1024):
        """
        Generate a human-like response using Groq LLM
        
        Args:
            user_query (str): User's question/message
            context (str): Additional context from RAG/knowledge base
            intent (str): Detected intent from K-NN classifier
            user_sentiment (str): Detected sentiment (positive/negative/neutral)
            max_tokens (int): Maximum tokens for response
            
        Returns:
            str: Generated response or error message
        """
        if not self.client:
            return "⚠️ Groq API not configured. Please set GROQ_API_KEY in .env file."
        
        try:
            # Build system prompt for Indonesian academic chatbot - ADAPTIVE MODE
            system_prompt = """Kamu adalah asisten chatbot kampus resmi Universitas Dian Nuswantoro (UDINUS) yang ramah dan responsif.

ATURAN PENTING - PRIORITAS RESPONS:

A. RESPONS SOSIAL/PERCAKAPAN (Prioritas Tertinggi):
   1. Ucapan TERIMA KASIH:
      - Jawab: "Sama-sama! Senang bisa membantu 😊", "Terima kasih kembali!", dll
      - Gunakan variasi yang natural
   
   2. MAAF/KOMPLAIN/SENTIMENT NEGATIF:
      - Jika user kecewa/marah/komplain tentang jawaban:
        → "Mohon maaf atas ketidaknyamanannya 🙏"
        → "Saya akan berusaha lebih baik. Ada yang bisa saya bantu lagi?"
      - Tunjukkan empati dan minta maaf dengan tulus
   
   3. SALAM/GREETING:
      - "Halo! Selamat datang di UDINUS Bot 🎓"
      - "Ada yang bisa saya bantu hari ini?"

B. PERTANYAAN AKADEMIK (Dengan Konteks):
   Topik yang boleh dijawab dengan detail:
   - Pendaftaran mahasiswa baru (PMB)
   - Administrasi & pembayaran UKT/SPP
   - Akademik (KRS, jadwal kuliah, mata kuliah)
   - Informasi dosen dan fakultas
   - Beasiswa dan bantuan pendidikan
   - Fasilitas kampus (perpustakaan, lab, dll)
   - Teknis akun mahasiswa
   
   WAJIB gunakan informasi dari Konteks yang diberikan!

C. PERTANYAAN DI LUAR AKADEMIK (Tanpa Konteks Relevan):
   - Jika pertanyaan tentang: cuaca, olahraga, hiburan, politik, teknologi umum, dll
   - Jawab dengan sopan: "Maaf, saya hanya bisa membantu pertanyaan seputar akademik dan kampus UDINUS 🎓"
   - Arahkan: "Silakan tanya tentang: pendaftaran, UKT, jadwal kuliah, beasiswa, atau fasilitas kampus."
   - JANGAN jawab pertanyaan tersebut

D. TIDAK ADA INFORMASI DI DATABASE:
   - "Maaf, saya tidak memiliki informasi detail tentang hal tersebut saat ini."
   - "Silakan hubungi admin kampus di [kontak] atau cek website resmi UDINUS."

GAYA KOMUNIKASI:
- Bahasa Indonesia formal tapi ramah dan hangat
- Emoji secukupnya (😊, 📚, ✅, 🎓, 🙏, dll)
- Jawaban ringkas (2-4 kalimat)
- Responsif terhadap emosi user
- Tunjukkan empati jika user menunjukkan kekecewaan

PENTING: Baca konteks percakapan dengan baik. Jika user berterima kasih atau komplain, prioritaskan respons sosial dulu!
"""
            
            # Build user message with context and sentiment
            user_message = f"Pertanyaan mahasiswa: {user_query}"
            
            if user_sentiment:
                user_message = f"[Sentiment terdeteksi: {user_sentiment}]\n" + user_message
            
            if context:
                user_message += f"""\n\nKonteks informasi dari database kampus:\n{context}\n\n===\n\nJawab berdasarkan konteks di atas dengan mempertimbangkan sentiment user."""
            else:
                user_message += """\n\nTIDAK ADA konteks spesifik.\nJika ini percakapan sosial (terima kasih, maaf, salam) → jawab dengan ramah.\nJika pertanyaan akademik → arahkan untuk bertanya topik yang benar.\nJika pertanyaan non-akademik (cuaca, olahraga, dll) → tolak sopan."""
            
            if intent:
                user_message = f"[Topik terdeteksi: {intent}]\n\n" + user_message
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            response = chat_completion.choices[0].message.content
            return response.strip()
            
        except Exception as e:
            logger.error(f"Groq API Error: {str(e)}")
            return f"⚠️ Maaf, terjadi kesalahan saat memproses respons. Silakan coba lagi."
    
    def generate_contextual_response(self, user_query, clustered_docs=None, intent="", user_sentiment=""):
        """
        Generate response with clustered documents as context
        
        Args:
            user_query (str): User's question
            clustered_docs (list): List of relevant documents from clustering
            intent (str): Classified intent
            user_sentiment (str): User sentiment
            
        Returns:
            str: Contextual response
        """
        context = ""
        
        if clustered_docs and len(clustered_docs) > 0:
            # Take top 3 most relevant docs
            top_docs = clustered_docs[:3]
            context = "\n\n".join([f"- {doc}" for doc in top_docs])
        
        return self.generate_response(user_query, context=context, intent=intent, user_sentiment=user_sentiment)


# Initialize Groq service instance
groq_service = GroqService()
