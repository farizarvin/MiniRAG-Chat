# src/generator.py
import re

class TemplateGenerator:
    def __init__(self):
        pass

    def _sentences(self, text):
        s = re.split(r'(?<=[\.!\?])\s+', text)
        return [sen.strip() for sen in s if sen.strip()]

    def score_sentence(self, sentence, query_terms):
        s = sentence.lower()
        return sum(1 for t in query_terms if t in s)

    def summarize(self, query, docs, top_sentences_per_doc=1):
        q_terms = query.lower().split()
        selected = []
        for doc in docs:
            sents = self._sentences(doc)
            scored = [(self.score_sentence(s, q_terms), s) for s in sents]
            scored = sorted(scored, key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0:
                selected.append(scored[0][1])
            elif sents:
                selected.append(sents[0])
        if not selected:
            return "Maaf, informasi terkait tidak ditemukan dalam korpus."
        # join up to 3 sentences
        return " ".join(selected[:3])
