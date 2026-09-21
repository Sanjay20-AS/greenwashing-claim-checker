from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_PATH = Path(__file__).parent / "rubric_kb.md"

class RubricRAG:
    def __init__(self, kb_path=str(KB_PATH), model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.kb_path = Path(kb_path)
        self.documents = self._load_sections()
        self.encoder = SentenceTransformer(model_name)
        embeddings = self.encoder.encode(self.documents, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _load_sections(self):
        text = self.kb_path.read_text(encoding="utf-8")
        sections, current = [], []
        for line in text.splitlines():
            if line.startswith("## ") and current:
                section = "\n".join(current).strip()
                if section: sections.append(section)
                current = []
            current.append(line)
        if current:
            section = "\n".join(current).strip()
            if section: sections.append(section)
        return sections

    def retrieve(self, query, top_k=4):
        if not query.strip(): return []
        q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q, k)
        return [{"score": float(s), "text": self.documents[int(i)]} for s, i in zip(scores[0], indices[0])]

    def context(self, query, top_k=4):
        results = self.retrieve(query, top_k)
        return "\n\n".join(
            f"[Rubric Source {i} | similarity={r['score']:.3f}]\n{r['text']}"
            for i, r in enumerate(results, 1)
        ) or "No rubric context retrieved."

if __name__ == "__main__":
    print(RubricRAG().context("carbon neutral target with no baseline or verification"))
