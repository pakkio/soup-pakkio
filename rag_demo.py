"""Tiny RAG demo: retrieve chunks of Soup's README/commands docs, answer with base Gemma 3 1B."""
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOCS = [Path.home() / "w/Soup/README.md", Path.home() / "w/Soup/docs/commands.md"]
MODEL = "google/gemma-3-1b-it"
STOP = set("the a an of to in is it and or for with on how do i what which can you me my "
           "does this that are be use using soup".split())


HOWTO = re.compile(r"\b(command|how (do|can|to|would)|which .*versions?|version|install|"
                   r"export|serve|resume|merge|push)\b", re.IGNORECASE)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9_\-\.]+", text.lower()) if t not in STOP]


def load_chunks():
    chunks = []
    for path in DOCS:
        buf = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if path.name == "commands.md":
                if line.startswith("soup ") or line.startswith("POST "):
                    chunks.append(line.strip()[:400])
                continue
            if line.strip():
                buf.append(line.strip())
            elif buf:
                chunks.append(" ".join(buf)[:700])
                buf = []
        if buf:
            chunks.append(" ".join(buf)[:700])
    return [c for c in chunks if len(c) > 25 and "<img" not in c and "<a href" not in c]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query, k=4):
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (
                        tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            scores.append((s, i))
        return [i for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def main():
    chunks = load_chunks()
    index = BM25(chunks)
    # Always-on overview: generic questions ("what is Soup?") have no keywords to retrieve on.
    pinned = [i for i, c in enumerate(chunks)
              if c.startswith("Soup turns the pain") or "Zero SSH" in c]
    print(f"[indexed {len(chunks)} chunks, {len(pinned)} pinned]", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    for q in sys.stdin.read().splitlines():
        q = q.strip()
        if not q:
            continue
        hits = index.top(q)
        if not hits and "soup" not in q.lower():
            # Nothing in the docs matches and it is not about Soup: don't let the model riff.
            print(f"\nQ: {q}\nA: The documentation does not cover that.", flush=True)
            continue
        if HOWTO.search(q):
            # How-to / command questions: exact excerpts only, short answer, no overview noise.
            ctx = "\n".join(f"- {chunks[i]}" for i in hits)
            style = (
                "Answer concisely in 1-3 sentences. Copy any command exactly from the "
                "excerpts, inline in backticks, and do not mention commands that are not "
                "needed. If the answer is not there, say the documentation does not cover it."
            )
        else:
            # Conceptual questions: overview pinned, fuller prose answer.
            picked = pinned + [i for i in hits if i not in pinned]
            ctx = "\n".join(f"- {chunks[i]}" for i in picked)
            style = (
                "Answer in 3-5 sentences of plain prose: say what it is, why it is useful "
                "and how it works. Never write code blocks. If the answer is not there, "
                "say the documentation does not cover it and stop."
            )
        prompt = (
            "Answer the question using ONLY the documentation excerpts below. "
            f"{style} Do not add facts that are not in the excerpts.\n\n"
            f"Documentation:\n{ctx}\n\nQuestion: {q}"
        )
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=220, do_sample=False)
        ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\nQ: {q}\nA: {ans}", flush=True)


if __name__ == "__main__":
    main()
