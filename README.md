# SmolLM-135M → Financial Domain Expert

> *Small models don't fail at language. They fail at specialization.*

A step-by-step pipeline transforming a generic 135M parameter model into a domain-aware financial assistant — using CPT, SFT, and (soon) RLHF.

---

## The Problem

`SmolLM-135M` out of the box:
- ✅ Fluent language
- ❌ No financial depth
- ❌ No instruction following
- ❌ No sense of good vs bad response

Three stages fix this. One at a time.

---

## Pipeline

```
SmolLM-135M ──► CPT ──► SFT ──► RLHF (WIP)
  (generic)    (knows)  (speaks)  (judges)
```

---

## Stage 1 — CPT (Continued Pre-Training)

**Goal:** Fill the brain with financial knowledge.

| What | Detail |
|------|--------|
| Base model | `HuggingFaceTB/SmolLM-135M` |
| Dataset | SEC 10K filings |
| Method | LoRA + QLoRA (4-bit) via Unsloth |
| LoRA rank | 32 |
| Tracked | Perplexity |

**Issue hit:** Catastrophic forgetting — model lost general knowledge even after lowering `embed_tokens` LR by 10x.

**Fix:** Mixed general + financial data → preserved base knowledge while absorbing the domain.

**Result:** Domain-aware model with noticeably improved financial language.

**Limitation:** Understands finance. Still can't follow instructions.

📓 [CPT Notebook](https://github.com/wasjunaid/fine_tuning/blob/main/continued_pre_trainning.ipynb)

---

## Stage 2 — SFT (Supervised Fine-Tuning)

**Goal:** Teach it to think before it speaks.

| What | Detail |
|------|--------|
| Dataset | `virattt/financial-qa-10k` |
| Template | Alpaca (Instruction → Input → Response) |
| Method | QLoRA rank 16 via Unsloth |
| Target modules | Attention + MLP only — no `embed_tokens`/`lm_head` |
| Extras | Sequence packing, loss masking, EOS token per sample |

**Key move:** Merged CPT adapter → then layered SFT on top. Clean separation of concerns.

**Result:** Structured, on-point financial Q&A. Model stops when it should.

**Also learned:** Perplexity doesn't evaluate SFT.
The real checks:
- Does it answer the actual question?
- Does it stop on time?
- Does it hallucinate?

📓 [SFT Notebook](https://github.com/wasjunaid/fine_tuning/blob/main/supervised_fine_tuning.ipynb)

---

## Stage 3 — RLHF (Coming Soon)

The model knows *how* to respond.  
It still can't tell a good answer from a bad one.

That's a reward problem. Human feedback fixes it.

> **Status:** 🔧 In progress — follow for updates.

---

## Stack

`Unsloth` · `LoRA` · `QLoRA` · `HuggingFace` · `PyTorch` · `Alpaca` · `PEFT`

---

## Questions / Suggestions?

Feel free to open an issue or reach out directly.  
Always open to feedback on the pipeline.
