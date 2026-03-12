# 💧 Project: Liquid NanoChat

**Goal:** To build, train, and deploy a "Liquid" version of a Large Language Model (LLM) that achieves high performance with a fraction of the memory footprint of a standard Transformer.

---

## 🎯 Project Objectives

* **Memory Efficiency:** Replace the $O(N^2)$ attention mechanism with $O(N)$ Liquid Recurrence.
* **Hardware Accessibility:** Enable full-cycle training (Pre-training to Chat) on a single laptop GPU.
* **Learning Focus:** Understand the transition from Fixed-Weight architectures to Continuous-Time (Liquid) architectures.

---

## 🚀 Stage 1: The Core Build (MVP)

In this stage, we modify the architecture of `nanochat` to support Liquid cells.

### 1. The Architectural "Swap"

* **The Component:** Replace `CausalSelfAttention` in `model.py` with a **CfC (Closed-form Continuous-time)** cell.
* **The Logic:** Implement a stateful `forward` pass that carries a hidden state $h$ across the sequence length.
* **The Hybrid Choice:** Retain a "Slim MLP" (1x expansion) to provide a stable knowledge base for the Liquid dynamics.

### 2. Initial "Smoke Test" Training

* **Dataset:** **TinyShakespeare** (approx. 1MB).
* **Hardware Settings:** * `device_batch_size: 1`
* `gradient_accumulation_steps: 16`
* `dtype: float16` (to save VRAM).


* **Metric of Success:** The model must "overfit" on a single paragraph and be able to recite it back within 100 iterations.

---

## 📈 Stage 2: The "Overnight" Pre-training

Once the core logic is verified, we scale to a meaningful (but small) dataset.

### 1. Data & Environment

* **Dataset:** **FineWeb-EDU (Sample)**. We will use a 100MB subset rather than the full terabyte.
* **Tracking:** Integrate **Weights & Biases (W&B)** to monitor:
* **Cross-Entropy Loss:** To track learning.
* **VRAM Usage:** To confirm the "Liquid" flat-line memory profile.
* **Tokens/Sec:** To balance speed vs. depth.



### 2. Hyperparameter Strategy

* **Sequence Length:** Start at 256, then test 1024.
* **Learning Rate:** Use a standard AdamW optimizer with a warm-up period, as Liquid nets can be sensitive to early large gradients.

---

## 💬 Stage 3: Post-Training (The "Chat" Phase)

Turning the base model into a helpful assistant using the existing `nanochat` infrastructure.

### 1. Supervised Fine-Tuning (SFT)

* Run `chat_sft.py` using a small instruction dataset like **SmolTalk**.
* **Goal:** Transition the model from "Predict the next word" to "Answer the user's question."

### 2. Deployment

* Use `chat_gui.py` to launch a local web interface.
* **Test:** Compare the response speed of the Liquid model vs. a standard Transformer of the same parameter count on your laptop.

---

## 🛠️ Stage 4: Extension Elements (Future Growth)

Once the basic LNN is functional, these modules can be added to increase "intelligence."

* **Learnable Time-Constants ($\tau$):** Allow the model to decide how fast its "liquid" neurons react to different types of information.
* **4-bit Quantization:** Integrate `bitsandbytes` to shrink the model weights, allowing for a "deeper" model (more layers) within the same VRAM.
* **Long-Context Expansion:** Test the model on sequences of 10,000+ tokens. Unlike a Transformer, the Liquid model should not slow down or crash as the context grows.
* **DPO Alignment:** Use `chat_rl.py` to provide "Preferred" vs "Rejected" answers to refine the model's personality.

---

## 📊 Verification & Metrics Table

| Metric | Target (First Pass) | Tool |
| --- | --- | --- |
| **VRAM Usage** | < 4GB | `nvidia-smi` / W&B |
| **Perplexity** | < 30 (on TinyShakespeare) | Training Logs |
| **State Stability** | No "NaN" gradients | W&B Gradients Plot |
| **Inference Latency** | < 50ms per token | `chat_gui.py` |

---

**Would you like me to generate the Python boilerplate for the CfC Liquid Cell to get you started on Stage 1?**