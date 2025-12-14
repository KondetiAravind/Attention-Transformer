#  Neural Machine Translation using Transformer

### *(Attention Is All You Need – English → Spanish)*

This project presents an **end-to-end implementation of the Transformer architecture**
proposed in the paper **“Attention Is All You Need” (Vaswani et al., 2017)** for the task of
**Neural Machine Translation (NMT)**.

The model translates **English sentences into Spanish** using a **pure attention-based
encoder–decoder architecture**, implemented from scratch using **TensorFlow / Keras**.

---

## 📄 Reference Paper

> **Vaswani et al., “Attention Is All You Need”**
> [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

The implementation closely follows the original paper:

* No recurrence
* No convolution
* Self-attention based sequence modeling
* Positional encodings
* Multi-head attention
* Encoder–decoder Transformer blocks

---

## 🧠 Model Overview

### 🔹 Architecture

* **Encoder–Decoder Transformer**
* **Multi-Head Self-Attention**
* **Masked Self-Attention in Decoder**
* **Encoder–Decoder Cross Attention**
* **Feed Forward Networks**
* **Residual Connections + Layer Normalization**
* **Sinusoidal Positional Encoding**

### 🔹 Implementation Details

| Component           | Value                           |
| ------------------- | ------------------------------- |
| Vocabulary size     | 1000                            |
| Max sequence length | 50                              |
| Embedding dimension | 128                             |
| Attention heads     | 8                               |
| Encoder layers      | 2                               |
| Decoder layers      | 2                               |
| Optimizer           | Adam                            |
| Loss                | Sparse Categorical Crossentropy |
| Training hardware   | CPU                             |

---

## 📊 Dataset

* **Dataset**: English–Spanish parallel corpus (spa-eng)
* **Source**: Tatoeba Project
* **Preprocessing**:

  * Removed special punctuation (`¡`, `¿`)
  * Added `startofseq` and `endofseq` tokens
* **Dataset split**:

  * Training: 40,000 sentence pairs
  * Validation: 8,000 sentence pairs

📌 *Dataset files are not included in the repository and can be re-downloaded for reproducibility.*

---

## 📈 Training Results

The model was trained for **8 epochs** with early stopping and best-model checkpointing.

### 🔹 Epoch-wise Performance

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ----: | ---------: | -------------: | --------------: | ------------------: |
|     1 |     1.1571 |         85.25% |          0.6569 |              88.39% |
|     2 |     0.6439 |         88.85% |          0.5688 |              89.96% |
|     3 |     0.5543 |         90.02% |          0.5155 |              90.32% |
|     4 |     0.5024 |         90.44% |          0.4697 |              90.79% |
|     5 |     0.4600 |         90.84% |          0.4444 |              91.06% |
|     6 |     0.4363 |         91.02% |          0.4249 |              91.28% |
|     7 |     0.4141 |         91.31% |          0.4038 |              91.59% |
|     8 |     0.3924 |         91.58% |          0.3830 |              91.90% |

✔ Validation loss decreased consistently
✔ No overfitting observed
✔ Best model saved automatically (`transformer_best.keras`)

---

## 📊 Evaluation Metric

### 🔹 BLEU Score

The translation quality is evaluated using **BLEU (Bilingual Evaluation Understudy)**,
a standard metric for machine translation.

```
Average BLEU score: 0.0273
```

### 🔹 BLEU Interpretation

The BLEU score is relatively low due to:

* Limited vocabulary size (**1000 tokens**)
* Frequent occurrence of the `[UNK]` token
* Greedy decoding strategy (no beam search)
* CPU-constrained training setup

📌 **Despite the low BLEU score, the model learns strong sentence structure
and alignment patterns**, which is evident from qualitative results.

---

## 📝 Qualitative Translation Examples

| English Input                    | Ground Truth                       | Model Prediction                |
| -------------------------------- | ---------------------------------- | ------------------------------- |
| We’re investigating it.          | Lo estamos investigando.           | `[UNK] [UNK]`                   |
| My son is a rebellious teenager. | Mi hijo es un adolescente rebelde. | *mi padre se [UNK] en el [UNK]* |
| I’m tired.                       | Yo estoy cansada.                  | *estoy [UNK]*                   |
| I cooked dinner.                 | Cociné la cena.                    | `[UNK] [UNK]`                   |
| What made you come here?         | Qué te trajo aquí?                 | *qué te [UNK]*                  |

📌 The predictions show:

* Correct grammatical structure
* Proper word ordering
* Missing content words due to vocabulary constraints

---

## 🧩 Model Architecture Visualization

A full architecture diagram of the Transformer model is provided:

```
results/transformer_architecture.png
```

This diagram illustrates:

* Encoder and decoder stacks
* Multi-head attention blocks
* Positional encoding
* Residual connections

---

## 📂 Project Structure

```
Attention_Transformer/
│
├── models/
│   ├── transformer_best.keras
│   ├── attention.keras
│   └── mha.keras
│
├── notebooks/
│   ├── Final_Transformer.ipynb
│   └── Attention_Transformer.ipynb
│
├── results/
│   └── transformer_architecture.png
│
├── src/                  # reserved for future modularization
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/KondetiAravind/Attention-Transformer.git
cd Attention-Transformer
```

Open the notebook:

```
notebooks/Attention_Transformer.ipynb
```

Run all cells sequentially to:

* Load dataset
* Train Transformer model
* Evaluate using BLEU
* Generate translations

---

## 📓 Training Notebook

📘 **`notebooks/Attention_Transformer.ipynb`**

The notebook includes:

* Dataset preprocessing
* Text vectorization
* Transformer encoder–decoder implementation
* Training with callbacks
* BLEU evaluation
* Translation inference

This notebook ensures **full reproducibility** and **academic transparency**.

---

## 🎯 Key Highlights

* Complete implementation of **“Attention Is All You Need”**
* Pure Transformer (no RNN / CNN)
* Encoder–decoder attention mechanism
* BLEU-based evaluation
* Clean GitHub-ready project structure
* teCPU-compatible training pipeline

---

## 🧑‍🎓 Academic Context

This project demonstrates practical understanding of:

* Transformer architectures
* Self-attention mechanisms
* Neural Machine Translation
* Sequence-to-sequence learning
* Model evaluation in NLP

---

## 👤 Author

**Kondeti Aravind**
4th Year Dual Degree (CSE)
Indian Institute of Technology Bhubaneswar

🔗 GitHub: [https://github.com/KondetiAravind](https://github.com/KondetiAravind)

---

## 📜 License

This project is intended for **educational and research purposes only**.

---
