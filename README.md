# AI Projects Portfolio | Artificial Intelligence Utils 📚

![AI Wizard](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaG5wYno0Y3M5amNibnU4ZTFycjY0Nzd2MW1jd3JwODRsc2c0ZTBraCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/0lGd2OXXHe4tFhb7Wh/giphy.gif)  
*"Turning coffee into neural networks since 2025" ☕➡️🤖*

Welcome to my AI project portfolio! This repository contains implementations and analyses of fundamental NLP techniques and modern transformer architectures. Below you'll find Batman-style tech briefings for each project!

---

## 🗂️ Project Catalog

### 1. 🔍 **Word Embeddings Showdown**  
**CBOW vs Skip-Gram vs GloVe**  
*NLP Fundamentals with Reuters Financial News*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)

#### 🎯 Key Insights
- Implemented 3 classic embedding models from scratch
- Developed multi-modal evaluation framework:
  - KNN Semantic Clustering 👯
  - SimLex-999 Benchmark 📈
  - Vector Arithmetic for Analogies ➕➖
  
#### 📉 Performance Matrix
| Model       | KNN Clustering | SimLex-999 ρ | Analogy Accuracy |
|-------------|----------------|-------------|------------------|
| **CBOW**    | 🌕🌕🌗🌑🌑      | 0.0954      | 0%              |
| **SkipGram**| 🌕🌕🌑🌑🌑      | 0.0504      | 0%              |
| **GloVe**   | 🌕🌑🌑🌑🌑      | 0.0659      | 0%              |

💡 **Epiphany Moment**: Even financial jargon needs bigger embeddings! (64-dim wasn't cutting it)

---

### 2. 🌀 **Transformer Times**  
**Chinese News Classification**  
*Battling with 10 News Categories*

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=WeightsAndBiases&logoColor=white)](https://wandb.ai)

#### 🚀 Turbocharged Architecture
- Scaled encoder layers: 2 → 8 🏗️
- Enhanced classification head with GAP 🎯
- Cosine decay + warmup scheduling 🔥

#### 📊 Results Evolution
| Version      | Accuracy | F1-Score | Key Improvement          |
|--------------|----------|----------|--------------------------|
| Baseline     | 81.19%   | 82.55%   | Initial Transformer      |
| +Preprocess  | 83.53%   | 83.34%   | Punctuation Ninjutsu ✂️  |
| Final Model  | 84.07%   | 84.19%   | Deep Encoder Magic 🧙    |

**Hot Take** 🔥: Commas matter! But sometimes they don't... 🤷

---

### 3. 🤖 **BERT Unleashed**  
**Dual Task Dominance**  
*Sentiment Analysis + Paraphrase Detection*

[![BERT](https://img.shields.io/badge/BERT-🤖-yellowgreen)](https://arxiv.org/abs/1810.04805)
![GPU Power](https://img.shields.io/badge/GPU-2xRTX2080Ti-76B900?logo=nvidia)

#### Mission Parameters
```python
{'tasks': ['SST2', 'MRPC'],
 'model': 'bert-mini',
 'secret_sauce': 'Custom LossCallback() 🕵️',
 'hardware': 'Enough CUDA cores to fry an egg 🍳'}
```
#### 📈 Performance Metrics
| Task | Accuracy | F1-Score | Prediction Prowess |
|------|----------|----------|----------------------------|
| SST2 | 82.80% | 83.11% | 4/5 Test Samples Correct 🎬|
| MRPC | 75.25% | 82.43% | 5/5 Real-world Correct 🌍 |

**Golden Insight** 💡: Small BERTs can play big! (But they still hate irony)

---
### 🧪 Lab Environment Specs
please refer to [reports]() accrodingly to each project:

**📜 Project Reports**
| Project                     | Report Link                       |
|-----------------------------|-----------------------------------|
| Word Embeddings Analysis    | [project1-2_report.pdf](./project1-2_report.pdf)|
| News Classification         | [project3_report.pdf](./project3_report.pdf)    |
| BERT Classification         | [project4_report.pdf](./project4_report.pdf)    |

---
## 📣 Future Quest Log
- Subword embeddings for rare financial terms 💼

- Hybrid positional encoding strategies 🧬

- Attention visualization toolkit 👀

- Domain-adaptive pretraining 🌐

---
Made with ❤️ (and probably too much caffeine) by **Zijin Cai**

*"If debugging is removing bugs, then programming must be putting them in." - Edsger Dijkstra*
