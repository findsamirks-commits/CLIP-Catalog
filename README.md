# CLIP-Catalog — Vision-Language Zero-Shot for Retail

> **New SKU onboarding at Milkbasket takes days** — annotate photos → retrain the classifier → redeploy. **CLIP-Catalog makes it seconds** — upload the image, get top-5 category predictions with confidence scores, and either auto-tag (high confidence) or route to a 3-second human review queue.
>
> OpenCLIP (ViT-B/32) fine-tuned on retail catalog data with a FAISS retrieval index, exposing three zero-shot inference modes: **text → SKU**, **image → SKU**, and **image → category**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-FF6F00?logo=huggingface&logoColor=white)](https://github.com/mlfoundations/open_clip)
[![FAISS](https://img.shields.io/badge/FAISS-sub--ms_retrieval-336791?logo=meta&logoColor=white)](https://faiss.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Why vision-language, not another supervised classifier?

Your existing [`Multimodal-SKU-Classifier`](https://github.com/findsamirks-commits/Multimodal-SKU-Classifier) is **supervised** — it classifies into a fixed set of categories you labeled during training. Every new category requires: collect images → annotate → retrain → redeploy.

CLIP-Catalog is **contrastive / self-supervised at pretraining, zero-shot at inference**. The model has seen 400M image-text pairs on the internet; it already knows what a "banana" looks like. Fine-tuning on *your* catalog teaches it your brand vocabulary and product styling — then at inference time, you can:

- Classify into **any text prompt** without retraining
- Retrieve SKUs by **natural-language query** (`"organic red apples 500g"`)
- Find **visually similar** SKUs for competitor matching / dedup / substitutes
- Handle **new SKUs instantly** — no new labels required

| | `Multimodal-SKU-Classifier` | **`CLIP-Catalog`** |
|---|---|---|
| Training signal | Supervised (fixed labels) | Contrastive InfoNCE (image↔text pairs) |
| New category? | Collect data + retrain | Just add the text prompt |
| Inference modes | Classification only | Text→SKU, Image→SKU, Image→category |
| Open-vocabulary detection | No | Extends to GroundingDINO / OWL-ViT |
| Foundation model lineage | None | Pretrained on 400M+ pairs (LAION, DataComp) |

---

## 🏗️ Four inference modes

### 1. Zero-shot classification
```python
from PIL import Image
from src.inference import CLIPRetriever

ret = CLIPRetriever("checkpoints/clip_catalog_best.pt")
labels = ["fresh produce", "dairy", "snacks", "beverages", "frozen"]

image = Image.open("new_sku_arrival.jpg")
scores = ret.zero_shot_classify(image, labels, prompt_template="a photo of {}")
# → {"fresh produce": 0.87, "dairy": 0.08, "snacks": 0.03, ...}
```

### 2. Text → SKU retrieval
```python
hits = ret.text_to_sku("organic red apples 500g pack", top_k=5)
# → [{"sku_id": "APL-ORG-500", "score": 0.91, "image_path": "..."}, ...]
```

### 3. Image → SKU retrieval (visual similarity / substitutes)
```python
query = Image.open("competitor_product_photo.jpg")
hits = ret.image_to_sku(query, top_k=5)
# → closest matching SKUs in your catalog
```

### 4. Image → category (open-vocabulary)
```python
# Use arbitrary category prompts at inference time — no retraining
scores = ret.zero_shot_classify(
    image,
    ["organic produce", "packaged snack", "dairy product", "millets and ancient grains"],
    prompt_template=["a photo of {}", "a product image of {}"],  # prompt ensemble
)
```

---

## 📁 Repository layout

```
CLIP-Catalog/
├── configs/clip_catalog.yaml       # Backbone + training + retrieval knobs
├── data/README.md                  # Fashion-Product-Images / ABO / custom catalog prep
├── notebooks/
│   └── 01_data_exploration.ipynb   # Zero-shot baseline + similarity heatmaps
├── src/
│   ├── data.py                     # CatalogDataset + image↔caption pair generation
│   ├── model.py                    # OpenCLIP wrapper with optional LoRA adapters
│   ├── train.py                    # Contrastive fine-tuning with cosine LR + AMP
│   ├── index.py                    # FAISS index builder (IndexFlatIP on L2-normalized embeds)
│   ├── eval.py                     # Recall@K, MRR, zero-shot top-1 accuracy
│   ├── inference.py                # CLIPRetriever: text→SKU, image→SKU, zero-shot classify
│   ├── onboarding_bridge.py        # Business layer: auto-tag vs human-review queue
│   └── utils.py
├── app/streamlit_app.py            # 3 tabs: text search · visual search · zero-shot classify
├── tests/test_eval.py              # Metric correctness + bridge threshold logic
└── docs/architecture.md            # CLIP math + fine-tuning recipe + mesh integration
```

---

## 🚀 Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                      # macOS/Linux
pip install -r requirements.txt

# 1. Download Fashion-Product-Images (smallest starter — 44K items)
#    See data/README.md for direct Kaggle link
python -m src.data --source fashion --input data/styles.csv \
    --image-dir data/images --output data/processed/catalog.jsonl

# 2. Fine-tune OpenCLIP ViT-B/32 (contrastive, 5-10 epochs)
python -m src.train --config configs/clip_catalog.yaml \
    --catalog data/processed/catalog.jsonl \
    --out checkpoints/clip_catalog_best.pt

# 3. Build FAISS index over catalog
python -m src.index --model-path checkpoints/clip_catalog_best.pt \
    --catalog data/processed/catalog.jsonl \
    --index-out data/processed/catalog.faiss

# 4. Evaluate retrieval metrics
python -m src.eval --model-path checkpoints/clip_catalog_best.pt \
    --catalog data/processed/catalog.jsonl \
    --queries data/processed/eval_queries.jsonl

# 5. Interactive demo
streamlit run app/streamlit_app.py
```

---

## 📊 Target metrics

> ⚠️ **Illustrative targets, not measured results yet.** Replace this table with the output of
> `python -m src.eval` (pretrained: omit `--model-path`; fine-tuned: pass the checkpoint).

Target for fine-tuning on Fashion-Product-Images (44K items, ~45 sub-categories):

| Metric | Pretrained OpenCLIP ViT-B/32 | **Fine-tuned (ours)** |
|---|---|---|
| Zero-shot top-1 accuracy | 42% | **71%** |
| Text→SKU Recall@5 | 51% | **84%** |
| Text→SKU Recall@10 | 63% | **91%** |
| Image→SKU Recall@5 | 68% | **89%** |
| Mean Reciprocal Rank | 0.41 | **0.67** |

The jump from pretrained to fine-tuned is the single clearest demonstration of value — pretrained CLIP already knows what clothes look like; fine-tuning teaches it your brand's product styling + taxonomy.

---

## 🧠 Onboarding bridge (business layer)

`src/onboarding_bridge.py` decouples the raw predictions from the actual catalog-management decision:

```python
from src.onboarding_bridge import OnboardingBridge

bridge = OnboardingBridge(
    auto_tag_confidence=0.70,       # ≥ 70% → auto-publish
    review_queue_confidence=0.40,   # [40%, 70%) → human review
    min_top1_margin=0.15,           # top-1 must beat top-2 by at least 15 pp
    forbidden_categories=["alcohol"],  # hard-block even if confident
)

decision = bridge.decide(zero_shot_scores, sku_metadata)
# → {"action": "auto_tag", "category": "fresh produce", "confidence": 0.87,
#    "audit": ["top1_score=0.87 ≥ 0.70", "margin=0.79 ≥ 0.15"]}
```

Possible actions:
- `auto_tag` → publish immediately with the predicted category
- `human_review` → route to a 3-second review queue (just confirm top-1)
- `reject` → reject (forbidden category, or no prediction > threshold)
- `escalate` → unusual case, route to category manager

This separation means a category manager can change confidence thresholds without touching the model.

---

## 🔗 Fits into the Autonomous Retail AI Mesh

```
┌─────────────────────────────────────────────────────────┐
│  Vendor upload: image + partial metadata (new SKU)      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  CLIP-Catalog  (OpenCLIP ViT-B/32 fine-tuned)           │
│  → zero_shot_classify + image_to_sku                    │
└──────┬──────────────────────────┬──────────────────────┘
       │                          │ visually similar SKUs
       ▼                          ▼
┌──────────────────┐     ┌──────────────────────┐
│ OnboardingBridge │     │ Multimodal-SKU-      │
│ (auto vs review) │     │ Classifier           │
│                  │     │ (supervised fallback)│
└───────┬──────────┘     └──────────────────────┘
        │ category + attributes
        ▼
┌────────────────────────────────────────────────┐
│ Downstream: BasketGNN, Fresh-Grade, Shelf-     │
│ Vision, Perishable-Markdown-Optimizer          │
│ — all now know about the new SKU immediately   │
└────────────────────────────────────────────────┘
```

**Mesh impact**: Every downstream model used to wait days for a new SKU to be properly categorized. Now it happens in the vendor upload flow.

---

## 🛠️ Config

`configs/clip_catalog.yaml`:

```yaml
model:
  backbone: ViT-B-32                # Options: ViT-B-32, ViT-B-16, ViT-L-14
  pretrained: laion2b_s34b_b79k     # OpenCLIP pretrained tag
  use_lora: false                   # LoRA adapters (via peft) for parameter-efficient
  lora_rank: 8
  freeze_text_encoder: false
  freeze_image_encoder: false

train:
  lr: 5.0e-5
  weight_decay: 0.01
  batch_size: 128
  epochs: 10
  warmup_epochs: 1
  grad_accum_steps: 1
  mixed_precision: true
  temperature: 0.07                 # InfoNCE initial temperature
  seed: 42
  device: auto

retrieval:
  top_k: 10
  normalize: true                   # L2-normalize embeddings before FAISS IP
  index_type: IndexFlatIP           # IndexFlatIP / IndexIVFFlat / IndexHNSWFlat

onboarding:
  auto_tag_confidence: 0.70
  review_queue_confidence: 0.40
  min_top1_margin: 0.15
  forbidden_categories: []
```

---

## 📚 Datasets

Full instructions in `data/README.md`. Supported out of the box:

| Dataset | Items | Modality | Notes |
|---|---|---|---|
| [Fashion-Product-Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) | 44K | image + 9 attribute fields | Best starter — compact & well-labeled |
| [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | 400K | image + text description | Gold standard; English descriptions |
| [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) | 491K | image + category + bbox | Good for OOD transfer evaluation |
| **Custom quick-commerce catalog** | — | image + metadata CSV | Bring your own |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Pure-Python tests covering Recall@K correctness, MRR mathematics, zero-shot prompt-ensembling logic, and OnboardingBridge decision rules — no model loading, no GPU needed.

---

## 📖 References

- Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*
- Ilharco et al. (2021) — *OpenCLIP: Reproducible scaling laws for contrastive image-text learning*
- Chen, Kornblith, Norouzi, Hinton (2020) — *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*
- Jia et al. (2021) — *ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Johnson, Douze, Jégou (2017) — *Billion-scale similarity search with GPUs (FAISS)*

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

**Built by [Samir Sandhwar](https://github.com/findsamirks-commits)** — Category Head at Milkbasket, IIT Madras Agentic AI alumnus. The vision-language zero-shot complement to [`Multimodal-SKU-Classifier`](https://github.com/findsamirks-commits/Multimodal-SKU-Classifier); part of the [Autonomous Retail AI Mesh](https://github.com/findsamirks-commits).

