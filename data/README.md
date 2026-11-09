# CLIP-Catalog Dataset Preparation Guide

This guide details the schemas and preprocessing steps required to format catalog data for contrastive fine-tuning, FAISS retrieval index construction, and zero-shot categorization evaluation.

---

## 🗂️ Target Schema: `catalog.jsonl`

All raw datasets are transformed by `src/data.py` into a standardized JSON Lines format (`catalog.jsonl`). Each line is a valid JSON object:

```json
{
  "sku_id": "SKU_FMCG_10482",
  "title": "Aashirvaad Superior MP Sharbati Whole Wheat Atta 5kg",
  "brand": "Aashirvaad",
  "category": "Staples & Grains",
  "subcategory": "Atta & Flours",
  "attributes": {
    "pack_size": "5 kg",
    "dietary_type": "Vegetarian",
    "shelf_life_days": 180
  },
  "caption": "Aashirvaad Superior MP Sharbati Whole Wheat Atta 5kg, Staples & Grains packaging photo",
  "image_path": "data/images/SKU_FMCG_10482.jpg",
  "split": "train"
}
```

### Field Definitions
- `sku_id` (str): Unique retail SKU identifier.
- `title` (str): Canonical product name shown to customer.
- `brand` (str): Brand name.
- `category` (str): Primary taxonomy classification (used for zero-shot benchmarking).
- `subcategory` (str): Granular hierarchy level.
- `attributes` (dict): Extracted metadata attributes (pack size, variant, ingredients).
- `caption` (str): Contrastive text target aligned with the SKU image during training.
- `image_path` (str): Relative or absolute local file path to the product image (RGB, JPEG/PNG).
- `split` (str): One of `"train"`, `"val"`, or `"test"`.

---

## 📦 Supported Datasets Out-of-the-Box

### 1. Fashion-Product-Images (Recommended Starter)
- **Scale**: 44,441 catalog items.
- **Modality**: High-resolution studio product photos + 9 attribute fields (`gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `year`, `usage`, `productDisplayName`).
- **Download**: [Kaggle Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- **Ingestion**:
  ```bash
  python -m src.data --source fashion \
      --input data/styles.csv \
      --image-dir data/images \
      --output data/processed/catalog.jsonl
  ```

### 2. Amazon Berkeley Objects (ABO)
- **Scale**: ~400,000 real-world products.
- **Modality**: Clean multi-view product imagery, 3D models, rich multi-attribute English descriptions.
- **Download**: [Amazon Berkeley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- **Ingestion**:
  ```bash
  python -m src.data --source abo \
      --input data/abo_listings.json.gz \
      --image-dir data/images \
      --output data/processed/catalog.jsonl
  ```

### 3. DeepFashion2
- **Scale**: 491,000 images with 13 commercial fashion categories and dense bounding boxes.
- **Modality**: In-the-wild and shop images. Useful for OOD (out-of-distribution) retrieval evaluation.
- **Download**: [DeepFashion2 GitHub Repo](https://github.com/switchablenorms/DeepFashion2)

### 4. Custom Quick-Commerce Catalog (Milkbasket Pattern)
Bring your own CSV or Parquet catalog dump containing at minimum `sku_id`, `product_name`, `category`, and `image_path`:
```bash
python -m src.data --source custom \
    --input data/raw_catalog.csv \
    --output data/processed/catalog.jsonl
```

---

## 🧪 Synthetic Dataset Generation (Testing & Demos)

For testing and running the Streamlit app without downloading large image dumps, `src/data.py` provides an instant synthetic catalog generator:

```python
from src.data import generate_synthetic_catalog

catalog = generate_synthetic_catalog(num_skus=120, output_path="data/processed/catalog.jsonl")
print(f"Generated {len(catalog)} synthetic catalog entries across 6 quick-commerce categories.")
```

