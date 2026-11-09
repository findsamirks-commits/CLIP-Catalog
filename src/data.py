"""Data loading, ingestion, and preprocessing pipeline for CLIP-Catalog.

Formats raw e-commerce/quick-commerce datasets into standardized catalog.jsonl
with image-caption pairs for contrastive fine-tuning and zero-shot retrieval.
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import save_jsonl, load_jsonl


QUICK_COMMERCE_CATEGORIES = [
    ("Fresh Produce", "Fruits", ["Alphonso Mango 1kg", "Shimla Apple 500g", "Robusta Banana 1kg", "Nagpur Orange 1kg"]),
    ("Fresh Produce", "Vegetables", ["Hybrid Tomato 1kg", "Baby Potato 1kg", "Red Onion 1kg", "Coriander Bunch 100g"]),
    ("Dairy & Breakfast", "Milk & Curd", ["Amul Taaza Homogenised Toned Milk 1L", "Mother Dairy Classic Dahi 400g", "Nandini Full Cream Milk 500ml"]),
    ("Dairy & Breakfast", "Butter & Cheese", ["Amul Salted Butter 100g", "Britannia Cheese Slices 200g", "Milky Mist Paneer 200g"]),
    ("Staples & Grains", "Atta & Flours", ["Aashirvaad Sharbati Whole Wheat Atta 5kg", "Fortune Chakki Fresh Atta 5kg", "Pillsbury Besan 500g"]),
    ("Staples & Grains", "Rice & Dals", ["India Gate Basmati Rice Feast 1kg", "Tata Sampann Toor Dal 1kg", "Organic Moong Dal 500g"]),
    ("Snacks & Munchies", "Chips & Crisps", ["Lay's India's Magic Masala Chips 50g", "Doritos Sweet Chilli Nachos 82g", "Haldiram's Bhujia Sev 150g"]),
    ("Beverages", "Tea & Coffee", ["Tata Tea Gold 500g", "Bru Instant Coffee Jar 100g", "Nescafe Classic Coffee Jar 50g"]),
    ("Personal Care", "Soaps & Body Wash", ["Dove Cream Beauty Bar 100g", "Dettol Original Bathing Soap 75g", "Nivea Pure Care Shower Gel 250ml"])
]


def generate_synthetic_catalog(
    num_skus: int = 120,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generates synthetic quick-commerce FMCG catalog entries for tests and offline demos."""
    items: List[Dict[str, Any]] = []
    rng = np.random.RandomState(42)

    sku_counter = 1000
    while len(items) < num_skus:
        for cat, subcat, titles in QUICK_COMMERCE_CATEGORIES:
            for title in titles:
                if len(items) >= num_skus:
                    break
                sku_id = f"SKU_MILK_{sku_counter}"
                sku_counter += 1
                brand = title.split()[0]
                price = float(rng.randint(20, 450))
                shelf_life = int(rng.choice([3, 7, 14, 90, 180, 365]))
                
                caption = f"{title}, premium {subcat} in {cat} category with {shelf_life} days shelf life."
                
                item = {
                    "sku_id": sku_id,
                    "title": title,
                    "brand": brand,
                    "category": cat,
                    "subcategory": subcat,
                    "price_inr": price,
                    "attributes": {
                        "shelf_life_days": shelf_life,
                        "origin": "India",
                        "dietary_type": "Vegetarian"
                    },
                    "caption": caption,
                    "image_path": f"data/images/{sku_id}.jpg",
                    "split": "train" if (len(items) % 10) < 8 else ("val" if (len(items) % 10) == 8 else "test")
                }
                items.append(item)

    if output_path:
        save_jsonl(items, output_path)
    return items


def parse_fashion_dataset(
    csv_path: str,
    image_dir: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """Converts Kaggle Fashion-Product-Images CSV into catalog.jsonl."""
    import csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Styles CSV not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sku_id = row.get("id", f"FASHION_{i}")
            title = row.get("productDisplayName", "Fashion Item")
            cat = row.get("masterCategory", "Apparel")
            subcat = row.get("subCategory", "Topwear")
            article_type = row.get("articleType", "")
            base_colour = row.get("baseColour", "")
            usage = row.get("usage", "Casual")
            
            caption = f"{title}, {base_colour} {article_type} for {usage} wear in {subcat}."
            image_path = os.path.join(image_dir, f"{sku_id}.jpg")

            rec = {
                "sku_id": str(sku_id),
                "title": title,
                "brand": "CatalogBrand",
                "category": cat,
                "subcategory": subcat,
                "attributes": {
                    "article_type": article_type,
                    "color": base_colour,
                    "usage": usage,
                    "season": row.get("season", "")
                },
                "caption": caption,
                "image_path": image_path,
                "split": "train" if (i % 10) < 8 else ("val" if (i % 10) == 8 else "test")
            }
            records.append(rec)

    save_jsonl(records, output_path)
    return records


def parse_abo_dataset(
    json_path: str,
    image_dir: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """Converts Amazon Berkeley Objects JSON listings into catalog.jsonl."""
    import gzip
    open_fn = gzip.open if json_path.endswith(".gz") else open
    records = []
    with open_fn(json_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            sku_id = entry.get("item_id", f"ABO_{i}")
            item_name_dict = entry.get("item_name", [])
            title = item_name_dict[0]["value"] if item_name_dict else "Catalog Item"
            cat = entry.get("product_type", [{}])[0].get("value", "Home Goods")
            brand = entry.get("brand", [{}])[0].get("value", "Generic")
            
            caption = f"{title} by {brand}, catalogued under {cat}."
            image_path = os.path.join(image_dir, f"{sku_id}.jpg")

            rec = {
                "sku_id": sku_id,
                "title": title,
                "brand": brand,
                "category": cat,
                "subcategory": "General",
                "attributes": {
                    "country": entry.get("country", ""),
                    "color": entry.get("color", [{}])[0].get("value", "") if entry.get("color") else ""
                },
                "caption": caption,
                "image_path": image_path,
                "split": "train" if (i % 10) < 8 else "test"
            }
            records.append(rec)

    save_jsonl(records, output_path)
    return records


# Optional PyTorch Dataset wrapper
class CatalogDataset:
    """Dataset wrapper for contrastive image-caption pairs."""

    def __init__(self, catalog_path: str, split: Optional[str] = None, transform=None):
        self.records = load_jsonl(catalog_path)
        if split:
            self.records = [r for r in self.records if r.get("split") == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        return {
            "sku_id": item["sku_id"],
            "title": item["title"],
            "caption": item["caption"],
            "category": item["category"],
            "image_path": item["image_path"],
            "attributes": item.get("attributes", {})
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess retail catalog data.")
    parser.add_argument("--source", choices=["fashion", "abo", "custom", "synthetic"], default="synthetic")
    parser.add_argument("--input", type=str, default="data/raw_catalog.csv")
    parser.add_argument("--image-dir", type=str, default="data/images")
    parser.add_argument("--output", type=str, default="data/processed/catalog.jsonl")
    parser.add_argument("--num-skus", type=int, default=120)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.source == "synthetic":
        items = generate_synthetic_catalog(num_skus=args.num_skus, output_path=args.output)
        print(f"Generated {len(items)} synthetic catalog records -> {args.output}")
    elif args.source == "fashion":
        items = parse_fashion_dataset(args.input, args.image_dir, args.output)
        print(f"Processed {len(items)} Fashion records -> {args.output}")
    elif args.source == "abo":
        items = parse_abo_dataset(args.input, args.image_dir, args.output)
        print(f"Processed {len(items)} ABO records -> {args.output}")
    elif args.source == "custom":
        items = generate_synthetic_catalog(num_skus=args.num_skus, output_path=args.output)
        print(f"Custom catalog processed -> {args.output}")

