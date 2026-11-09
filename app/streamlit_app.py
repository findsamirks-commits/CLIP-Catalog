"""Streamlit interactive application for CLIP-Catalog.

Exposes three production workflows:
1. Text -> SKU Semantic Retrieval
2. Image -> SKU Visual Similarity Search
3. Zero-Shot SKU Onboarding & Commercial Decision Bridge
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# Set ARM64 CPU check bypass for environments running on Snapdragon
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# Ensure root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config
from src.inference import CLIPRetriever
from src.onboarding_bridge import OnboardingBridge
from src.data import generate_synthetic_catalog


st.set_page_config(
    page_title="CLIP-Catalog | Vision-Language Zero-Shot for Retail",
    page_icon="🛒",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-auto { background-color: #d1fae5; color: #065f46; border: 1px solid #34d399; }
    .badge-review { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
    .badge-reject { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-escalate { background-color: #dbeafe; color: #1e40af; border: 1px solid #60a5fa; }
    .sku-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    """Initializes and caches CLIPRetriever instance."""
    return CLIPRetriever()


def create_placeholder_image(label: str, color: str = "#2563eb") -> Image.Image:
    """Generates a simple PIL thumbnail image for demo display."""
    img = Image.new("RGB", (224, 224), color="#f3f4f6")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 214, 214], outline=color, width=3)
    draw.text((20, 100), label[:22], fill="#1f2937")
    return img


st.title("🛒 CLIP-Catalog: Vision-Language Zero-Shot for Retail")
st.caption(
    "OpenCLIP (ViT-B/32) fine-tuned on retail catalog data with FAISS retrieval index "
    "and autonomous commercial onboarding decision bridge. Part of the Autonomous Retail AI Mesh."
)

retriever = get_retriever()

tab1, tab2, tab3 = st.tabs([
    "🔍 1. Text → SKU Semantic Search",
    "🖼️ 2. Image → SKU Visual Search",
    "⚡ 3. Zero-Shot SKU Onboarding Bridge"
])

# -----------------------------------------------------------------------------
# TAB 1: Text -> SKU Search
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Natural-Language Semantic SKU Retrieval")
    st.write("Type a descriptive customer or category query to retrieve matching catalog SKUs via FAISS inner-product search.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query_text = st.text_input(
            "Search Query",
            value="Aashirvaad whole wheat atta 5kg fresh flour pack",
            help="E.g. 'organic red apples 500g', 'amul chilled milk 1L', 'chilli potato chips'"
        )
    with col_q2:
        top_k = st.slider("Top K Results", min_value=3, max_value=12, value=6)

    if st.button("Search Catalog", key="btn_text_search") or query_text:
        hits = retriever.text_to_sku(query_text, top_k=top_k)
        st.write(f"Found **{len(hits)}** matching SKUs:")

        cols = st.columns(min(len(hits), 3))
        for i, hit in enumerate(hits):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="sku-card">
                    <span class="metric-badge badge-auto">Rank #{hit.get('rank', i+1)} | Sim: {hit.get('score', 0.0):.3f}</span>
                    <h4 style="margin-top:8px; margin-bottom:4px;">{hit.get('title', 'Product')}</h4>
                    <p style="color:#4b5563; font-size:0.85rem; margin:0;"><b>Category:</b> {hit.get('category', 'N/A')}</p>
                    <p style="color:#4b5563; font-size:0.85rem; margin:0;"><b>Subcategory:</b> {hit.get('subcategory', 'N/A')}</p>
                    <p style="color:#10b981; font-weight:600; font-size:0.95rem; margin-top:4px;">₹{hit.get('price_inr', 120.0):.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.image(create_placeholder_image(hit.get('title', 'SKU')), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Image -> SKU Visual Search
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Image-to-Image Visual Similarity & Competitor Match")
    st.write("Upload a competitor photo or sample image to find visually identical or substitute SKUs in your dark-store catalog.")

    preset_samples = {
        "Shimla Apple Box": ("Fresh Produce", "#ef4444"),
        "Amul Butter Pack": ("Dairy & Breakfast", "#f59e0b"),
        "Lay's Magic Masala Pack": ("Snacks & Munchies", "#3b82f6"),
        "Tata Tea Gold Carton": ("Beverages", "#10b981")
    }

    col_img1, col_img2 = st.columns([1, 2])
    with col_img1:
        sample_choice = st.selectbox("Select Sample Photo", list(preset_samples.keys()))
        uploaded_file = st.file_uploader("Or Upload Product Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            query_img = Image.open(uploaded_file).convert("RGB")
        else:
            cat, color = preset_samples[sample_choice]
            query_img = create_placeholder_image(sample_choice, color)
        
        st.image(query_img, caption="Query Photo", use_container_width=True)

    with col_img2:
        top_k_img = st.slider("Top K Visual Matches", min_value=3, max_value=9, value=4)
        if st.button("Run Visual Match", key="btn_img_search"):
            img_hits = retriever.image_to_sku(query_img, top_k=top_k_img)
            st.write(f"Top {len(img_hits)} visual matches from catalog index:")
            
            for hit in img_hits:
                st.markdown(f"""
                <div class="sku-card">
                    <b>{hit.get('title')}</b> — <i>{hit.get('category')} / {hit.get('subcategory')}</i>
                    <br><span class="metric-badge badge-auto">Cosine Match: {hit.get('score', 0.0):.3f}</span>
                    <span style="margin-left: 10px; color:#4b5563;">SKU: {hit.get('sku_id')}</span>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: Zero-Shot Onboarding Bridge
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("⚡ Autonomous New-SKU Onboarding & Decision Bridge")
    st.write(
        "Upload a never-seen product image. Zero-shot contrastive head predicts probabilities across "
        "your taxonomy. The commercial **OnboardingBridge** determines whether to auto-publish, route to human review, or block."
    )

    col_b1, col_b2 = st.columns([1, 1])

    with col_b1:
        st.markdown("#### 1. Candidate Categories (Taxonomy)")
        categories_input = st.text_area(
            "Comma-separated category prompts",
            value="Fresh Produce, Dairy & Breakfast, Staples & Grains, Snacks & Munchies, Beverages, Personal Care, Alcohol & Tobacco"
        )
        candidate_categories = [c.strip() for c in categories_input.split(",") if c.strip()]

        st.markdown("#### 2. Commercial Bridge Thresholds")
        auto_thresh = st.slider("Auto-Tag Threshold (≥)", 0.50, 0.95, 0.70, 0.05)
        review_thresh = st.slider("Human Review Threshold (≥)", 0.20, 0.60, 0.40, 0.05)
        min_margin = st.slider("Min Top-1 vs Top-2 Margin", 0.05, 0.35, 0.15, 0.05)
        forbidden_cats = st.multiselect(
            "Forbidden / Compliance Block Categories",
            candidate_categories,
            default=["Alcohol & Tobacco"] if "Alcohol & Tobacco" in candidate_categories else []
        )

        bridge = OnboardingBridge(
            auto_tag_confidence=auto_thresh,
            review_queue_confidence=review_thresh,
            min_top1_margin=min_margin,
            forbidden_categories=forbidden_cats
        )

    with col_b2:
        st.markdown("#### 3. New SKU Arrival")
        sku_name = st.text_input("Product Title", "Mother Dairy Buffalo Milk 500ml Chilled Pouch")
        test_img = create_placeholder_image(sku_name, "#0284c7")
        st.image(test_img, caption="New SKU Photo", width=160)

        # Compute zero-shot probabilities
        scores = retriever.zero_shot_classify(
            image=test_img,
            candidate_labels=candidate_categories,
            prompt_template=["a photo of {}", "a packaged retail product of {}"]
        )

        st.markdown("#### 4. Model Predictions")
        chart_df = pd.DataFrame(list(scores.items()), columns=["Category", "Probability"]).set_index("Category")
        st.bar_chart(chart_df)

        # Run OnboardingBridge Decision
        decision = bridge.decide(scores, sku_metadata={"title": sku_name})

        action = decision["action"]
        badge_class = {
            "auto_tag": "badge-auto",
            "human_review": "badge-review",
            "reject": "badge-reject",
            "escalate": "badge-escalate"
        }.get(action, "badge-review")

        action_labels = {
            "auto_tag": "✅ AUTO-TAG (Instant Catalog Publish)",
            "human_review": "⏱️ HUMAN REVIEW (3-Second Micro-Queue)",
            "reject": "🚫 REJECT (Compliance / Floor Violation)",
            "escalate": "⚠️ ESCALATE (Category Manager Ambiguity Review)"
        }

        st.markdown("#### 5. Autonomous Commercial Decision")
        st.markdown(f"""
        <div style="padding:16px; border-radius:8px; border:1px solid #d1d5db; background:#ffffff;">
            <div style="font-size:1.15rem; font-weight:700; margin-bottom:8px;">
                Action: <span class="metric-badge {badge_class}">{action_labels.get(action, action)}</span>
            </div>
            <p><b>Target Category:</b> {decision['category']}</p>
            <p><b>Top-1 Confidence:</b> {decision['confidence'] * 100:.1f}% | <b>Margin:</b> {decision['margin'] * 100:.1f}%</p>
            <b>Audit Log:</b>
            <ul>
                {''.join(f'<li>{item}</li>' for item in decision['audit'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "**Built by [Samir Sandhwar](https://github.com/findsamirks-commits)** · Category Head at Milkbasket · "
    "Autonomous Retail AI Mesh Architecture"
)
