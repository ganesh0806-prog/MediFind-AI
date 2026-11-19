import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediFind AI",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------
# SIMPLE NATIVE STREAMLIT STYLING
# No HTML is used in this application.
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f9f6;
    }
    [data-testid="stHeader"] {
        background: rgba(245, 249, 246, 0.9);
    }
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2ece5;
        padding: 12px;
        border-radius: 12px;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #087f5b;
        color: white;
        border: 0;
        border-radius: 9px;
        font-weight: 700;
    }
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #066b4d;
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# Expected file: models/medifind_model.pkl
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "medifind_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found. Add medifind_model.pkl inside the models folder."
        )

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


try:
    model = load_model()
    tfidf = model["tfidf_vectorizer"]
    tfidf_matrix = model["tfidf_matrix"]
    medicines = model["medicines"].copy()
except Exception as error:
    st.error(f"Unable to load the MediFind model: {error}")
    st.info(
        "Check that models/medifind_model.pkl is present in your GitHub repository."
    )
    st.stop()


# Check required columns.
required_columns = [
    "medicine_name",
    "generic_name",
    "brand_name",
    "strength",
    "dosage_form",
    "prescription_type",
    "category",
]
missing_columns = [column for column in required_columns if column not in medicines.columns]

if missing_columns:
    st.error(
        "The model's medicine data is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ---------------------------------------------------------
# SEARCH HELPERS
# ---------------------------------------------------------
def normalize_text(value):
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def search_medicines(query, top_n=100):
    query = query.strip()
    query_vector = tfidf.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()

    results = medicines.copy()
    results["similarity_score"] = scores

    normalized_query = normalize_text(query)
    results["generic_normalized"] = (
        results["generic_name"].fillna("").apply(normalize_text)
    )
    results["name_normalized"] = (
        results["medicine_name"].fillna("").apply(normalize_text)
    )
    results["brand_normalized"] = (
        results["brand_name"].fillna("").apply(normalize_text)
    )

    results["generic_match"] = results["generic_normalized"].str.contains(
        re.escape(normalized_query), na=False
    )
    results["name_match"] = (
        results["name_normalized"].str.contains(re.escape(normalized_query), na=False)
        | results["brand_normalized"].str.contains(re.escape(normalized_query), na=False)
    )
    results["exact_generic"] = results["generic_normalized"].eq(normalized_query)

    results["final_score"] = (
        results["similarity_score"]
        + results["exact_generic"].astype(int) * 1.0
        + results["generic_match"].astype(int) * 0.5
        + results["name_match"].astype(int) * 0.3
    )

    direct_matches = results[
        results["generic_match"] | results["name_match"]
    ].copy()

    if direct_matches.empty:
        direct_matches = results[results["similarity_score"] > 0].copy()

    direct_matches = direct_matches.sort_values(
        "final_score", ascending=False
    )
    direct_matches = direct_matches.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return direct_matches.head(top_n)


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
st.title("💊 MediFind AI")
st.caption("SMART MEDICINE DISCOVERY")

st.success("Discover Medicines. Search Smarter.")
st.write(
    "Explore medicines by name, brand or generic name. "
    "Find relevant catalogue information using AI-powered medicine search."
)

st.divider()


# ---------------------------------------------------------
# SEARCH FORM
# ---------------------------------------------------------
st.header("🔍 Find Your Medicine")

with st.form("medicine_search_form"):
    query = st.text_input(
        "Medicine name",
        placeholder="Try Paracetamol, Crocin, Vitamin C...",
    )
    submitted = st.form_submit_button(
        "🔍 Search Medicines",
        use_container_width=True,
    )


# ---------------------------------------------------------
# CHECKBOX FILTERS
# ---------------------------------------------------------
st.subheader("☑️ Select Your Preferences")
st.caption("Tick one or more options to filter medicines.")

all_categories = sorted(
    medicines["category"].dropna().astype(str).unique().tolist()
)
prescription_types = sorted(
    medicines["prescription_type"].dropna().astype(str).unique().tolist()
)

with st.container(border=True):
    st.markdown("**Medicine Categories**")
    selected_categories = []

    category_columns = st.columns(3)
    for index, category in enumerate(all_categories):
        with category_columns[index % 3]:
            if st.checkbox(category, key=f"category_{category}"):
                selected_categories.append(category)

    st.divider()
    st.markdown("**Prescription Type**")
    selected_prescription = []

    prescription_columns = st.columns(max(1, min(3, len(prescription_types))))
    for index, prescription in enumerate(prescription_types):
        with prescription_columns[index % len(prescription_columns)]:
            if st.checkbox(prescription, key=f"prescription_{prescription}"):
                selected_prescription.append(prescription)


# ---------------------------------------------------------
# ILLUSTRATIVE PHARMACY SHELVES
# ---------------------------------------------------------
st.divider()
st.header("🏪 Explore Our Pharmacy")
st.caption(
    "Illustrative medicine shelf display. These examples do not represent verified stock."
)

shelves = [
    {
        "title": "💊 Pain Relief & General Medicines",
        "products": [
            ("🟩", "PARACETAMOL", "Paracetamol"),
            ("🟦", "CROCIN", "Crocin"),
            ("🟧", "DOLO", "Dolo"),
            ("🟪", "IBUPROFEN", "Ibuprofen"),
            ("🟥", "CALPOL", "Calpol"),
        ],
    },
    {
        "title": "🌿 Vitamins & Supplements",
        "products": [
            ("🟧", "VITAMIN C", "Vitamin C"),
            ("🟥", "BECOSULES", "Becosules"),
            ("🟦", "VITAMIN D", "Vitamin D"),
            ("🟩", "MULTIVITAMIN", "Multivitamin"),
            ("🟪", "ZINC", "Zinc"),
        ],
    },
    {
        "title": "🩺 Healthcare & Wellness",
        "products": [
            ("🟦", "ALLERGY CARE", "Allergy Care"),
            ("🟩", "DIGESTIVE", "Digestive Care"),
            ("🟪", "HEALTH CARE", "Healthcare"),
            ("🟧", "SKIN CARE", "Skin Care"),
            ("🟥", "WELLNESS", "Wellness"),
        ],
    },
]

for shelf in shelves:
    st.subheader(shelf["title"])
    shelf_columns = st.columns(len(shelf["products"]))
    for index, (symbol, label, product_name) in enumerate(shelf["products"]):
        with shelf_columns[index]:
            with st.container(border=True):
                st.markdown(f"### {symbol}")
                st.markdown(f"**{label}**")
                st.caption(product_name)


# ---------------------------------------------------------
# CATALOGUE COUNTS
# ---------------------------------------------------------
st.divider()
st.header("📂 Medicine Catalogue")

category_counts = medicines["category"].value_counts()
count_columns = st.columns(5)

for index, (category, count) in enumerate(category_counts.items()):
    with count_columns[index % 5]:
        st.metric(str(category), f"{count} medicines")


# ---------------------------------------------------------
# MEDICINE RESULT CARDS
# ---------------------------------------------------------
def display_products(data):
    product_columns = st.columns(3)

    for index, (_, row) in enumerate(data.iterrows()):
        with product_columns[index % 3]:
            with st.container(border=True):
                st.markdown("### 💊 " + str(row.get("medicine_name", "N/A")))
                st.write("**Generic:**", row.get("generic_name", "N/A"))
                st.write("**Brand:**", row.get("brand_name", "N/A"))
                st.write("**Strength:**", row.get("strength", "N/A"))
                st.write("**Form:**", row.get("dosage_form", "N/A"))
                st.write(
                    "**Prescription type:**",
                    row.get("prescription_type", "N/A"),
                )


# ---------------------------------------------------------
# SEARCH RESULTS
# ---------------------------------------------------------
if submitted:
    st.divider()
    st.header("💊 Search Results")

    if not query.strip():
        st.warning("Please enter a medicine name.")
    else:
        results = search_medicines(query)

        if selected_categories:
            results = results[results["category"].isin(selected_categories)]

        if selected_prescription:
            results = results[
                results["prescription_type"].isin(selected_prescription)
            ]

        if results.empty:
            st.warning("No medicines match your search and selected filters.")
        else:
            st.success(f"{len(results)} matching catalogue entries found.")
            st.caption(
                "Similarity is a text-matching score only. It does not establish "
                "medical equivalence, safety, or suitability."
            )
            display_products(results)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------
st.divider()
st.header("Why Choose MediFind AI?")

feature_1, feature_2, feature_3 = st.columns(3)

with feature_1:
    with st.container(border=True):
        st.markdown("### 🤖 AI-Powered Search")
        st.write(
            "Find relevant catalogue entries using TF-IDF and cosine similarity."
        )

with feature_2:
    with st.container(border=True):
        st.markdown("### 💊 Medicine Catalogue")
        st.write(
            "Explore medicine names, brands, strengths and categories."
        )

with feature_3:
    with st.container(border=True):
        st.markdown("### 📍 Store Locator")
        st.write(
            "Medical store discovery is planned as the next project feature."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("### 💊 MediFind AI")
st.caption(
    "Smart Medicine Search & Medical Store Locator | "
    "AI & Data Science Academic Mini Project"
)
st.caption(
    "This application uses demonstration catalogue data. "
    "Shelf displays are illustrative and do not indicate actual stock availability. "
    "Search similarity does not establish medical equivalence, safety or suitability. "
    "Consult a qualified healthcare professional before making medicine decisions."
)
