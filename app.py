import html
import pickle
import re
from pathlib import Path

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediFind AI",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f5f9f6;
        color: #19382b;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .navbar {
        background: #ffffff;
        border: 1px solid #e3ece6;
        border-radius: 15px;
        padding: 18px 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
    }

    .logo {
        font-size: 27px;
        font-weight: 800;
        color: #07805a;
    }

    .logo span { color: #203b2e; }

    .nav-text {
        font-size: 13px;
        color: #687c70;
        font-weight: 600;
    }

    .hero {
        background: linear-gradient(120deg, #ddf4e6, #f1faf4);
        border: 1px solid #d8eee0;
        border-radius: 20px;
        padding: 35px 40px;
        margin-bottom: 25px;
    }

    .hero-tag {
        display: inline-block;
        background: #ffffff;
        color: #07805a;
        padding: 7px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 800;
    }

    .hero h1 {
        color: #173b2c;
        font-size: 42px;
        line-height: 1.2;
        font-weight: 800;
        margin: 18px 0 10px 0;
    }

    .hero p {
        color: #60786a;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #193b2c;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #809087;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .shelf-heading {
        font-size: 19px;
        font-weight: 800;
        color: #234532;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .shelf-row {
        display: flex;
        gap: 15px;
        overflow-x: auto;
        padding: 16px 12px 0;
        background: linear-gradient(#eef4ef, #e5eee7);
        border: 1px solid #dce7df;
        border-radius: 13px 13px 0 0;
    }

    .shelf-product {
        flex: 0 0 125px;
        height: 155px;
        background: white;
        border: 1px solid #dce5df;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 9px;
        box-shadow: 0 3px 7px rgba(0,0,0,.07);
    }

    .pack {
        width: 68px;
        height: 85px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
        font-size: 10px;
        font-weight: 800;
        padding: 5px;
    }

    .pack.green { background: linear-gradient(135deg,#0e9d6b,#075c43); }
    .pack.blue { background: linear-gradient(135deg,#3c8fce,#17538a); }
    .pack.orange { background: linear-gradient(135deg,#f2a34d,#c96a1e); }
    .pack.purple { background: linear-gradient(135deg,#9c79c9,#60428b); }
    .pack.red { background: linear-gradient(135deg,#e66b6b,#a83232); }

    .shelf-product small {
        font-size: 10px;
        font-weight: 700;
        color: #52665b;
        margin-top: 8px;
        text-align: center;
    }

    .shelf-board {
        height: 13px;
        background: linear-gradient(#c99a6b,#95683f);
        border-radius: 2px;
        box-shadow: 0 5px 8px rgba(0,0,0,.12);
        margin-bottom: 20px;
    }

    .product-card {
        background: white;
        border: 1px solid #e5eee8;
        border-radius: 15px;
        padding: 19px;
        min-height: 225px;
        margin-bottom: 15px;
        box-shadow: 0 4px 14px rgba(20,70,45,.04);
    }

    .product-icon {
        font-size: 29px;
        background: #eaf7ef;
        border-radius: 12px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 13px;
    }

    .product-name {
        color: #203c2e;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .product-detail {
        color: #718279;
        font-size: 12px;
        margin: 6px 0;
    }

    .product-tag {
        display: inline-block;
        background: #e8f7ee;
        color: #087f5b;
        padding: 6px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 800;
        margin-top: 10px;
    }

    .feature-card {
        background: white;
        border: 1px solid #e5eee8;
        border-radius: 14px;
        padding: 22px;
        min-height: 150px;
    }

    .feature-icon { font-size: 27px; }

    .feature-title {
        color: #203c2e;
        font-size: 16px;
        font-weight: 800;
        margin-top: 9px;
    }

    .feature-description {
        color: #78887f;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 7px;
    }

    .footer-box {
        background: #173b2c;
        color: #d8e9df;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 45px;
    }

    .footer-logo {
        color: white;
        font-size: 23px;
        font-weight: 800;
    }

    .footer-text {
        color: #b8cfc0;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 10px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #087f5b;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #066b4d;
        color: white;
    }

    .stTextInput input {
        border-radius: 10px;
        min-height: 45px;
        border: 1px solid #dce8e0;
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
    st.info("Check that models/medifind_model.pkl is present in your GitHub repository.")
    st.stop()


# Check expected columns so missing data is explained clearly.
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
    st.error("The model's medicine data is missing these columns: " + ", ".join(missing_columns))
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

    results["generic_normalized"] = results["generic_name"].fillna("").apply(normalize_text)
    results["name_normalized"] = results["medicine_name"].fillna("").apply(normalize_text)
    results["brand_normalized"] = results["brand_name"].fillna("").apply(normalize_text)

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

    direct_matches = results[results["generic_match"] | results["name_match"]].copy()

    if direct_matches.empty:
        direct_matches = results[results["similarity_score"] > 0].copy()

    direct_matches = direct_matches.sort_values("final_score", ascending=False)
    direct_matches = direct_matches.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return direct_matches.head(top_n)


# ---------------------------------------------------------
# HEADER AND HERO
# Every HTML block is rendered with st.markdown(...,
# unsafe_allow_html=True). Do not use st.write for raw HTML.
# ---------------------------------------------------------
st.markdown(
    """
    <div class="navbar">
        <div class="logo">💊 MediFind<span> AI</span></div>
        <div class="nav-text">SMART MEDICINE DISCOVERY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">✦ YOUR SMART PHARMACY</div>
        <h1>Discover Medicines.<br>Search Smarter.</h1>
        <p>
            Explore medicines by name, brand or generic name.
            Find relevant catalogue information using AI-powered medicine search.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SEARCH FORM
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">🔍 Find Your Medicine</div>',
    unsafe_allow_html=True,
)

with st.form("medicine_search_form"):
    search_col, button_col = st.columns([5, 1])

    with search_col:
        query = st.text_input(
            "Medicine name",
            placeholder="Try Paracetamol, Crocin, Vitamin C...",
        )

    with button_col:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)


# ---------------------------------------------------------
# CHECKBOX FILTERS
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">☑️ Select Your Preferences</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Tick one or more options to filter medicines.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Medicine Categories")
    all_categories = sorted(medicines["category"].dropna().astype(str).unique().tolist())
    selected_categories = []

    category_columns = st.columns(3)
    for index, category in enumerate(all_categories):
        with category_columns[index % 3]:
            if st.checkbox(category, key=f"category_{category}"):
                selected_categories.append(category)

    st.divider()
    st.markdown("#### Prescription Type")

    prescription_types = sorted(
        medicines["prescription_type"].dropna().astype(str).unique().tolist()
    )
    selected_prescription = []

    prescription_columns = st.columns(max(1, min(3, len(prescription_types))))
    for index, prescription in enumerate(prescription_types):
        with prescription_columns[index % len(prescription_columns)]:
            if st.checkbox(prescription, key=f"prescription_{prescription}"):
                selected_prescription.append(prescription)


# ---------------------------------------------------------
# ILLUSTRATIVE PHARMACY SHELVES
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">🏪 Explore Our Pharmacy</div>',
    unsafe_allow_html=True,
)

shelves = [
    {
        "title": "💊 Pain Relief & General Medicines",
        "products": [
            ("PARACETAMOL", "green", "Paracetamol"),
            ("CROCIN", "blue", "Crocin"),
            ("DOLO", "orange", "Dolo"),
            ("IBUPROFEN", "purple", "Ibuprofen"),
            ("CALPOL", "red", "Calpol"),
        ],
    },
    {
        "title": "🌿 Vitamins & Supplements",
        "products": [
            ("VITAMIN C", "orange", "Vitamin C"),
            ("BECOSULES", "red", "Becosules"),
            ("VITAMIN D", "blue", "Vitamin D"),
            ("MULTIVITAMIN", "green", "Multivitamin"),
            ("ZINC", "purple", "Zinc"),
        ],
    },
    {
        "title": "🩺 Healthcare & Wellness",
        "products": [
            ("ALLERGY CARE", "blue", "Allergy Care"),
            ("DIGESTIVE", "green", "Digestive Care"),
            ("HEALTH CARE", "purple", "Healthcare"),
            ("SKIN CARE", "orange", "Skin Care"),
            ("WELLNESS", "red", "Wellness"),
        ],
    },
]

for shelf in shelves:
    st.markdown(
        f'<div class="shelf-heading">{html.escape(shelf["title"])}</div>',
        unsafe_allow_html=True,
    )

    shelf_html = '<div class="shelf-row">'
    for label, color, product_name in shelf["products"]:
        shelf_html += f"""
            <div class="shelf-product">
                <div class="pack {color}">{html.escape(label)}</div>
                <small>{html.escape(product_name)}</small>
            </div>
        """

    shelf_html += '</div><div class="shelf-board"></div>'
    st.markdown(shelf_html, unsafe_allow_html=True)

st.caption("Illustrative shelf display only. It does not represent verified stock.")


# ---------------------------------------------------------
# CATALOGUE COUNTS
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">📂 Medicine Catalogue</div>',
    unsafe_allow_html=True,
)

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
            name = html.escape(str(row.get("medicine_name", "N/A")))
            generic = html.escape(str(row.get("generic_name", "N/A")))
            brand = html.escape(str(row.get("brand_name", "N/A")))
            strength = html.escape(str(row.get("strength", "N/A")))
            dosage = html.escape(str(row.get("dosage_form", "N/A")))
            prescription = html.escape(str(row.get("prescription_type", "N/A")))

            card_html = f"""
                <div class="product-card">
                    <div class="product-icon">💊</div>
                    <div class="product-name">{name}</div>
                    <div class="product-detail">Generic: {generic}</div>
                    <div class="product-detail">Brand: {brand}</div>
                    <div class="product-detail">Strength: {strength}</div>
                    <div class="product-detail">Form: {dosage}</div>
                    <span class="product-tag">{prescription}</span>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# SEARCH RESULTS
# ---------------------------------------------------------
if submitted:
    st.markdown("---")

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

        st.markdown(
            '<div class="section-title">💊 Search Results</div>',
            unsafe_allow_html=True,
        )

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
st.markdown("---")
st.markdown(
    '<div class="section-title">Why Choose MediFind AI?</div>',
    unsafe_allow_html=True,
)

feature_1, feature_2, feature_3 = st.columns(3)

with feature_1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered Search</div>
            <div class="feature-description">
                Find relevant catalogue entries using TF-IDF and cosine similarity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">💊</div>
            <div class="feature-title">Medicine Catalogue</div>
            <div class="feature-description">
                Explore medicine names, brands, strengths and categories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📍</div>
            <div class="feature-title">Store Locator</div>
            <div class="feature-description">
                Medical store discovery is planned as the next project feature.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer-box">
        <div class="footer-logo">💊 MediFind AI</div>
        <div class="footer-text">
            Smart Medicine Search &amp; Medical Store Locator<br>
            AI &amp; Data Science Academic Mini Project<br><br>
            This application uses demonstration catalogue data.
            Shelf displays are illustrative and do not indicate actual stock availability.
            Search similarity does not establish medical equivalence, safety or suitability.
            Consult a qualified healthcare professional before making medicine decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
