
import streamlit as st
import pickle
import re
import html
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MediFind AI | Smart Pharmacy",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# PROFESSIONAL CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7faf8;
    color: #18352a;
}

.block-container {
    max-width: 1250px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* NAVIGATION */

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: white;
    padding: 17px 25px;
    border: 1px solid #e8efeb;
    border-radius: 14px;
    margin-bottom: 25px;
}

.logo {
    font-size: 25px;
    font-weight: 800;
    color: #087f5b;
}

.logo span {
    color: #1d3329;
}

.nav-caption {
    color: #62766b;
    font-size: 13px;
    font-weight: 600;
}

/* HERO */

.hero {
    background: linear-gradient(120deg, #e3f7ed 0%, #f2faf5 65%, #e8f5ee 100%);
    padding: 55px 45px;
    border-radius: 20px;
    border: 1px solid #d9eee2;
    margin-bottom: 28px;
}

.hero-tag {
    display: inline-block;
    color: #087f5b;
    background: white;
    border: 1px solid #d6eee0;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 15px;
}

.hero h1 {
    color: #173b2c;
    font-size: 43px;
    line-height: 1.15;
    font-weight: 800;
    margin: 0 0 14px 0;
}

.hero p {
    color: #60776b;
    font-size: 16px;
    max-width: 600px;
    line-height: 1.7;
}

/* SECTION */

.section-heading {
    font-size: 23px;
    font-weight: 800;
    color: #1b382c;
    margin-top: 32px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #809087;
    font-size: 13px;
    margin-bottom: 18px;
}

/* CATEGORY */

.category-card {
    background: white;
    border: 1px solid #e6eee9;
    border-radius: 13px;
    padding: 17px 12px;
    text-align: center;
    min-height: 100px;
}

.category-icon {
    font-size: 26px;
}

.category-name {
    color: #294638;
    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
}

/* PRODUCT CARDS */

.product-card {
    background: white;
    border: 1px solid #e6eee9;
    border-radius: 16px;
    padding: 20px;
    min-height: 245px;
    margin-bottom: 15px;
    transition: 0.2s ease;
    box-shadow: 0 4px 15px rgba(25, 65, 45, 0.035);
}

.product-card:hover {
    border-color: #9bd6b6;
    box-shadow: 0 8px 25px rgba(25, 65, 45, 0.08);
}

.product-icon {
    width: 48px;
    height: 48px;
    background: #e9f7ef;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    margin-bottom: 14px;
}

.product-name {
    color: #203c2e;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 7px;
}

.product-generic {
    color: #728279;
    font-size: 12px;
    margin-bottom: 13px;
}

.product-info {
    color: #53695e;
    font-size: 12px;
    margin: 5px 0;
}

.product-tag {
    display: inline-block;
    background: #e8f7ee;
    color: #087f5b;
    padding: 6px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 800;
    margin-top: 12px;
}

/* TRUST CARDS */

.feature-card {
    background: white;
    border: 1px solid #e6eee9;
    border-radius: 15px;
    padding: 23px;
    min-height: 150px;
}

.feature-icon {
    font-size: 26px;
    margin-bottom: 10px;
}

.feature-title {
    color: #203c2e;
    font-size: 16px;
    font-weight: 800;
}

.feature-description {
    color: #77877e;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 7px;
}

/* FOOTER */

.footer {
    background: #173b2c;
    color: #d8e9df;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 45px;
}

.footer-logo {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.footer-small {
    color: #b4cbbd;
    font-size: 12px;
    line-height: 1.8;
}

/* BUTTONS */

.stButton > button {
    background: #087f5b;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    background: #066b4d;
    color: white;
    border: none;
}

.stTextInput input {
    border: 1px solid #dce8e0;
    border-radius: 10px;
    min-height: 45px;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 9px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6eee9;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    with open("models/medifind_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
    tfidf = model["tfidf_vectorizer"]
    tfidf_matrix = model["tfidf_matrix"]
    medicines = model["medicines"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# SEARCH FUNCTION
# ==========================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def search_medicines(query, top_n=30):

    normalized_query = normalize_text(query)

    query_vector = tfidf.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = medicines.copy()
    results["similarity_score"] = scores

    results["generic_normalized"] = (
        results["generic_name"].fillna("").apply(normalize_text)
    )

    results["name_normalized"] = (
        results["medicine_name"].fillna("").apply(normalize_text)
    )

    results["brand_normalized"] = (
        results["brand_name"].fillna("").apply(normalize_text)
    )

    results["generic_match"] = results[
        "generic_normalized"
    ].str.contains(re.escape(normalized_query), na=False)

    results["name_match"] = (
        results["name_normalized"].str.contains(
            re.escape(normalized_query), na=False
        )
        |
        results["brand_normalized"].str.contains(
            re.escape(normalized_query), na=False
        )
    )

    results["exact_generic"] = (
        results["generic_normalized"] == normalized_query
    )

    results["final_score"] = (
        results["similarity_score"]
        + results["exact_generic"].astype(int) * 1.0
        + results["generic_match"].astype(int) * 0.5
        + results["name_match"].astype(int) * 0.3
    )

    direct = results[
        results["generic_match"] | results["name_match"]
    ].copy()

    if direct.empty:
        direct = results[
            results["similarity_score"] >= 0.15
        ].copy()

    direct = direct.sort_values(
        "final_score", ascending=False
    )

    direct = direct.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return direct.head(top_n)


# ==========================================
# NAVIGATION
# ==========================================

st.markdown("""
<div class="navbar">
    <div class="logo">💊 MediFind<span> AI</span></div>
    <div class="nav-caption">
        SMART MEDICINE DISCOVERY &nbsp; | &nbsp; HEALTHCARE TECHNOLOGY
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ SMART MEDICINE SEARCH</div>
    <h1>Discover Medicines.<br>Search Smarter.</h1>
    <p>
        Explore our medicine catalogue using medicine names,
        brand names or generic names with AI-powered text search.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SEARCH AREA
# ==========================================

st.markdown(
    '<div class="section-heading">Find Your Medicine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Start by entering a medicine or brand name</div>',
    unsafe_allow_html=True
)

with st.form("search_form"):

    col1, col2 = st.columns([5, 1])

    with col1:
        query = st.text_input(
            "Medicine name",
            placeholder="Try Paracetamol, Crocin, Vitamin C..."
        )

    with col2:
        st.write("")
        st.write("")
        submitted = st.form_submit_button(
            "🔍 Search",
            use_container_width=True
        )


# ==========================================
# CATEGORY SECTION
# ==========================================

st.markdown(
    '<div class="section-heading">Explore Categories</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Browse the available medicine catalogue</div>',
    unsafe_allow_html=True
)

category_icons = {
    "Vitamins & Supplements": "🌿",
    "Pain Relief": "💊",
    "Allergy": "🌸",
    "Diabetes": "🩺",
    "Antibiotics": "🧪",
    "Other": "➕"
}

category_counts = (
    medicines["category"].fillna("Other").value_counts()
)

category_cols = st.columns(5)

for i, (category, count) in enumerate(category_counts.items()):

    with category_cols[i % 5]:

        icon = category_icons.get(category, "💊")

        st.markdown(f"""
        <div class="category-card">
            <div class="category-icon">{icon}</div>
            <div class="category-name">{html.escape(str(category))}</div>
            <div style="color:#91a198;font-size:11px;">
                {count} items
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# SEARCH RESULTS
# ==========================================

if submitted:

    if not query.strip():

        st.warning("Please enter a medicine name.")

    else:

        results = search_medicines(query)

        st.markdown("---")

        if results.empty:

            st.warning("No matching medicines found.")

        else:

            st.markdown(
                '<div class="section-heading">Search Results</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"{len(results)} matching catalogue records"
            )

            selected_category = st.selectbox(
                "Filter by category",
                ["All Categories"] +
                sorted(results["category"].dropna().unique().tolist())
            )

            if selected_category != "All Categories":
                results = results[
                    results["category"] == selected_category
                ]

            if results.empty:
                st.info("No results in this category.")

            else:

                combination_mask = (
                    results["generic_name"]
                    .fillna("")
                    .astype(str)
                    .str.contains(r"\+|,", regex=True)
                )

                exact_results = results[~combination_mask]
                combination_results = results[combination_mask]

                def display_cards(data):

                    cols = st.columns(3)

                    for index, (_, row) in enumerate(data.iterrows()):

                        with cols[index % 3]:

                            name = html.escape(
                                str(row.get("medicine_name", "N/A"))
                            )

                            generic = html.escape(
                                str(row.get("generic_name", "N/A"))
                            )

                            brand = html.escape(
                                str(row.get("brand_name", "N/A"))
                            )

                            strength = html.escape(
                                str(row.get("strength", "N/A"))
                            )

                            dosage = html.escape(
                                str(row.get("dosage_form", "N/A"))
                            )

                            prescription = html.escape(
                                str(row.get("prescription_type", "N/A"))
                            )

                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-icon">💊</div>
                                <div class="product-name">{name}</div>
                                <div class="product-generic">
                                    {generic}
                                </div>
                                <div class="product-info">
                                    <b>Brand:</b> {brand}
                                </div>
                                <div class="product-info">
                                    <b>Strength:</b> {strength}
                                </div>
                                <div class="product-info">
                                    <b>Form:</b> {dosage}
                                </div>
                                <span class="product-tag">
                                    {prescription}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                if not exact_results.empty:

                    st.markdown("#### Matching catalogue entries")
                    display_cards(exact_results)

                if not combination_results.empty:

                    st.markdown("#### Combination products")
                    display_cards(combination_results)


# ==========================================
# FEATURES
# ==========================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">Why MediFind AI?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Technology designed to simplify medicine discovery</div>',
    unsafe_allow_html=True
)

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI-Powered Search</div>
        <div class="feature-description">
            Find relevant catalogue entries using
            TF-IDF and cosine similarity.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💊</div>
        <div class="feature-title">Medicine Catalogue</div>
        <div class="feature-description">
            Explore medicine brands, generic names,
            strengths and categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📍</div>
        <div class="feature-title">Store Discovery</div>
        <div class="feature-description">
            A medical store locator is planned
            as the next project feature.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    <div class="footer-logo">💊 MediFind AI</div>
    <div class="footer-small">
        Smart Medicine Search & Medical Store Locator<br>
        AI & Data Science Academic Mini Project<br><br>
        This application uses demonstration catalogue data.
        Search similarity does not establish medical equivalence,
        safety or suitability. Consult a qualified healthcare
        professional before making medicine decisions.
    </div>
</div>
""", unsafe_allow_html=True)
