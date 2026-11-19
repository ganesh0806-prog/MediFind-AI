
import streamlit as st
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="MediFind AI",
    page_icon="💊",
    layout="wide"
)

# -------------------------------
# LOAD MACHINE LEARNING MODEL
# -------------------------------

@st.cache_resource
def load_model():
    with open("models/medifind_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model

try:
    model = load_model()

    tfidf = model["tfidf_vectorizer"]
    tfidf_matrix = model["tfidf_matrix"]
    medicines = model["medicines"]

except Exception as e:
    st.error(f"Could not load the MediFind model: {e}")
    st.stop()


# -------------------------------
# TEXT NORMALIZATION
# -------------------------------

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


# -------------------------------
# MEDICINE SEARCH
# -------------------------------

def search_medicines(query, top_n=10):

    normalized_query = normalize_text(query)

    query_vector = tfidf.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = medicines.copy()
    results["similarity_score"] = scores

    results["generic_normalized"] = (
        results["generic_name"]
        .fillna("")
        .apply(normalize_text)
    )

    results["name_normalized"] = (
        results["medicine_name"]
        .fillna("")
        .apply(normalize_text)
    )

    results["brand_normalized"] = (
        results["brand_name"]
        .fillna("")
        .apply(normalize_text)
    )

    # Identify direct text matches
    results["generic_match"] = (
        results["generic_normalized"].str.contains(
            re.escape(normalized_query),
            na=False
        )
    )

    results["name_match"] = (
        results["name_normalized"].str.contains(
            re.escape(normalized_query),
            na=False
        )
        |
        results["brand_normalized"].str.contains(
            re.escape(normalized_query),
            na=False
        )
    )

    results["exact_generic"] = (
        results["generic_normalized"] == normalized_query
    )

    # Ranking
    results["final_score"] = (
        results["similarity_score"]
        + results["exact_generic"].astype(int) * 1.0
        + results["generic_match"].astype(int) * 0.5
        + results["name_match"].astype(int) * 0.3
    )

    # Prioritize direct matches
    direct = results[
        results["generic_match"] | results["name_match"]
    ].copy()

    # Use similarity fallback only if no direct matches exist
    if direct.empty:
        direct = results[
            results["similarity_score"] >= 0.15
        ].copy()

    direct = direct.sort_values(
        "final_score",
        ascending=False
    )

    direct = direct.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return direct.head(top_n)


# -------------------------------
# WEBSITE DESIGN
# -------------------------------

st.title("💊 MediFind AI")

st.subheader(
    "Smart Medicine Search & Medical Store Locator"
)

st.write(
    "Find medicines from a searchable medicine catalogue "
    "using machine learning."
)

st.divider()

# -------------------------------
# SEARCH SECTION
# -------------------------------

st.header("🔍 Find Your Medicine")

search_query = st.text_input(
    "Enter medicine name, generic name or brand",
    placeholder="Example: Paracetamol, Crocin, Vitamin C"
)

if st.button("Search Medicines", type="primary"):

    if not search_query.strip():
        st.warning("Please enter a medicine name.")

    else:
        results = search_medicines(search_query)

        if results.empty:
            st.warning("No matching medicines found.")

        else:
            st.success(
                f"Found {len(results)} matching medicine records."
            )

            # Separate single generic and combination products
            generic_values = (
                results["generic_name"].fillna("").astype(str)
            )

            combination_mask = generic_values.str.contains(
                r"\+|,",
                regex=True
            )

            exact_results = results[~combination_mask]
            combination_results = results[combination_mask]

            st.subheader("Matching Medicines")

            if not exact_results.empty:
                st.write("**Catalogue matches**")
                st.dataframe(
                    exact_results[
                        [
                            "medicine_name",
                            "generic_name",
                            "brand_name",
                            "strength",
                            "similarity_score"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            if not combination_results.empty:
                st.write("**Combination products**")
                st.dataframe(
                    combination_results[
                        [
                            "medicine_name",
                            "generic_name",
                            "brand_name",
                            "strength",
                            "similarity_score"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

# -------------------------------
# ABOUT SECTION
# -------------------------------

st.divider()

st.header("About MediFind AI")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Medicine Records", len(medicines))

with col2:
    st.metric("Search Algorithm", "TF-IDF")

with col3:
    st.metric("Similarity Method", "Cosine")

st.info(
    "MediFind is an academic demonstration project. "
    "Its catalogue and inventory are demonstration data. "
    "Text similarity does not establish medical equivalence "
    "or suitability. Consult a qualified healthcare professional "
    "before making medicine decisions."
)

st.caption("MediFind AI | AI & Data Science Mini Project")

