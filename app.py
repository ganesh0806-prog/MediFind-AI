
import streamlit as st
import pickle
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="MediFind AI",
    page_icon="💊",
    layout="wide"
)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

@st.cache_resource
def load_model():

    with open("models/medifind_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

tfidf = model["tfidf_vectorizer"]
tfidf_matrix = model["tfidf_matrix"]
medicines = model["medicines"]

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------

def normalize_text(text):

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())

# -----------------------------
# MEDICINE SEARCH
# -----------------------------

def improved_medicine_search(search_query, top_n=10):

    query = normalize_text(search_query)

    query_vector = tfidf.transform([search_query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = medicines.copy()

    results["similarity_score"] = scores

    results["normalized_generic"] = (
        results["generic_name"]
        .fillna("")
        .apply(normalize_text)
    )

    results["normalized_name"] = (
        results["medicine_name"]
        .fillna("")
        .apply(normalize_text)
    )

    results["normalized_brand"] = (
        results["brand_name"]
        .fillna("")
        .apply(normalize_text)
    )

    results["exact_generic"] = (
        results["normalized_generic"] == query
    )

    results["generic_match"] = (
        results["normalized_generic"].str.contains(
            re.escape(query), regex=True, na=False
        )
    )

    results["name_match"] = (
        results["normalized_name"].str.contains(
            re.escape(query), regex=True, na=False
        )
        |
        results["normalized_brand"].str.contains(
            re.escape(query), regex=True, na=False
        )
    )

    results["final_score"] = (
        results["similarity_score"]
        + results["exact_generic"].astype(int) * 1.0
        + results["generic_match"].astype(int) * 0.5
        + results["name_match"].astype(int) * 0.3
    )

    results = results[
        results["generic_match"]
        | results["name_match"]
        | (results["similarity_score"] > 0)
    ]

    results = results.sort_values(
        "final_score",
        ascending=False
    )

    results = results.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return results.head(top_n)

# -----------------------------
# WEBSITE HEADER
# -----------------------------

st.title("💊 MediFind AI")
st.subheader("Smart Medicine Search and Medical Store Finder")

st.write(
    "Search medicines by name, generic name or brand name."
)

st.divider()

# -----------------------------
# MEDICINE SEARCH INTERFACE
# -----------------------------

search_query = st.text_input(
    "Enter medicine name",
    placeholder="Example: Paracetamol, Vitamin C, Cetirizine"
)

if st.button("Search Medicine", type="primary"):

    if search_query.strip():

        results = improved_medicine_search(search_query)

        if not results.empty:

            st.success(f"Found {len(results)} matching records")

            st.dataframe(
                results[
                    [
                        "medicine_name",
                        "generic_name",
                        "brand_name",
                        "category",
                        "strength",
                        "similarity_score"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning("No matching medicines found.")

    else:

        st.warning("Please enter a medicine name.")

# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "MediFind AI | Academic Mini Project | "
    "Demonstration data only. Search similarity does not establish "
    "medical equivalence or suitability. Consult a qualified pharmacist "
    "or healthcare professional before making medicine decisions."
)

