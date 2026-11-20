

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
    page_title="MediFind AI - Virtual Pharmacy",
    page_icon="💊",
    layout="wide",
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f9f6;
}
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce9e0;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent
        / "models"
        / "medifind_model.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "medifind_model.pkl not found inside models folder."
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


try:
    model = load_model()

    tfidf = model["tfidf_vectorizer"]
    tfidf_matrix = model["tfidf_matrix"]
    medicines = model["medicines"].copy()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ---------------------------------------------------------
# CHECK COLUMNS
# ---------------------------------------------------------
required_columns = [
    "medicine_name",
    "generic_name",
    "brand_name",
    "strength",
    "dosage_form",
    "prescription_type",
    "category",
]

missing = [
    column for column in required_columns
    if column not in medicines.columns
]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()


# ---------------------------------------------------------
# CREATE DEMONSTRATION PHARMACY LOCATIONS
# 5 racks x 10 rows x 20 slots = 1000 positions
# ---------------------------------------------------------
RACKS = ["A", "B", "C", "D", "E"]
ROWS_PER_RACK = 10
SLOTS_PER_ROW = 20


def medicine_key(row):

    return "|".join([
        str(row["medicine_name"]).strip().lower(),
        str(row["strength"]).strip().lower(),
        str(row["dosage_form"]).strip().lower()
    ])


medicines["_medicine_key"] = medicines.apply(
    medicine_key,
    axis=1
)

unique_medicines = (
    medicines[
        ["_medicine_key", "medicine_name", "strength", "dosage_form"]
    ]
    .drop_duplicates("_medicine_key")
    .sort_values("_medicine_key")
    .reset_index(drop=True)
)

location_map = {}

for index, row in unique_medicines.iterrows():

    rack_index = index // (
        ROWS_PER_RACK * SLOTS_PER_ROW
    )

    if rack_index >= len(RACKS):
        rack_index = len(RACKS) - 1

    position_in_rack = index % (
        ROWS_PER_RACK * SLOTS_PER_ROW
    )

    row_number = (
        position_in_rack // SLOTS_PER_ROW
    ) + 1

    slot_number = (
        position_in_rack % SLOTS_PER_ROW
    ) + 1

    location_map[row["_medicine_key"]] = {
        "rack": RACKS[rack_index],
        "row": row_number,
        "slot": slot_number,
    }


medicines["Rack"] = medicines["_medicine_key"].map(
    lambda key: location_map[key]["rack"]
)

medicines["Row"] = medicines["_medicine_key"].map(
    lambda key: location_map[key]["row"]
)

medicines["Slot"] = medicines["_medicine_key"].map(
    lambda key: location_map[key]["slot"]
)


# ---------------------------------------------------------
# SEARCH FUNCTIONS
# ---------------------------------------------------------
def normalize_text(value):

    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def search_medicines(query, top_n=100):

    query = query.strip()

    query_vector = tfidf.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).ravel()

    results = medicines.copy()

    results["similarity_score"] = scores

    normalized_query = normalize_text(query)

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
        results["generic_normalized"].eq(normalized_query)
    )

    results["final_score"] = (
        results["similarity_score"]
        + results["exact_generic"].astype(int) * 1.0
        + results["generic_match"].astype(int) * 0.5
        + results["name_match"].astype(int) * 0.3
    )

    matches = results[
        results["generic_match"] | results["name_match"]
    ].copy()

    if matches.empty:
        matches = results[
            results["similarity_score"] > 0
        ].copy()

    matches = matches.sort_values(
        "final_score",
        ascending=False
    )

    matches = matches.drop_duplicates(
        subset=["generic_name", "brand_name", "strength"]
    )

    return matches.head(top_n)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("💊 MediFind AI")

st.caption("SMART MEDICINE SEARCH & VIRTUAL PHARMACY")

st.success("Discover Medicines. Find Their Exact Shelf Location.")

st.write(
    "Search for a medicine and discover its virtual pharmacy "
    "rack, row and shelf position."
)

st.divider()


# ---------------------------------------------------------
# MEDICINE SEARCH AND FILTERS
# ---------------------------------------------------------
st.header("🔍 Find Your Medicine")

categories = sorted(
    medicines["category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

prescription_types = sorted(
    medicines["prescription_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

with st.container(border=True):

    with st.form("medicine_search"):

        query = st.text_input(
            "Medicine name",
            placeholder="Search Crocin, Dolo, Paracetamol..."
        )

        st.write("**Medicine Category**")

        selected_categories = []

        category_columns = st.columns(3)

        for index, category in enumerate(categories):

            with category_columns[index % 3]:

                if st.checkbox(
                    category,
                    key=f"cat_{category}"
                ):
                    selected_categories.append(category)

        st.divider()

        st.write("**Prescription Type**")

        selected_prescription = []

        prescription_columns = st.columns(
            max(1, min(3, len(prescription_types)))
        )

        for index, prescription in enumerate(prescription_types):

            with prescription_columns[
                index % len(prescription_columns)
            ]:

                if st.checkbox(
                    prescription,
                    key=f"pres_{prescription}"
                ):
                    selected_prescription.append(prescription)

        submitted = st.form_submit_button(
            "🔍 Search Medicine",
            use_container_width=True
        )


# ---------------------------------------------------------
# DISPLAY LOCATION DETAILS
# ---------------------------------------------------------
def display_location(row):

    rack = row["Rack"]
    row_number = int(row["Row"])
    slot = int(row["Slot"])

    st.success("📍 Medicine location found!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RACK", rack)

    with col2:
        st.metric("ROW", row_number)

    with col3:
        st.metric("SLOT", slot)

    st.info(
        f"Go to **Rack {rack} → Row {row_number} → Slot {slot}**"
    )


# ---------------------------------------------------------
# VIRTUAL PHARMACY RACK VIEW
# ---------------------------------------------------------
def display_rack_layout(selected_medicine):

    selected_rack = selected_medicine["Rack"]
    selected_row = int(selected_medicine["Row"])
    selected_slot = int(selected_medicine["Slot"])

    st.header("🏪 Virtual Pharmacy Layout")

    st.write(
        "The highlighted row shows the medicine's assigned "
        "demonstration location."
    )

    rack_columns = st.columns(5)

    for rack_index, rack_name in enumerate(RACKS):

        with rack_columns[rack_index]:

            st.markdown(f"### Rack {rack_name}")

            with st.container(border=True):

                for row_number in range(1, ROWS_PER_RACK + 1):

                    if (
                        rack_name == selected_rack
                        and row_number == selected_row
                    ):

                        st.success(
                            f"📦 ROW {row_number}\n\n"
                            f"💊 {selected_medicine['medicine_name']}\n\n"
                            f"Slot {selected_slot}"
                        )

                    else:

                        st.caption(
                            f"ROW {row_number}  |  Medicine slots"
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

            results = results[
                results["category"].isin(selected_categories)
            ]

        if selected_prescription:

            results = results[
                results["prescription_type"].isin(
                    selected_prescription
                )
            ]

        if results.empty:

            st.warning(
                "No medicines found. Try another medicine name "
                "or change the selected filters."
            )

        else:

            st.success(
                f"{len(results)} matching catalogue entries found."
            )

            first_result = results.iloc[0]

            # Exact demonstration location for top result
            display_location(first_result)

            # Display rack and row
            display_rack_layout(first_result)

            st.divider()

            st.subheader("Other Matching Medicines")

            st.caption(
                "Each medicine has its own assigned demonstration "
                "rack, row and slot."
            )

            for _, row in results.iterrows():

                with st.container(border=True):

                    st.markdown(
                        f"### 💊 {row['medicine_name']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**Generic:**",
                            row["generic_name"]
                        )

                        st.write(
                            "**Brand:**",
                            row["brand_name"]
                        )

                        st.write(
                            "**Strength:**",
                            row["strength"]
                        )

                    with col2:

                        st.write(
                            "**Category:**",
                            row["category"]
                        )

                        st.write(
                            "**Form:**",
                            row["dosage_form"]
                        )

                        st.write(
                            "**Prescription:**",
                            row["prescription_type"]
                        )

                    st.info(
                        f"📍 Rack {row['Rack']} | "
                        f"Row {int(row['Row'])} | "
                        f"Slot {int(row['Slot'])}"
                    )

            st.caption(
                "Search similarity is text matching only. "
                "It does not establish medical equivalence, "
                "safety or suitability."
            )


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------
st.divider()

st.header("Why Choose MediFind AI?")

col1, col2, col3 = st.columns(3)

with col1:

    with st.container(border=True):

        st.markdown("### 🤖 AI Search")

        st.write(
            "Search medicines using TF-IDF "
            "and cosine similarity."
        )

with col2:

    with st.container(border=True):

        st.markdown("### 🏪 Virtual Pharmacy")

        st.write(
            "Explore the demonstration rack, "
            "row and shelf positions."
        )

with col3:

    with st.container(border=True):

        st.markdown("### 📍 Store Locator")

        st.write(
            "Medical store discovery is planned "
            "as a future feature."
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
    "Rack, row and slot assignments are simulated for this "
    "academic demonstration. They do not represent the actual "
    "layout or stock of a real pharmacy. Search similarity does "
    "not establish medical equivalence, safety or suitability."
)
