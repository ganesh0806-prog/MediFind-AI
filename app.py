
import pickle
import re
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediFind AI - Virtual Pharmacy",
    page_icon="💊",
    layout="wide",
)


# =========================================================
# 2. FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "medifind_model.pkl"

PURCHASE_FILE = BASE_DIR / "purchase_history.csv"


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model file not found. "
            "Place medifind_model.pkl inside the models folder."
        )

    with MODEL_PATH.open("rb") as file:

        return pickle.load(file)


try:

    model = load_model()

    tfidf = model["tfidf_vectorizer"]

    tfidf_matrix = model["tfidf_matrix"]

    medicines = model["medicines"].copy()

except Exception as error:

    st.error(f"Unable to load MediFind model: {error}")

    st.info(
        "Check that models/medifind_model.pkl "
        "exists in your GitHub repository."
    )

    st.stop()


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "medicine_name",
    "generic_name",
    "brand_name",
    "strength",
    "dosage_form",
    "prescription_type",
    "category",
]

missing_columns = [
    column
    for column in required_columns
    if column not in medicines.columns
]

if missing_columns:

    st.error(
        "Missing columns in medicine dataset: "
        + ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# 5. CREATE VIRTUAL PHARMACY LOCATIONS
# =========================================================

RACKS = ["A", "B", "C", "D", "E"]

ROWS_PER_RACK = 10

SLOTS_PER_ROW = 20

TOTAL_POSITIONS = (
    len(RACKS)
    * ROWS_PER_RACK
    * SLOTS_PER_ROW
)


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
        [
            "_medicine_key",
            "medicine_name",
            "strength",
            "dosage_form"
        ]
    ]
    .drop_duplicates("_medicine_key")
    .sort_values("_medicine_key")
    .reset_index(drop=True)
)


# Check available virtual shelf positions
if len(unique_medicines) > TOTAL_POSITIONS:

    st.error(
        "There are more unique medicines than available "
        "virtual shelf positions. Increase the number of racks."
    )

    st.stop()


location_map = {}


for index, row in unique_medicines.iterrows():

    rack_index = index // (
        ROWS_PER_RACK * SLOTS_PER_ROW
    )

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

        "slot": slot_number
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


# =========================================================
# 6. SEARCH FUNCTIONS
# =========================================================

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
        results["generic_match"]
        | results["name_match"]
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
        subset=[
            "generic_name",
            "brand_name",
            "strength"
        ]
    )

    return matches.head(top_n)


# =========================================================
# 7. PURCHASE RECORD FUNCTION
# =========================================================

def record_purchase(row, quantity):

    # Indian Standard Time
    purchase_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    transaction_id = str(uuid.uuid4())[:8].upper()

    purchase_record = {

        "transaction_id": transaction_id,

        "medicine_name": row["medicine_name"],

        "generic_name": row["generic_name"],

        "brand_name": row["brand_name"],

        "strength": row["strength"],

        "quantity": int(quantity),

        "rack": row["Rack"],

        "row": int(row["Row"]),

        "slot": int(row["Slot"]),

        "purchase_date": purchase_time.strftime("%d-%m-%Y"),

        "day": purchase_time.strftime("%A"),

        "purchase_time": purchase_time.strftime("%I:%M:%S %p"),

        "year": purchase_time.year,

        "timestamp": purchase_time.isoformat(),

    }

    new_record = pd.DataFrame([purchase_record])

    if PURCHASE_FILE.exists():

        new_record.to_csv(
            PURCHASE_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_record.to_csv(
            PURCHASE_FILE,
            index=False
        )

    return purchase_record


# =========================================================
# 8. DISPLAY PURCHASE RECEIPT
# =========================================================

def display_receipt(record):

    st.success("Purchase recorded successfully!")

    with st.container(border=True):

        st.subheader("🧾 Purchase Receipt")

        st.write(
            "**Transaction ID:**",
            record["transaction_id"]
        )

        st.write(
            "**Medicine:**",
            record["medicine_name"]
        )

        st.write(
            "**Generic Name:**",
            record["generic_name"]
        )

        st.write(
            "**Quantity Purchased:**",
            record["quantity"]
        )

        st.divider()

        st.write(
            "**Purchase Date:**",
            record["purchase_date"]
        )

        st.write(
            "**Day:**",
            record["day"]
        )

        st.write(
            "**Purchase Time:**",
            record["purchase_time"]
        )

        st.write(
            "**Year:**",
            record["year"]
        )

        st.divider()

        st.write(
            f"**Shelf Location:** Rack {record['rack']} "
            f"| Row {record['row']} "
            f"| Slot {record['slot']}"
        )

        st.caption(
            "Time recorded using Indian Standard Time (IST)."
        )


# =========================================================
# 9. PAGE HEADER
# =========================================================

st.title("💊 MediFind AI")

st.caption(
    "SMART MEDICINE SEARCH & VIRTUAL PHARMACY"
)

st.success(
    "Discover Medicines. Find Their Shelf Location."
)

st.write(
    "Search medicines, locate their virtual pharmacy "
    "rack and row, and record purchase transactions."
)

st.divider()


# =========================================================
# 10. MEDICINE SEARCH AND FILTERS
# =========================================================

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

    with st.form("medicine_search_form"):

        query = st.text_input(
            "Medicine name",
            placeholder="Search Crocin, Dolo, Paracetamol..."
        )

        st.divider()

        st.write("**Choose Medicine Category**")

        selected_categories = []

        category_columns = st.columns(3)

        for index, category in enumerate(categories):

            with category_columns[index % 3]:

                if st.checkbox(
                    category,
                    key=f"category_{category}"
                ):

                    selected_categories.append(category)

        st.divider()

        st.write("**Choose Prescription Type**")

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
                    key=f"prescription_{prescription}"
                ):

                    selected_prescription.append(prescription)

        submitted = st.form_submit_button(
            "🔍 Search Medicines",
            use_container_width=True
        )


# =========================================================
# 11. SAVE SEARCH IN SESSION
# This keeps results visible after a purchase.
# =========================================================

if submitted:

    st.session_state["last_query"] = query

    st.session_state["last_categories"] = selected_categories

    st.session_state["last_prescriptions"] = selected_prescription

    st.session_state["has_searched"] = True


# =========================================================
# 12. DISPLAY MEDICINE LOCATION
# =========================================================

def display_location(row):

    st.success("📍 Medicine location found!")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "RACK",
            row["Rack"]
        )

    with col2:

        st.metric(
            "ROW",
            int(row["Row"])
        )

    with col3:

        st.metric(
            "SLOT",
            int(row["Slot"])
        )

    st.info(
        f"Go to Rack {row['Rack']} "
        f"→ Row {int(row['Row'])} "
        f"→ Slot {int(row['Slot'])}"
    )


# =========================================================
# 13. VIRTUAL PHARMACY LAYOUT
# =========================================================

def display_rack_layout(selected_medicine):

    selected_rack = selected_medicine["Rack"]

    selected_row = int(selected_medicine["Row"])

    selected_slot = int(selected_medicine["Slot"])

    st.header("🏪 Virtual Pharmacy Layout")

    st.caption(
        "The highlighted row shows the selected medicine's "
        "demonstration location."
    )

    rack_columns = st.columns(5)

    for rack_index, rack_name in enumerate(RACKS):

        with rack_columns[rack_index]:

            st.markdown(f"### Rack {rack_name}")

            with st.container(border=True):

                for row_number in range(
                    1,
                    ROWS_PER_RACK + 1
                ):

                    if (
                        rack_name == selected_rack
                        and row_number == selected_row
                    ):

                        st.success(
                            f"💊 ROW {row_number}\n\n"
                            f"{selected_medicine['medicine_name']}\n\n"
                            f"Slot {selected_slot}"
                        )

                    else:

                        st.caption(
                            f"ROW {row_number} "
                            "| Medicine slots"
                        )


# =========================================================
# 14. SEARCH RESULTS AND PURCHASE BUTTONS
# =========================================================

if st.session_state.get("has_searched", False):

    saved_query = st.session_state.get(
        "last_query",
        ""
    )

    saved_categories = st.session_state.get(
        "last_categories",
        []
    )

    saved_prescriptions = st.session_state.get(
        "last_prescriptions",
        []
    )

    st.divider()

    st.header("💊 Search Results")

    if not saved_query.strip():

        st.warning("Please enter a medicine name.")

    else:

        results = search_medicines(saved_query)

        if saved_categories:

            results = results[
                results["category"].isin(saved_categories)
            ]

        if saved_prescriptions:

            results = results[
                results["prescription_type"].isin(
                    saved_prescriptions
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

            st.caption(
                "Search similarity is text matching only. "
                "It does not establish medical equivalence, "
                "safety or suitability."
            )

            # Show the first result's shelf location
            first_result = results.iloc[0]

            display_location(first_result)

            display_rack_layout(first_result)

            st.divider()

            st.subheader("Matching Medicines")

            # Display individual medicine cards
            for index, (_, row) in enumerate(results.iterrows()):

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
                            "**Dosage Form:**",
                            row["dosage_form"]
                        )

                        st.write(
                            "**Category:**",
                            row["category"]
                        )

                        st.write(
                            "**Prescription Type:**",
                            row["prescription_type"]
                        )

                    st.info(
                        f"📍 Rack {row['Rack']} "
                        f"| Row {int(row['Row'])} "
                        f"| Slot {int(row['Slot'])}"
                    )

                    st.divider()

                    # Purchase form for each medicine
                    with st.form(
                        f"purchase_form_{index}_{row.name}"
                    ):

                        st.write("**🛒 Purchase Medicine**")

                        quantity = st.number_input(
                            "Quantity",
                            min_value=1,
                            max_value=100,
                            value=1,
                            step=1,
                            key=f"quantity_{index}_{row.name}"
                        )

                        purchase_clicked = st.form_submit_button(
                            "🛒 Confirm Purchase",
                            use_container_width=True
                        )

                        if purchase_clicked:

                            record = record_purchase(
                                row,
                                quantity
                            )

                            st.session_state[
                                "latest_purchase"
                            ] = record

                            st.rerun()


# =========================================================
# 15. SHOW LATEST PURCHASE RECEIPT
# =========================================================

if "latest_purchase" in st.session_state:

    st.divider()

    display_receipt(
        st.session_state["latest_purchase"]
    )


# =========================================================
# 16. PURCHASE HISTORY
# =========================================================

st.divider()

st.header("📋 Purchase History")


if PURCHASE_FILE.exists():

    try:

        purchase_history = pd.read_csv(
            PURCHASE_FILE
        )

        if not purchase_history.empty:

            st.success(
                f"Total purchase transactions: "
                f"{len(purchase_history)}"
            )

            st.dataframe(
                purchase_history,
                use_container_width=True,
                hide_index=True
            )

            csv_data = purchase_history.to_csv(
                index=False
            )

            st.download_button(
                "📥 Download Purchase History",
                data=csv_data,
                file_name="medifind_purchase_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info("No purchases recorded yet.")

    except Exception as error:

        st.error(
            f"Unable to read purchase history: {error}"
        )

else:

    st.info(
        "No purchases recorded yet. "
        "Purchase records will appear here."
    )


# =========================================================
# 17. FEATURES
# =========================================================

st.divider()

st.header("Why Choose MediFind AI?")

feature1, feature2, feature3 = st.columns(3)

with feature1:

    with st.container(border=True):

        st.markdown("### 🤖 AI-Powered Search")

        st.write(
            "Find medicines using TF-IDF "
            "and cosine similarity."
        )

with feature2:

    with st.container(border=True):

        st.markdown("### 🏪 Virtual Pharmacy")

        st.write(
            "Find the demonstration rack, "
            "row and shelf position."
        )

with feature3:

    with st.container(border=True):

        st.markdown("### 📋 Purchase History")

        st.write(
            "Record purchase quantity, date, "
            "day, time and year."
        )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.markdown("### 💊 MediFind AI")

st.caption(
    "Smart Medicine Search & Virtual Pharmacy | "
    "AI & Data Science Academic Mini Project"
)

st.caption(
    "Rack, row and slot assignments are simulated "
    "for this academic demonstration. Purchase records "
    "are demonstration records and do not represent "
    "verified payments or real pharmacy sales. "
    "Search similarity does not establish medical "
    "equivalence, safety or suitability."
)
