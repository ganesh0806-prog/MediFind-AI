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
