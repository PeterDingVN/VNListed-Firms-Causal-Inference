import streamlit as st

# ------------------------------
# PAGE TITLE
# ------------------------------
st.markdown(
    "<h1 style='text-align: center; color:#4A90E2;'>✨ Financial Forecast ✨</h1>",
    unsafe_allow_html=True
)

st.write("---")

# ------------------------------
# FILE UPLOAD
# ------------------------------
st.subheader("Please drag or upload data here:")
uploaded_file = st.file_uploader("Upload your financial dataset:", type=["csv", "xlsx"])

st.write("---")

# ------------------------------
# METRIC DROPDOWN
# ------------------------------
metrics = ["ROA", "ROE", "EBITDA", "Value_add", "Revenue"]
selected_metric = st.selectbox("Choose a metric to forecast:", metrics)

# ------------------------------
# PREDICT BUTTON
# ------------------------------
if st.button("🔮 Predict"):
    if uploaded_file is None:
        st.error("Please upload a dataset first.")
    else:
        # Placeholder prediction (replace with your model)
        prediction_value = 123.45  # example

        st.markdown("---")
        st.success(
            f"**{selected_metric}** for **next year** is:\n\n"
            f"### 🟦 {prediction_value}"
        )
