import streamlit as st
from src.app.server import *
import pickle as pk
import tensorflow as tf

# Page config
st.title("Forecasting")

st.sidebar.title("Metrics")
metric = st.sidebar.radio(
        "Choose one metric you want to forecast",
        ('ROA', 'ROE', 'Value_add', 'Revenue', 'EBITDA')
        )
st.sidebar.markdown(
    """
    <div style="height:50vh;"></div>
    """,
    unsafe_allow_html=True
)

st.markdown(
"""
    <style>
        [data-testid="stSidebar"] {
            background-color: #F6CF68; 
        }
        [data-testid="stSidebar"] * {
            color: black;
        }
        .stApp {
        background-color: #626262;
    }
    
    /* Title text color */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    
    /* Uploaded file box text */
    .css-1cpxqw2, .css-1d391kg {
        color: white !important;
    }
    
    div.stAlert p {
        color: #ffffff !important;
        font-weight: 200;
    }
    </style>
""",
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Please upload your excel data file here",
    type=["xlsx", "xls"],
    accept_multiple_files=False
)

predict_button = st.button("Predict", icon="📈")

# Page logic
st2 = st.empty()

if predict_button:
    if uploaded_file is None:
        st2.error("❌ No file uploaded. Please upload an Excel file.")
    else:
        # Phase 1: data verification
        st2.success("Checking data...")

        ## One sheet only policy
        if not len(pd.ExcelFile(uploaded_file).sheet_names) == 1:
            st2.empty()
            st.error("❌ Please upload excel file with only one sheet")
            st.stop()

        ## List of col and data basic clean
        req_col = col_list(metric)
        df = translation(pd.read_excel(uploaded_file))

        ## Case when data input is cleaned
        if all(col in df.columns for col in req_col):
            df = df[req_col]

        ## Case when users input original data scraped from FiinProX
        else:
            try:
                df = col_trans(uploaded_file)
                df = df[req_col]

        ## Case other than those cases (irrelevant or wrong formatted data)
            except Exception:
                st2.empty()
                st.error(f"❌ Please make sure your data start at A1, or else, not intervened since download from FiinProX."
                         f"\nPlease check your data columns if they match required col: {req_col}. "
                         f"\nFor equivalent Vietnamese name of the columns, check out /data/var_definition.txt")
                st.stop()

        ## check for null
        if df.shape[0] <= 0 or df.isna().sum().sum()>0:
            st2.empty()
            st.error("❌ Columns are matched but input data contains missing values, "
                     "please fill the data or drop the entire row")
            st.stop()

        st2.empty()


        # Phase 2: pred making using models
        st2.success("Predicting...")
        ## Set index for company, year -> keep the id while only inputing features into model
        df = df.set_index(['company', 'year'])

        ## Predict revenue, ebitda, value_add
        if metric in ['EBITDA', 'Value_add', 'Revenue']:
            with open(f'src/app/model/{metric.lower()}.pkl', 'rb') as f:
                model = pk.load(f)
            ### Main
            output = model.predict(df)
            if metric == 'Revenue':
                output = np.exp(output)
        else:
            with open(f'src/app/model/{metric.lower()}/{metric.lower()}_prepro.pkl', 'rb') as f:
                prepro = pk.load(f)
            ### Preprocess input data
            df2 = prepro.transform(df)
            df2 = df2.reshape(df2.shape[0], 1, df2.shape[1])
            ### Main
            model = tf.keras.models.load_model(f'src/app/model/{metric.lower()}/{metric.lower()}_model.keras')
            output = model.predict(df2)


        st2.empty()

        # Phase 3: return result
        out_data = pd.DataFrame(output, columns=[metric])
        out_data[['company', 'year']] = df.reset_index()[['company', 'year']]
        out_data[metric] = out_data[metric].apply(
            lambda x: f"{x:,.3f}"
        )

        st2.success("Loading result ...")
        st.dataframe(out_data[['company', 'year', metric]])
        st2.empty()