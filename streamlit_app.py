import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm import LiteLLM
from matplotlib.figure import Figure

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("EDA using Natural Language")

try:
    ai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    ai_api_key  = st.text_input("Enter OpenAI api key: ", type='password')
    if not ai_api_key:
        st.info("You must add your API key to continue!")
        st.stop()

csv_data = st.sidebar.file_uploader("Upload CSV file", type='csv')

if csv_data:
    df = pd.read_csv(csv_data)
    st.write("### Preview of data", df.head(3))

    if "user_query_widget" not in st.session_state:
        st.session_state.user_query_widget = ""

    def apply_template():
        if st.session_state.template_selector != "Select a template...":
            st.session_state.user_query_widget = st.session_state.template_selector
            st.session_state.template_selector = "Select a template..."

    user_query = st.text_input(
        "Query your data in natural langauge",
        key = "user_query_widget"
    )

    template = st.selectbox("Quick questions", [
        "Select a template...",
        "Show the first 5 rows",
        "What is the shape of the dataset?",
        "List all column names",
        "Show summary statistics for numeric columns",
        "Count missing values in each column",
        "Find the most frequent value in column X",      # change X to a real column
        "Show correlation matrix",
        "Plot a histogram of column Y"],
        key="template_selector",
        on_change=apply_template
    )


    if st.button("Analyze"):
        if not user_query.strip():
            st.warning("Please enter a query!")
        else:
            with st.spinner("Processing your request"):
                try:
                    llm = LiteLLM(
                        model = 'gpt-4o-mini',
                        temperature = 0
                    )

                    pai.config.set({"llm" : llm})

                    smart_df = pai.SmartDataframe(df)

                    answer = smart_df.chat(user_query)
                    answer_str = str(answer)

                    st.success("Result: ")
                    
                    if isinstance(answer, pd.DataFrame):
                        st.dataframe(answer)
                    elif isinstance(answer, Figure):
                        st.pyplot(answer)
                    elif answer_str.strip().endswith(('.png', '.jpg', '.jpeg')):                        st.image(answer_str.strip())   
                    else:
                        st.write(answer)  # fallback to just showing whatever answer

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.info("Check API key, or the answer was non-executable")
