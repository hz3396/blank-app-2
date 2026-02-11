import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Science Project",page_icon="📊",layout="wide")

st.sidebar.title("Diabetes Data Analysis 🩺")

page = st.sidebar.selectbox("Select the Page",["01 Introduction", "02 Data Viz"])

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")
df = load_data()

if page == "01 Introduction":
    st.title("Diabetes Dataset 🩺")
    rows = st.slider("Select number of rows to display",min_value=10,max_value=len(df),value=50,step=10)

    st.dataframe(df.head(rows), use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing, use_container_width=True)
    st.subheader("Outcome Distribution")
    fig, ax = plt.subplots()
    df["Outcome"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif page == "02 Data Viz":
    st.title("Data Visualization 📊")
    tab1, tab2, tab3 = st.tabs(["Line Chart","Bar Chart","Correlation Heatmap"])
    with tab1:
        st.subheader("Line Chart")
        x_var = st.selectbox("Select X axis variable",df.columns,index=0)
        y_var = st.selectbox("Select Y axis variable",df.columns,index=1)
        chart_df = df[[x_var, y_var]].sort_values(by=x_var)
        st.line_chart(chart_df.set_index(x_var))
    with tab2:
        st.subheader("Bar Chart")
        x_var2 = st.selectbox("Select X axis variable",df.columns,index=0,key="bar_x")
        y_var2 = st.selectbox("Select Y axis variable",df.columns,index=1,key="bar_y")
        bar_df = df.groupby(x_var2)[y_var2].mean().reset_index()
        st.bar_chart(bar_df.set_index(x_var2))
    with tab3:
        st.subheader("Correlation Heatmap")
        df_numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(df_numeric.corr(),annot=True,cmap="coolwarm",ax=ax)
        st.pyplot(fig)
