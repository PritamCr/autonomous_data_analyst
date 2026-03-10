import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from agent import DataAgent
from utils.data_loader import load_csv

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🤖 Autonomous AI Data Analyst")

# --------------------------
# 0️⃣ Session State Initialization
# --------------------------
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
    st.session_state["trained_score"] = None
    st.session_state["model_type"] = None
if "suggested_features" not in st.session_state:
    st.session_state["suggested_features"] = []
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = []

# --------------------------
# 1️⃣ Upload CSV
# --------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = load_csv(uploaded_file)
    st.success("CSV Loaded!")
    st.dataframe(df.head())

    agent = DataAgent(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # --------------------------
    # 2️⃣ Correlation Heatmap
    # --------------------------
    st.subheader("Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap", key="heatmap_cb"):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --------------------------
    # 3️⃣ Histogram & Scatter
    # --------------------------
    st.subheader("Histogram")
    hist_feature = st.selectbox("Select feature for histogram", numeric_cols, key="hist_feature")
    if st.button("Generate Histogram", key="hist_btn"):
        fig = px.histogram(df, x=hist_feature, title=f"Histogram of {hist_feature}")
        st.plotly_chart(fig)

    st.subheader("Scatter Plot")
    x_feature = st.selectbox("Select X feature", numeric_cols, key="scatter_x")
    y_feature = st.selectbox("Select Y feature", numeric_cols, key="scatter_y")
    if st.button("Generate Scatter Plot", key="scatter_btn"):
        fig = px.scatter(df, x=x_feature, y=y_feature, title=f"{x_feature} vs {y_feature}")
        st.plotly_chart(fig)

    # --------------------------
    # 4️⃣ AI Insights / Explain Patterns
    # --------------------------
    st.subheader("AI Insights / Pattern Explanation")
    if st.button("Explain Patterns", key="insights_btn"):
        with st.spinner("LLM analyzing dataset patterns..."):
            explanation = agent.insights()
        st.session_state["pattern_explanation"] = explanation

    if "pattern_explanation" in st.session_state:
        st.write(st.session_state["pattern_explanation"])

    # --------------------------
    # 5️⃣ AI Feature Selection
    # --------------------------
    st.subheader("AI Feature Selection")

    target_col = st.selectbox(
        "Select target column",
        df.columns,
        key="target_col"
    )

    # Get all possible features except target
    all_features = [col for col in df.columns if col != target_col]

    # Button to get AI suggestions
    if st.button("Suggest Features", key="suggest_features_btn"):

        raw_features = agent.feature_suggestions(target_col)

        # Clean feature names
        cleaned = [
            f.strip().replace('"', '').replace("'", "")
            for f in raw_features
            if f.strip() != target_col
        ]

        # Remove duplicates
        unique = []
        for f in cleaned:
            if f not in unique:
                unique.append(f)

        st.session_state["suggested_features"] = unique

    # Show suggested features
    if st.session_state.get("suggested_features"):
        st.info(f"🤖 AI Suggested Features: {', '.join(st.session_state['suggested_features'])}")

    # Multiselect with ALL features but default = AI suggestions
    selected_features = st.multiselect(
        "Select features for training",
        options=all_features,
        default=st.session_state.get("suggested_features", []),
        key="selected_features_multiselect"
    )

    st.session_state["selected_features"] = selected_features

    # --------------------------
    # 6️⃣ Train ML Model
    # --------------------------
    st.subheader("Train Model with Selected Features")
    if st.button("Run Model", key="train_model_btn"):
        if selected_features:
            train_df = df[selected_features + [target_col]]
            model_type, score, model = agent.train(target_col, df=train_df, return_model=True)
            # Save in session_state
            st.session_state["trained_model"] = model
            st.session_state["trained_score"] = score
            st.session_state["model_type"] = model_type

    # Display trained model info if exists
    if st.session_state["trained_model"] is not None:
        st.success(f"Trained Model Type: {st.session_state['model_type']}")
        st.success(f"Score: {st.session_state['trained_score']:.4f}")

        # Feature importance
        fig = agent.get_feature_importance_chart(target_col, df[selected_features])
        if fig:
            st.plotly_chart(fig)
        else:
            st.info("Feature importance not available for this model type.")

    # --------------------------
    # 7️⃣ AutoML
    # --------------------------
    st.subheader("AutoML")
    if st.button("Run AutoML", key="automl_btn"):
        if selected_features:
            train_df = df[selected_features + [target_col]]
            results, best_model = agent.auto_ml(target_col, df=train_df)
            st.write("All Model Results:", results)
            st.success(f"Best Model: {best_model[0]} with Score: {best_model[1]:.4f}")
            # Save best model in session_state
            st.session_state["trained_model"] = agent.get_model(target_col)
            st.session_state["trained_score"] = best_model[1]
            st.session_state["model_type"] = "Classification" if df[target_col].nunique() < 10 else "Regression"
