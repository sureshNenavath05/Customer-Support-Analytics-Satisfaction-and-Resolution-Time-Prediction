# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

from utils.preprocessing import process_input_data

# ========== LOAD MODELS & VECTORIZERS ========== #
st.set_page_config(page_title="Customer Support Predictions", layout="wide")

satisfaction_model = joblib.load("model/satisfaction_model.pkl")
satisfaction_features = joblib.load("model/feature_names.pkl")
resolution_model = joblib.load("model/linear_regression_model.pkl")
resolution_features = joblib.load("model/model_features.pkl")

desc_vectorizer = joblib.load("model/desc_vectorizer.pkl")
res_vectorizer = joblib.load("model/res_vectorizer.pkl")

# Define required columns
REQUIRED_COLUMNS = {
    "satisfaction": [
        'Customer Age','Customer Gender','Product Purchased',
        'Date of Purchase','Ticket Type','Ticket Subject',
        'Ticket Description','Resolution','Ticket Priority',
        'Ticket Channel','First Response Time','Time to Resolution'
    ],
    
    "resolution": [
        'Customer Age','Customer Gender','Product Purchased',
        'Date of Purchase','Ticket Type','Ticket Subject',
        'Ticket Description','Resolution','Ticket Priority',
        'Ticket Channel','First Response Time'
    ]
}

# Sample row
SAMPLE_ROWS = {
    "Customer Age": 35,
    "Customer Gender": "Male",
    "Product Purchased": "Product A",
    "Date of Purchase": "2023-05-15",
    "Ticket Type": "Technical",
    "Ticket Subject": "Device not starting",
    "Ticket Description": "The product stopped working after two days.",
    "Resolution": "Replaced with a new device.",
    "Ticket Priority": "High",
    "Ticket Channel": "Email",
    "First Response Time": "2023-05-15 10:30:00",
    "Time to Resolution": "2023-05-17 15:00:00"
}

# Required columns for uploaded prediction results
REQUIRED_COLUMNS_PREDICTIONS = {
    "satisfaction": [
        'Customer Age','Customer Gender','Product Purchased',
        'Date of Purchase','Ticket Type','Ticket Subject',
        'Ticket Description','Resolution','Ticket Priority',
        'Ticket Channel','First Response Time','Time to Resolution',
        'Customer Satisfaction Rating'
    ],
    
    "resolution": [
        'Customer Age','Customer Gender','Product Purchased',
        'Date of Purchase','Ticket Type','Ticket Subject',
        'Ticket Description','Resolution','Ticket Priority',
        'Ticket Channel','First Response Time','Predicted_Resolution_Hours'
    ]
}

# Sample row for visualizing prediction files
SAMPLE_ROWS_PREDICTIONS = {
    "Customer Age": 35,
    "Customer Gender": "Male",
    "Product Purchased": "Product A",
    "Date of Purchase": "2023-05-15",
    "Ticket Type": "Technical",
    "Ticket Subject": "Device not starting",
    "Ticket Description": "The product stopped working after two days.",
    "Resolution": "Replaced with a new device.",
    "Ticket Priority": "High",
    "Ticket Channel": "Email",
    "First Response Time": "2023-05-15 10:30:00",
    "Time to Resolution": "2023-05-17 15:00:00",
    
    # For Satisfaction Visualization
    "Customer Satisfaction Rating": 1,
    "Satisfaction_Prob": 0.87,
    
    # For Resolution Visualization
    "Predicted_Resolution_Hours": 48.5
}

# ========== PAGE HEADER ========== #
st.markdown("""
<style>
.big-font {
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.title("📊 Customer Support Ticket Prediction")
st.markdown('<div class="big-font">Welcome! Upload your CSV file to begin.</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🧮 Satisfaction Prediction",
    "⏱️ Resolution Time Prediction",
    "📈 Satisfaction Visuals",
    "📊 Resolution Visuals"
])

# ========== TAB 0: SATISFACTION PREDICTION ========== #
with tabs[0]:
    st.subheader("🧾 Required Format: Satisfaction Prediction")
    sample_df_satis = pd.DataFrame([SAMPLE_ROWS])[REQUIRED_COLUMNS["satisfaction"]]
    st.dataframe(sample_df_satis)

    sample_csv_satis = sample_df_satis.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample Satisfaction CSV", data=sample_csv_satis,
                       file_name="sample_satisfaction.csv", mime="text/csv")

    file1 = st.file_uploader("📁 Upload CSV for Satisfaction Prediction", type=["csv"], key="satisfaction")
    if file1:
        try:
            df = pd.read_csv(file1, encoding='ISO-8859-1')
            st.success(f"✅ Loaded file with {len(df)} rows.")

            missing_cols = [col for col in REQUIRED_COLUMNS["satisfaction"] if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS["satisfaction"]]

            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            if extra_cols:
                st.info(f"ℹ️ Extra columns (will be ignored): {', '.join(extra_cols)}")

            df = df[[col for col in df.columns if col in REQUIRED_COLUMNS["satisfaction"]]]

            X, df_out = process_input_data(
                df, task="satisfaction",
                model_cols=satisfaction_features,
                desc_vectorizer=desc_vectorizer,
                res_vectorizer=res_vectorizer
            )

            df_out['Customer Satisfaction Rating'] = satisfaction_model.predict(X)
            df_out['Satisfaction_Prob'] = satisfaction_model.predict_proba(X)[:, 1]

            st.markdown("### 🔍 Prediction Output")
            st.dataframe(df_out.head())
            st.download_button("📥 Download Results", df_out.to_csv(index=False), file_name="satisfaction_results.csv")

            # 📊 Visual Insights
            st.markdown("### 📊 Visual Insights")

            with st.expander("📈 Distribution of Important Features"):
                num_features = ['Customer Age', 'ticket_length', 'sentiment_score', 'Satisfaction_Prob']
                selected = st.multiselect("📌 Choose features:", num_features, default=['Satisfaction_Prob'])
                for feature in selected:
                    if feature in df_out.columns:
                        fig2, ax2 = plt.subplots()
                        sns.histplot(df_out[feature], kde=True, ax=ax2)
                        ax2.set_title(f"Distribution of {feature}")
                        st.pyplot(fig2)

            # 🔍 Top 10 Important Features
            st.markdown("### 🧠 Top 10 Important Features Influencing Satisfaction Model")
            try:
                if hasattr(satisfaction_model, 'feature_importances_'):
                    importances = satisfaction_model.feature_importances_
                elif hasattr(satisfaction_model, 'coef_'):
                    importances = np.abs(satisfaction_model.coef_[0])
                else:
                    importances = None

                if importances is not None:
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(10)

                    fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                                     title="Top 10 Features - Satisfaction Prediction", height=500)
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Model does not support feature importance inspection.")
            except Exception as e:
                st.error(f"Could not display feature importances: {e}")

        except Exception as e:
            st.error(f"❌ Error during satisfaction prediction: {e}")

# ========== TAB 1: RESOLUTION TIME ESTIMATION ========== #
with tabs[1]:
    st.subheader("🧾 Required Format: Resolution Time Prediction")
    sample_df_resolution = pd.DataFrame([SAMPLE_ROWS])[REQUIRED_COLUMNS["resolution"]]
    st.dataframe(sample_df_resolution)

    sample_csv_res = sample_df_resolution.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample Resolution CSV", data=sample_csv_res,
                       file_name="sample_resolution.csv", mime="text/csv")

    file2 = st.file_uploader("📁 Upload CSV for Resolution Time Estimation", type=["csv"], key="resolution")

    if file2:
        try:
            df = pd.read_csv(file2, encoding='ISO-8859-1')
            st.success(f"✅ Loaded file with {len(df)} rows.")

            missing_cols = [col for col in REQUIRED_COLUMNS["resolution"] if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS["resolution"]]

            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            if extra_cols:
                st.info(f"ℹ️ Extra columns (will be ignored): {', '.join(extra_cols)}")

            df = df[[col for col in df.columns if col in REQUIRED_COLUMNS["resolution"]]]

            X, df_out = process_input_data(
                df, task="resolution",
                model_cols=resolution_features,
                desc_vectorizer=desc_vectorizer,
                res_vectorizer=res_vectorizer
            )

            df_out['Predicted_Resolution_Hours'] = resolution_model.predict(X)

            st.markdown("### 🔍 Prediction Output")
            st.dataframe(df_out.head())
            st.download_button("📥 Download Predictions", df_out.to_csv(index=False), file_name="resolution_results.csv")

            # 📊 Visual Insights
            st.markdown("### 📊 Visual Insights")
                        
            with st.expander("📈 Distribution of Important Features"):
                num_features = ['Customer Age', 'ticket_length', 'sentiment_score', 'Predicted_Resolution_Hours']
                selected_res = st.multiselect("📌 Choose features:", num_features, default=['Predicted_Resolution_Hours'])
                for feature in selected_res:
                    if feature in df_out.columns:
                        fig4, ax4 = plt.subplots()
                        sns.histplot(df_out[feature], kde=True, ax=ax4)
                        ax4.set_title(f"Distribution of {feature}")
                        st.pyplot(fig4)

            # 🔍 Top 10 Important Features
            st.markdown("### 🧠 Top 10 Important Features Influencing Resolution Time")
            try:
                if hasattr(resolution_model, 'feature_importances_'):
                    importances = resolution_model.feature_importances_
                elif hasattr(resolution_model, 'coef_'):
                    importances = np.abs(resolution_model.coef_)
                else:
                    importances = None

                if importances is not None:
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(10)

                    fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                                     title="Top 10 Features - Resolution Time", height=500)
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Model does not support feature importance inspection.")
            except Exception as e:
                st.error(f"Could not display feature importances: {e}")

        except Exception as e:
            st.error(f"❌ Error during resolution time prediction: {e}")

# ========== TAB 2: SATISFACTION RESULTS VISUALIZATION ========== #
with tabs[2]:
    st.subheader("📊 Satisfaction Data Visualizations")
    st.markdown("Please upload the prediction CSV downloaded from the Satisfaction Prediction tab.")

    # Show required structure
    sample_df_tab2 = pd.DataFrame([SAMPLE_ROWS_PREDICTIONS])[REQUIRED_COLUMNS_PREDICTIONS["satisfaction"]]
    st.dataframe(sample_df_tab2)

    file_viz1 = st.file_uploader("📁 Upload Satisfaction Prediction CSV", type=["csv"], key="viz_satisfaction")

    if file_viz1:
        try:
            df_viz1 = pd.read_csv(file_viz1)
            st.success(f"✅ Loaded {len(df_viz1)} rows.")

            # Predefined visualizations
            st.markdown("### 📊 Predefined Visualizations")
            vis_pairs = [
                ('Ticket Priority', 'Customer Satisfaction Rating'),
                ('Customer Gender', 'Customer Satisfaction Rating'),
                ('Product Purchased', 'Customer Satisfaction Rating'),
                ('Customer Age', 'Customer Satisfaction Rating'),
                ('Ticket Channel', 'Customer Satisfaction Rating'),
                ('Ticket Type', 'Customer Satisfaction Rating')
            ]

            for i in range(0, len(vis_pairs), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(vis_pairs):
                        x, y = vis_pairs[i + j]
                        if x in df_viz1.columns and y in df_viz1.columns:
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                if df_viz1[x].nunique() <= 10:
                                    sns.boxplot(data=df_viz1, x=x, y=y, ax=ax)
                                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                                else:
                                    sns.histplot(data=df_viz1, x=y, kde=True, ax=ax)
                                ax.set_title(f"{y} by {x}")
                                st.pyplot(fig)

            # Custom visualization
            st.markdown("### 🎨 Custom Visualization")
            num_cols = df_viz1.select_dtypes(include=['float', 'int']).columns.tolist()
            cat_cols = df_viz1.select_dtypes(include='object').columns.tolist()

            plot_type = st.selectbox("📊 Choose plot type:", ['Histogram', 'Boxplot', 'Bar', 'Pie'], key="satis_plot")
            x_axis = st.selectbox("X-Axis", options=cat_cols + num_cols, key="satis_x")
            y_axis = st.selectbox("Y-Axis (optional)", options=[None] + num_cols, key="satis_y") if plot_type in ['Boxplot', 'Bar'] else None

            if st.button("Generate Chart", key="satis_custom_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                if plot_type == "Histogram":
                    sns.histplot(df_viz1[x_axis], kde=True, ax=ax)
                elif plot_type == "Boxplot" and y_axis:
                    sns.boxplot(x=df_viz1[x_axis], y=df_viz1[y_axis], ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                elif plot_type == "Bar" and y_axis:
                    sns.barplot(x=df_viz1[x_axis], y=df_viz1[y_axis], ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                elif plot_type == "Pie":
                    pie_data = df_viz1[x_axis].value_counts()
                    ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
                    ax.axis('equal')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error loading satisfaction data: {e}")

with tabs[3]:
    st.subheader("📊 Resolution Time Visualizations")
    st.markdown("Please upload the prediction CSV downloaded from the Resolution Time tab.")

    # Show required structure
    sample_df_tab3 = pd.DataFrame([SAMPLE_ROWS_PREDICTIONS])[REQUIRED_COLUMNS_PREDICTIONS["resolution"]]
    st.dataframe(sample_df_tab3)

    file_viz2 = st.file_uploader("📁 Upload Resolution Prediction CSV", type=["csv"], key="viz_resolution")

    if file_viz2:
        try:
            df_viz2 = pd.read_csv(file_viz2)
            st.success(f"✅ Loaded {len(df_viz2)} rows.")

            # Predefined visualizations
            st.markdown("### 📊 Predefined Visualizations")
            vis_pairs = [
                ('Ticket Priority', 'Predicted_Resolution_Hours'),
                ('Customer Gender', 'Predicted_Resolution_Hours'),
                ('Product Purchased', 'Predicted_Resolution_Hours'),
                ('Customer Age', 'Predicted_Resolution_Hours'),
                ('Ticket Channel', 'Predicted_Resolution_Hours'),
                ('Ticket Type', 'Predicted_Resolution_Hours')
            ]

            for i in range(0, len(vis_pairs), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(vis_pairs):
                        x, y = vis_pairs[i + j]
                        if x in df_viz2.columns and y in df_viz2.columns:
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.boxplot(data=df_viz2, x=x, y=y, ax=ax)
                                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                                ax.set_title(f"{y} by {x}")
                                st.pyplot(fig)

            # Custom plot
            st.markdown("### 🎨 Custom Visualization")
            num_cols = df_viz2.select_dtypes(include=['float', 'int']).columns.tolist()
            cat_cols = df_viz2.select_dtypes(include='object').columns.tolist()

            plot_type = st.selectbox("📊 Choose plot type:", ['Histogram', 'Boxplot', 'Bar', 'Pie'], key="res_plot")
            x_axis = st.selectbox("X-Axis", options=cat_cols + num_cols, key="res_x")
            y_axis = st.selectbox("Y-Axis (optional)", options=[None] + num_cols, key="res_y") if plot_type in ['Boxplot', 'Bar'] else None

            if st.button("Generate Chart", key="res_custom_plot"):
                fig, ax = plt.subplots(figsize=(6, 4))
                if plot_type == "Histogram":
                    sns.histplot(df_viz2[x_axis], kde=True, ax=ax)
                elif plot_type == "Boxplot" and y_axis:
                    sns.boxplot(x=df_viz2[x_axis], y=df_viz2[y_axis], ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                elif plot_type == "Bar" and y_axis:
                    sns.barplot(x=df_viz2[x_axis], y=df_viz2[y_axis], ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                elif plot_type == "Pie":
                    pie_data = df_viz2[x_axis].value_counts()
                    ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
                    ax.axis('equal')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error loading resolution data: {e}")
