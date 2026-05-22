import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ==========================================
# BASIC CONFIGURATION
# ==========================================

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

APP_TITLE = "EdTech Student Performance Prediction System"
APP_SUBTITLE = "Interactive analytics and machine learning dashboard for student performance prediction"
DEVELOPER_NAME = "Muhammad Ahsan"

RANDOM_STATE = 42

DATA_FILE = Path(__file__).resolve().parent / "student_performance.csv"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

REGRESSION_ARTIFACT = ARTIFACT_DIR / "exam_score_model.joblib"
CLASSIFICATION_ARTIFACT = ARTIFACT_DIR / "final_grade_model.joblib"

FEATURE_COLUMNS = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Motivation",
    "OnlineCourses",
    "AssignmentCompletion",
    "EduTech",
    "StressLevel",
]

REGRESSION_TARGET = "ExamScore"
CLASSIFICATION_TARGET = "FinalGrade"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

def apply_custom_styles():

    st.markdown(
        """
        <style>

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #0f766e 60%, #14b8a6 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1rem;
        }

        .footer-box {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #475569;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data(show_spinner=False)
def load_data():

    try:

        if not DATA_FILE.exists():
            st.error(f"Dataset not found: {DATA_FILE}")
            st.stop()

        df = pd.read_csv(DATA_FILE)

        # Remove duplicates
        df = df.drop_duplicates()

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()


# ==========================================
# TRAIN MODELS
# ==========================================

@st.cache_resource(show_spinner=True)
def train_models(df):

    data = df.copy()

    X = data[FEATURE_COLUMNS]

    y_reg = data[REGRESSION_TARGET]
    y_clf = data[CLASSIFICATION_TARGET]

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
    ) = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    # ==========================================
    # REGRESSION MODEL
    # ==========================================

    regression_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ==========================================
    # CLASSIFICATION MODEL
    # ==========================================

    classification_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Train models
    regression_pipeline.fit(X_train, y_reg_train)
    classification_pipeline.fit(X_train, y_clf_train)

    # ==========================================
    # REGRESSION EVALUATION
    # ==========================================

    reg_predictions = regression_pipeline.predict(X_test)

    regression_metrics = {
        "MAE": mean_absolute_error(y_reg_test, reg_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_reg_test, reg_predictions)),
        "R2 Score": r2_score(y_reg_test, reg_predictions),
    }

    # ==========================================
    # CLASSIFICATION EVALUATION
    # ==========================================

    clf_predictions = classification_pipeline.predict(X_test)

    report = classification_report(
        y_clf_test,
        clf_predictions,
        output_dict=True,
        zero_division=0,
    )

    classification_metrics = {
        "Accuracy": accuracy_score(y_clf_test, clf_predictions),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1 Score": report["weighted avg"]["f1-score"],
        "Confusion Matrix": confusion_matrix(
            y_clf_test,
            clf_predictions
        ),
    }

    # Save models
    joblib.dump(regression_pipeline, REGRESSION_ARTIFACT)
    joblib.dump(classification_pipeline, CLASSIFICATION_ARTIFACT)

    return {
        "regression_pipeline": regression_pipeline,
        "classification_pipeline": classification_pipeline,
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
    }


# ==========================================
# HEADER
# ==========================================

def render_header(df):

    st.markdown(
        f"""
        <div class="hero-card">

        <h1>{APP_TITLE}</h1>

        <p>{APP_SUBTITLE}</p>

        <p>
        <strong>Developer:</strong> {DEVELOPER_NAME}<br>
        <strong>Dataset:</strong> student_performance.csv<br>
        <strong>Records:</strong> {df.shape[0]}<br>
        <strong>Features:</strong> {df.shape[1]}
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR
# ==========================================

def render_sidebar():

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go To",
        [
            "Home",
            "Dataset Overview",
            "EDA",
            "Machine Learning",
            "Model Evaluation",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Developed by {DEVELOPER_NAME}")

    return page


# ==========================================
# HOME PAGE
# ==========================================

def render_home(df):

    st.title("Project Overview")

    st.write(
        """
        This application analyzes student learning behavior,
        academic engagement, and educational technology usage
        to predict exam scores and final grades.
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Targets", 2)

    st.markdown("### Objectives")

    st.write("- Analyze student behavior")
    st.write("- Perform exploratory data analysis")
    st.write("- Build machine learning models")
    st.write("- Predict ExamScore and FinalGrade")


# ==========================================
# DATASET OVERVIEW
# ==========================================

def render_dataset_overview(df):

    st.title("Dataset Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())


# ==========================================
# DISTRIBUTION PLOT
# ==========================================

def render_distribution_plot(df, column, title, color):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        bins=25,
        ax=ax,
        color=color,
    )

    ax.set_title(title)

    st.pyplot(fig)


# ==========================================
# EDA
# ==========================================

def render_eda(df):

    st.title("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        render_distribution_plot(
            df,
            "StudyHours",
            "Study Hours Distribution",
            "#0f766e",
        )

    with c2:
        render_distribution_plot(
            df,
            "Attendance",
            "Attendance Distribution",
            "#2563eb",
        )

    c3, c4 = st.columns(2)

    with c3:
        render_distribution_plot(
            df,
            "ExamScore",
            "Exam Score Distribution",
            "#7c3aed",
        )

    with c4:

        grade_counts = df["FinalGrade"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.barplot(
            x=grade_counts.index,
            y=grade_counts.values,
            ax=ax,
        )

        ax.set_title("Final Grade Distribution")
        ax.set_xlabel("Final Grade")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap")

    st.pyplot(fig)


# ==========================================
# USER INPUT FORM
# ==========================================

def build_input_form(df, form_key):

    defaults = {
        column: int(df[column].median())
        for column in FEATURE_COLUMNS
    }

    ranges = {
        column: (
            int(df[column].min()),
            int(df[column].max())
        )
        for column in FEATURE_COLUMNS
    }

    with st.form(form_key):

        st.subheader("Enter Student Information")

        values = {}

        c1, c2 = st.columns(2)

        for index, column in enumerate(FEATURE_COLUMNS):

            target = c1 if index % 2 == 0 else c2

            with target:

                min_val, max_val = ranges[column]

                values[column] = st.number_input(
                    column,
                    min_value=min_val,
                    max_value=max_val,
                    value=defaults[column],
                    step=1,
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        return pd.DataFrame([values])

    return pd.DataFrame()


# ==========================================
# MACHINE LEARNING PAGE
# ==========================================

def render_machine_learning(df, model_state):

    st.title("Machine Learning Predictions")

    tab1, tab2 = st.tabs(
        [
            "Exam Score Prediction",
            "Final Grade Prediction",
        ]
    )

    # ==========================================
    # REGRESSION
    # ==========================================

    with tab1:

        user_input = build_input_form(
            df,
            "regression_form"
        )

        if not user_input.empty:

            model = model_state["regression_pipeline"]

            prediction = model.predict(user_input)[0]

            st.success("Prediction completed successfully")

            st.metric(
                "Predicted Exam Score",
                f"{prediction:.2f}"
            )

    # ==========================================
    # CLASSIFICATION
    # ==========================================

    with tab2:

        user_input = build_input_form(
            df,
            "classification_form"
        )

        if not user_input.empty:

            model = model_state["classification_pipeline"]

            prediction = model.predict(user_input)[0]

            st.success("Prediction completed successfully")

            st.metric(
                "Predicted Final Grade",
                f"Class {prediction}"
            )


# ==========================================
# MODEL EVALUATION
# ==========================================

def render_model_evaluation(model_state):

    st.title("Model Evaluation")

    st.subheader("Regression Metrics")

    reg_metrics = model_state["regression_metrics"]

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{reg_metrics['MAE']:.2f}")
    c2.metric("RMSE", f"{reg_metrics['RMSE']:.2f}")
    c3.metric("R2 Score", f"{reg_metrics['R2 Score']:.3f}")

    st.markdown("---")

    st.subheader("Classification Metrics")

    clf_metrics = model_state["classification_metrics"]

    c4, c5, c6 = st.columns(3)

    c4.metric("Accuracy", f"{clf_metrics['Accuracy']:.3f}")
    c5.metric("Precision", f"{clf_metrics['Precision']:.3f}")
    c6.metric("Recall", f"{clf_metrics['Recall']:.3f}")

    st.metric("F1 Score", f"{clf_metrics['F1 Score']:.3f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        clf_metrics["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)


# ==========================================
# FOOTER
# ==========================================

def render_footer():

    st.markdown(
        f"""
        <div class="footer-box">

        <strong>EdTech Student Performance Prediction System</strong><br>

        Developed by {DEVELOPER_NAME}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    apply_custom_styles()

    df = load_data()

    model_state = train_models(df)

    render_header(df)

    page = render_sidebar()

    st.markdown("---")

    if page == "Home":
        render_home(df)

    elif page == "Dataset Overview":
        render_dataset_overview(df)

    elif page == "EDA":
        render_eda(df)

    elif page == "Machine Learning":
        render_machine_learning(df, model_state)

    elif page == "Model Evaluation":
        render_model_evaluation(model_state)

    render_footer()


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    main()