# ==========================================
# 🚢 Titanic Survival AI System – Professional & Real-World Ready
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap as shap


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Titanic AI System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Titanic Image Banner (small & centered)
# -------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 15px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg" width="350">
        <h2 style='text-align: center; margin-top: 10px;'>🚢 Titanic Survival Prediction System</h2>
        <p style='text-align: center; color: gray;'>Machine Learning • Analytics • Explainable AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.info("""
👋 Welcome! This system predicts whether a passenger would survive the Titanic disaster.

👉 Steps:
1. Upload dataset
2. Explore insights in dashboard
3. Enter passenger details
4. Click Predict

💡 Tip: Try Female, 1st Class, Age 25 → high survival
""")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Titanic Dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload Titanic dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(df):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1 if df['FamilySize'].max() <= 1 else 0
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

df = preprocess(df)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")
show_data = st.sidebar.checkbox("Show Raw Data")
show_eda = st.sidebar.checkbox("Show Analysis", True)

# -------------------------------
# Show Raw Data
# -------------------------------
if show_data:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

# -------------------------------
# Feature Selection
# -------------------------------
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
X = df[features]
y = df['Survived']

# -------------------------------
# Hyperparameter Tuning for Random Forest
# -------------------------------
with st.spinner("Training models..."):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf_params = {
        'n_estimators':[100,200],
        'max_depth':[5,10,None],
        'min_samples_split':[2,5],
        'min_samples_leaf':[1,2]
    }
    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_

    # Logistic Regression for comparison
    lr = LogisticRegression(max_iter=300)
    lr.fit(X_train, y_train)

# -------------------------------
# Model Metrics
# -------------------------------
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("🌳 Random Forest Accuracy", f"{accuracy_score(y_test, rf.predict(X_test))*100:.2f}%")
col2.metric("📈 Logistic Regression Accuracy", f"{accuracy_score(y_test, lr.predict(X_test))*100:.2f}%")
col3.metric("📦 Total Passengers", len(df))

# -------------------------------
# Interactive Dashboard with Tabs
# -------------------------------
if show_eda:
    st.subheader("📊 Data Analysis & Insights")
    tab1, tab2, tab3 = st.tabs(["Survival Overview","Feature Importance","Correlation Heatmap"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df, names='Survived', title="Survival Ratio"), use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(df, x='Age', color='Survived', title="Age Distribution"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(px.histogram(df, x='Pclass', color='Survived', title="Survival by Class"), use_container_width=True)
        with col4:
            st.plotly_chart(px.histogram(df, x='Sex', color='Survived', title="Survival by Gender"), use_container_width=True)

    with tab2:
        st.subheader("🌟 Feature Importance (Random Forest)")
        importance = pd.DataFrame({'Feature':features,'Importance':rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        fig = px.bar(importance, x='Feature', y='Importance', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🟢 Correlation Heatmap")
        corr = df[features+['Survived']].corr()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# -------------------------------
# AI Insights
# -------------------------------
st.subheader("🤖 AI Insights")
overall = df['Survived'].mean()*100
female = df[df['Sex']==1]['Survived'].mean()*100
male = df[df['Sex']==0]['Survived'].mean()*100
st.info(f"""
📌 Overall Survival Rate: {overall:.2f}%
👩 Female Survival: {female:.2f}%
👨 Male Survival: {male:.2f}%

🎯 Observations:
- Females had higher survival
- Higher-class passengers survived more
- Age, Fare, and Family Size influence outcomes
""")

# -------------------------------
# Prediction Section (Card Style)
# -------------------------------
# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔮 Predict Survival")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3],
        help="1 = First Class (rich), 2 = Middle, 3 = Lower Class"
    )

    sex = st.selectbox(
        "Sex",
        ["Male","Female"],
        help="Females had higher survival rate"
    )

    age = st.slider(
        "Age",
        1, 80, 25,
        help="Children had better survival chances"
    )

with col2:
    fare = st.number_input(
        "Fare",
        0.0, 500.0, 50.0,
        help="Higher fare → better survival chances"
    )

    family = st.slider(
        "Family Size",
        1, 10, 1,
        help="Large families had lower survival"
    )

sex_val = 1 if sex=="Female" else 0
input_data = np.array([[pclass, sex_val, age, fare, family]])
    
st.markdown("### 💡 Try Example Scenarios")

col1, col2 = st.columns(2)

with col1:
    if st.button("👩 High Survival Example"):
        st.success("Female, 1st Class, Age 25 → High Survival")

with col2:
    if st.button("👨 Low Survival Example"):
        st.error("Male, 3rd Class, Age 40 → Low Survival")

if st.button("🚀 Predict"):

    with st.spinner("Analyzing passenger data..."):
        result = rf.predict(input_data)[0]
        prob = rf.predict_proba(input_data)[0][1]

    # Show result AFTER spinner
    if result == 1:
        st.success(f"✅ Survived (Probability: {prob*100:.2f}%)")
    else:
        st.error(f"❌ Not Survived (Probability: {prob*100:.2f}%)")

    
    st.subheader("📊 Prediction ")

    prob_df = pd.DataFrame({
    'Outcome': ['Not Survived', 'Survived'],
    'Probability': [1-prob, prob]
     })

    st.plotly_chart(px.bar(prob_df, x='Outcome', y='Probability', text='Probability'),
    use_container_width=True)

    result_df = pd.DataFrame({
    "Pclass":[pclass],
    "Sex":[sex],
    "Age":[age],
    "Fare":[fare],
    "FamilySize":[family],
    "Prediction":[result],
    "Probability":[prob]
})

    st.download_button(
    "📥 Download Prediction",
    result_df.to_csv(index=False),
    file_name="prediction.csv"
)

    # Show survival result
    if result == 1:
        st.success(f"✅ Survived (Probability: {prob*100:.2f}%)")
    else:
        st.error(f"❌ Not Survived (Probability: {prob*100:.2f}%)")

    # Prediction explanation
    st.subheader("🧠 Why this prediction?")
    reasons = [
        "Female → higher survival" if sex_val==1 else "Male → lower survival",
        "1st Class → higher survival" if pclass==1 else ("3rd Class → lower survival" if pclass==3 else "2nd Class → moderate survival"),
        "Child → slightly higher survival" if age<12 else "Adult → normal survival chance",
        "High fare → better survival" if fare>100 else "Moderate fare → normal chance",
        "Large family → lower survival" if family>4 else "Small family → higher chance"
    ]
    for r in reasons:
        st.write("✔", r)

   # -------------------------------
# -------------------------------
# SHAP Explainability (FINAL FIX)
# -------------------------------
st.subheader("📊 SHAP Explainability")

explainer = shap.Explainer(rf)
shap_values = explainer(input_data)

# Select ONLY class 1 (Survived)
shap_value_single = shap_values[0, :, 1]

fig, ax = plt.subplots()

shap.plots.waterfall(shap_value_single, show=False)

st.pyplot(fig)

    
# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🏆 Titanic AI | Streamlit • ML • Interactive Dashboard • Real-World Ready</p>",
    unsafe_allow_html=True
)
