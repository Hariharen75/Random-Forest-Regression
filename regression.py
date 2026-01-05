import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Random Forest Regression", layout="wide")
st.title("💰 Salary Prediction using Random Forest Regression")

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Salary Regression CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------------------------------
# Validate Columns
# -------------------------------------------------
required_cols = {"age", "experience_years", "education_level", "salary"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain: age, experience_years, education_level, salary")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# Preview
# -------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Visualization (FIXED)
# -------------------------------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        df,
        x="experience_years",
        y="salary",
        title="Experience vs Salary"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(
        df,
        x="education_level",
        y="salary",
        title="Salary by Education Level"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# Encode Categorical Column
# -------------------------------------------------
le = LabelEncoder()
df["education_level"] = le.fit_transform(df["education_level"])

# -------------------------------------------------
# Features & Target
# -------------------------------------------------
X = df[["age", "experience_years", "education_level"]]
y = df["salary"]

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

st.success(f"✅ R² Score: {r2_score(y_test, y_pred):.2f}")
st.info(f"📉 Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):,.0f}")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🔮 Predict Salary")

age = st.number_input("Age", 18, 65, 30)
experience = st.number_input("Experience (Years)", 0, 45, 5)
education = st.selectbox("Education Level", le.classes_)

if st.button("Predict Salary"):
    edu_encoded = le.transform([education])[0]
    prediction = model.predict([[age, experience, edu_encoded]])[0]
    st.success(f"💵 Predicted Salary: ₹{prediction:,.0f}")
