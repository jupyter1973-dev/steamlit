import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('titanic_rf_model.joblib')

# Judul
st.title("🎈 Prediksi Keselamatan Penumpang Titanic")

# Form input
st.sidebar.header("Input Fitur Penumpang")
pclass = st.sidebar.selectbox("Kelas Tiket (Pclass)", [1,2,3])
sex    = st.sidebar.selectbox("Jenis Kelamin (Sex)", ["female","male"])
age    = st.sidebar.slider("Usia (Age)", 0.0, 100.0, 30.0)
fare   = st.sidebar.number_input("Harga Tiket (Fare)", min_value=0.0, value=32.0)
emb    = st.sidebar.selectbox("Pelabuhan (Embarked)", ["C","Q","S"])

# Siapkan DataFrame input
input_dict = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "Fare": fare,
    "Embarked": emb
}
df_input = pd.DataFrame([input_dict])

# Preprocessing sama seperti training
df_input['Sex'] = df_input['Sex'].map({'female':0,'male':1})
df_input = pd.get_dummies(df_input, columns=['Embarked'], drop_first=True)
for col in ['Embarked_Q','Embarked_S']:
    if col not in df_input:
        df_input[col] = 0

# Prediksi
pred = model.predict(df_input)[0]
prob = model.predict_proba(df_input)[0][pred]

# Tampilkan hasil
st.subheader("Hasil Prediksi")
status = "🟢 Selamat" if pred==1 else "🔴 Tidak Selamat"
st.write(f"**{status}** dengan probabilitas {prob:.2%}")

# Feature importance
st.subheader("Feature Importance")
importances = model.feature_importances_
feat_names = df_input.columns
imp_df = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

st.bar_chart(data=imp_df.set_index('feature'))
