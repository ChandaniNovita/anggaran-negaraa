import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("data/government_expenditures.csv")
    return data

data = load_data()

# Judul aplikasi
st.title("Dashboard Monitoring Keuangan Negara")

# Sidebar untuk memilih kategori pengeluaran
categories = data.columns[2:].tolist()
selected_category = st.sidebar.selectbox("Pilih Kategori Pengeluaran", categories)

# Visualisasi tren pengeluaran
st.header(f"Tren Pengeluaran: {selected_category}")
category_data = data[selected_category].dropna().astype(float)
years = data["Unnamed: 1"].dropna()

plt.figure(figsize=(10, 5))
plt.plot(years, category_data, marker='o')
plt.xlabel("Tahun")
plt.ylabel("Jumlah Pengeluaran (Miliar Rp)")
plt.grid(True)
st.pyplot(plt)

# Fitur prediksi menggunakan ARIMA
st.header("Prediksi Pengeluaran Masa Depan")
if st.button("Lakukan Prediksi"):
    try:
        model = ARIMA(category_data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        
        st.subheader("Hasil Prediksi (5 Tahun ke Depan)")
        forecast_years = [int(years.iloc[-1]) + i + 1 for i in range(5)]
        forecast_df = pd.DataFrame({
            "Tahun": forecast_years,
            "Prediksi Pengeluaran": forecast
        })
        st.table(forecast_df)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Informasi tambahan
st.sidebar.markdown("""
**Catatan:**
- Data berasal dari Kementerian Keuangan.
- Prediksi menggunakan model ARIMA.
""")