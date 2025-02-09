import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("government_expenditures.csv")
        return data
    except FileNotFoundError:
        st.error("File government_expenditures.csv tidak ditemukan. Pastikan file berada di lokasi yang benar.")
        return pd.DataFrame()

data = load_data()

# Periksa apakah data berhasil dimuat
if not data.empty:
    # Judul aplikasi
    st.title("Dashboard Monitoring Keuangan Negara")

    # Sidebar untuk memilih kategori pengeluaran
    categories = data.columns[2:].tolist()
    selected_category = st.sidebar.selectbox("Pilih Kategori Pengeluaran", categories)

    # Visualisasi tren pengeluaran
    st.header(f"Tren Pengeluaran: {selected_category}")
    category_data = data[selected_category].dropna().astype(float)
    years = data["Unnamed: 1"].dropna().astype(int)

    # Periksa apakah data ada
    if not category_data.empty and not years.empty:
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
                # Model ARIMA
                model = ARIMA(category_data, order=(5, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=5)

                # Hasil prediksi
                st.subheader("Hasil Prediksi (5 Tahun ke Depan)")
                forecast_years = [int(years.iloc[-1]) + i + 1 for i in range(5)]
                forecast_df = pd.DataFrame({
                    "Tahun": forecast_years,
                    "Prediksi Pengeluaran (Miliar Rp)": forecast
                })
                st.table(forecast_df)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
    else:
        st.warning("Data tren pengeluaran tidak tersedia.")
else:
    st.warning("Data tidak berhasil dimuat. Periksa kembali file data.")
