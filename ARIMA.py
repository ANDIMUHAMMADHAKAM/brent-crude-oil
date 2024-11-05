import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from math import sqrt


def arima():
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    st.title("ARIMA BOX-JENKINS")
    st.write("Salah satu syarat ARIMA BOX-JENKINS adalah datanya harus bersifat stasioner. Maka dari itu digunakan data chg(close). Maka dari itu terlebih dahulu untuk melihat plot ACF dan PACF untuk melihat apakah terdapat nilai AR dan MA.")

    # Membuat plot ACF dan PACF
    fig_acf_pacf = plt.figure(figsize=(12, 8))
    ax1_acf = fig_acf_pacf.add_subplot(211)
    plot_acf(data['chg(close)'].dropna(), lags=40, ax=ax1_acf)

    ax2_pacf = fig_acf_pacf.add_subplot(212)
    plot_pacf(data['chg(close)'].dropna(), lags=40, ax=ax2_pacf)

    # Menampilkan plot di Streamlit
    st.pyplot(fig_acf_pacf)
    st.write("")
    st.write("Terlihat pada grafik bahwa AR dan MA masing-masing bernilai 1 dengan menggunakan kolom hasil differencing (chg(close)) sehingga didapatkan model ARIMA (1,1,1).")
    st.write("")
    st.write("Sebelum memasuki analisa ARIMA(1,1,1), terlebih dahulu untuk membagi dataset menjadi data train dan data test (dapat dilihat pada menu Data Train dan Data Test).")
    # Pisahkan data menjadi train dan test
    train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

    # Persiapkan data ARIMA
    train_arima = train_data['Close']
    test_arima = test_data['Close']

    history = [x for x in train_arima]
    y = test_arima
    predictions = list()

    # ARIMA model
    order = (1, 1, 1)

    # Melakukan prediksi dan menyimpan hasilnya
    for i in range(len(y)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        obs = y[i]
        history.append(obs)

    # Menghitung metrik evaluasi
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = sqrt(mse)
    st.title("Evaluasi Model ARIMA (1,1,1)")

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write("", predictions)
    #st.write("Actual Values:", y)

    # Menampilkan metrik evaluasi
    st.subheader("Metrik Evaluasi")
    st.write(f'R-Square: {r2}')
    st.write(f'MSE: {mse}')
    st.write(f'MAE: {mae}')
    st.write(f'RMSE: {rmse}')

    # Plot hasil prediksi di Matplotlib
    st.subheader("Plot ARIMA (1,1,1)")
    fig_prediction = plt.figure(figsize=(16, 8))
    plt.plot(data.index[-600:], data['Close'].tail(600), color='green', label='Data Aktual')
    plt.plot(test_data.index, y, color='red', label='Data Test')
    plt.plot(test_data.index[-len(predictions):], predictions, color='blue', label='Prediksi')
    plt.title('Prediksi harga minyak')
    plt.xlabel('Date')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)

    # Menampilkan plot di Streamlit
    st.pyplot(fig_prediction)