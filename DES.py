from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from math import sqrt
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt

def EXPONENTIAL():
    st.title("Exponential Smoothing")
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    #selected_columns = ['Close']
    #st.dataframe(data[selected_columns])

    # Pisahkan data menjadi train dan test
    # Split data into training and testing sets
    train_df, test_df = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]
    train_des = train_df['Close']
    test_des = test_df['Close']

    history = [x for x in train_des]
    y = test_des
    prediksi = list()

    DES = Holt(history)
    Model_fit_DES = DES.fit()
    yhat_DES = Model_fit_DES.forecast(1)[0]
    prediksi.append(yhat_DES)
    history.append(y[0])
    # Make predictions
    for i in range(1, len(y)):
        # predict
        DES = Holt(history)
        Model_fit_DES = DES.fit()
        yhat_DES = Model_fit_DES.forecast(1)[0]

        # invert transformed prediction
        prediksi.append(yhat_DES)

        # observation
        obs = y[i]
        history.append(obs)
    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write("", prediksi)
    #st.write("Actual Values:", y)

    # Report performance
    st.subheader("Metrik Evaluasi")
    r2=r2_score(y,prediksi)
    st.write('R-Squared: '+str(r2))
    mse = mean_squared_error(y, prediksi)
    st.write('MSE: '+str(mse))
    mae = mean_absolute_error(y, prediksi)
    st.write('MAE: '+str(mae))
    rmse = sqrt(mean_squared_error(y, prediksi))
    st.write('RMSE: '+str(rmse))
    # Streamlit App
    st.subheader("Plot")

    # Plot hasil prediksi di Matplotlib
    fig_prediction_des = plt.figure(figsize=(16, 8))
    plt.plot(data.index[-600:], data['Close'].tail(600), color='green', label='Data Aktual')
    plt.plot(test_df.index, y, color='red', label='Data Test')
    plt.plot(test_df.index, prediksi, color='blue', label='Prediksi DES')
    plt.title('Prediksi harga minyak')
    plt.xlabel('Date')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)

    # Menampilkan plot di Streamlit
    st.pyplot(fig_prediction_des)