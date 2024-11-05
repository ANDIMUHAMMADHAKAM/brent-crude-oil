import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import math
def rnn():
    st.title("RNN")
    #st.subheader("Data yang digunakan")
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    #selected_columns = ['Close']
    #st.dataframe(data[selected_columns])
    #Split data into training and testing sets
    train_data = data.iloc[:int(len(data)*0.8)]
    test_data = data.iloc[int(len(data)*0.8):]

    # Feature Scaling
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

    # Prepare training data
    timesteps = 7
    X_train, y_train = [], []

    for i in range(timesteps, len(train_scaled)):
        X_train.append(train_scaled[i-timesteps:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.20))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=40, batch_size=32)

    # Prepare testing data
    combine = pd.concat((train_data['Close'], test_data['Close']), axis=0)
    test_inputs = combine[len(combine) - len(test_data) - timesteps:].values.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)
    X_test = []

    for i in range(timesteps, len(test_data) + timesteps):
        X_test.append(test_inputs[i - timesteps:i, 0])

    X_test = np.array(X_test)

    # Ensure X_test is a 3D array
    if len(X_test.shape) == 2:
        X_test = np.expand_dims(X_test, axis=2)

    # Make predictions
    prediksi_harga_close = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2])))
    prediksi_harga_close = scaler.inverse_transform(prediksi_harga_close.reshape(-1, 1))
    st.subheader("Hasil Prediksi")
    st.dataframe(prediksi_harga_close)
    # Visualize predictions
    st.subheader("Plot")
    plt.figure(figsize=(16, 8))
    plt.plot(data.index[-600:], data['Close'].tail(600), color='green', label='Data Aktual')
    plt.plot(test_data.index, test_data['Close'], color='red', label='Data Test')
    plt.plot(test_data.index, prediksi_harga_close, color='blue', label='Prediksi')
    plt.title('Prediksi harga minyak')
    plt.xlabel('Date')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Display performance metrics
    r2=r2_score(test_data['Close'],prediksi_harga_close)
    mse = mean_squared_error(test_data['Close'], prediksi_harga_close)
    mae = mean_absolute_error(test_data['Close'], prediksi_harga_close)
    rmse = sqrt(mean_squared_error(test_data['Close'], prediksi_harga_close))

    st.subheader("Metrik Evaluasi")
    st.write('R-Square: ' + str(r2))
    st.write('MSE: ' + str(mse))
    st.write('MAE: ' + str(mae))
    st.write('RMSE: ' + str(rmse))