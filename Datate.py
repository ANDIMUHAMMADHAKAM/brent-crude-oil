from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

def datate():
    st.title("Data Train dan Data Test")
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    # Pisahkan data menjadi train dan test
    train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]
    st.subheader("Data Train")
    Train=train_data['Close']
    st.dataframe(Train)

    st.subheader("Data Test")
    Test=test_data['Close']
    st.dataframe(Test)
    # Plot data train dan test
    st.subheader("Plot Data Train dan Test")
    fig_train_test = plt.figure(figsize=(16, 8))
    plt.xlabel('Dates')
    plt.ylabel('Close')
    plt.plot(data['Close'], 'green', label='Train data')
    plt.plot(test_data['Close'], 'blue', label='Test data')
    plt.legend()

    # Menampilkan plot di Streamlit
    st.pyplot(fig_train_test)
