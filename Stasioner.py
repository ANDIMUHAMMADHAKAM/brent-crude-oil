from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

def statsioner():
    st.title("Uji stasioner")
    st.write("Data yang digunakan pada tahap analisis time series yaitu kolom close. Alasan menggunakan kolom close adalah sesuai dengan definisinya, Close adalah harga penutupan yaitu harga aset pada akhir sesi perdagangan")
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
    selected_columns = ['Close', 'chg(close)']
    st.dataframe(data[selected_columns])
    st.write("")

    X = data['Close']
    # Melakukan ADF Test
    result = adfuller(X)
    adf_statistic, p_value, critical_values = result[0], result[1], result[4]

    # Menampilkan hasil di Streamlit
    st.write(f'ADF Statistic: {adf_statistic}')
    st.write(f'p-value: {p_value}')
    st.write('Critical Values:')
    for key, value in critical_values.items():
        st.write(f'{key}: {value}')

    st.write("")
    st.write("Karena uji diatas menunjukkan data tidak stasioner, terlebih dahulu melakukan differencing terhadap data yang telah ada pada kolom chg(close)")
    st.write("Berikut uji ADF menggunakan variabel chg(close)")

    X1=data['chg(low)']
    result = adfuller(X1)
    adf_statistic, p_value, critical_values = result[0], result[1], result[4]
    # Menampilkan hasil di Streamlit
    st.write(f'ADF Statistic: {adf_statistic}')
    st.write(f'p-value: {p_value}')
    st.write('Critical Values:')
    for key, value in critical_values.items():
        st.write(f'{key}: {value}')
    st.subheader("Plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    data['chg(close)'].plot(kind='line', title='chg(close)', ax=ax)
    ax.spines[['top', 'right']].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(fig)