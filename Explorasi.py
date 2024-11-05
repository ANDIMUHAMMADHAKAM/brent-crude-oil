import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt

def Eksplorasi():
    st.title("Eksplorasi data")
    # Simpan DataFrame ke dalam variabel data (gantilah ini sesuai dengan kebutuhan Anda)
    # Define a custom date parsing function
    dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
    data = pd.read_excel('Copy of brentcrudeoil (1).xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse)

    # Menampilkan keseluruhan data
    st.subheader("Keseluruhan Data")
    st.dataframe(data)

    # Menampilkan plot menggunakan Matplotlib
    st.subheader("Plot Data")
    selected_data = st.selectbox("Pilih Data", ["Low", "Close", "High"])

    # Membuat plot dengan Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data[selected_data])
    ax.set_title(selected_data)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    # Menghilangkan spines atas dan kanan
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

    st.write("Ketiga plot menghasilkan pola trend dan bersifat non stasioner. Untuk pembuktiannya, dapat menggunakan uji ADF")
