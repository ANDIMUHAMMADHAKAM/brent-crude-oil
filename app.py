import streamlit as st
from ARIMA import arima
from DES import EXPONENTIAL
from Explorasi import Eksplorasi
from RNN import rnn
from Stasioner import statsioner
from Datate import datate
st.set_page_config(page_title="Harga Minyak Mentah")


page = st.sidebar.selectbox("Menu", ("Eksplorasi","Uji Stasioner","Data Train dan Data Test","ARIMA", "EXPONENTIAL SMOOTHING","RNN"))
if page == "ARIMA":
    arima()
elif page=="EXPONENTIAL SMOOTHING":
    EXPONENTIAL()
elif page=="Eksplorasi":
    Eksplorasi()
elif page=="Data Train dan Data Test":
    datate()
elif page=="Uji Stasioner":
    statsioner()
else :
    rnn()



