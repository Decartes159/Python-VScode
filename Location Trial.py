import streamlit as st
import geocoder 

def get_ip():
    
    ip = st.context.ip_address
    return ip
st.title("IP")
st.markdown(get_ip())