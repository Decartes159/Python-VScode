import streamlit as st

def get_ip():
    
    ip = st.context.ip_address
    return ip
st.title("IP")
st.markdown(get_ip())