import streamlit as st

def get_ip():
    
    ip = st.context.ip_address
    st.write(ip)
st.title("IP")
get_ip()