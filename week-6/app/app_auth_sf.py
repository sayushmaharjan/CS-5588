import streamlit as st
import snowflake.connector
import hashlib
import random
import string
import smtplib
from dotenv import load_dotenv
import os
from email.message import EmailMessage

# Load environment variables from .env
load_dotenv()

# -------------------------
# Helper: Snowflake connection
# -------------------------
def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

# -------------------------
# Helper: Hash password
# -------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------
# Send OTP via Gmail
# -------------------------
def send_otp(email: str, otp: str):
    msg = EmailMessage()
    msg['Subject'] = 'Your WeatherTwin Verification Code'
    msg['From'] = os.getenv("SMTP_EMAIL")
    msg['To'] = email
    msg.set_content(f"Your OTP for WeatherTwin registration is: {otp}")
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
        smtp.send_message(msg)

# -------------------------
# Registration
# -------------------------
def register_user():
    st.subheader("📝 Register")
    email = st.text_input("Enter Gmail", key="reg_email")
    password = st.text_input("Enter password", type="password", key="reg_pass")
    
    if st.button("Send OTP"):
        if email and password:
            otp = ''.join(random.choices(string.digits, k=6))
            st.session_state["otp"] = otp
            st.session_state["reg_email_temp"] = email
            st.session_state["reg_pass_temp"] = password
            send_otp(email, otp)
            st.success("OTP sent to your Gmail!")
        else:
            st.warning("Please enter Gmail and password")

    otp_input = st.text_input("Enter OTP", key="otp_input")
    if st.button("Verify & Register"):
        if otp_input == st.session_state.get("otp"):
            email = st.session_state.get("reg_email_temp")
            hashed_pw = hash_password(st.session_state.get("reg_pass_temp"))
            conn = get_snowflake_conn()
            cs = conn.cursor()
            cs.execute("""
                INSERT INTO AUTH_USERS (GMAIL, HASHED_PASSWORD)
                VALUES (%s, %s)
            """, (email, hashed_pw))
            cs.close()
            conn.close()
            st.success("✅ Registration complete! You can now log in.")
            st.session_state["otp"] = None
        else:
            st.error("❌ Incorrect OTP")

# -------------------------
# Login
# -------------------------
def login_user():
    st.subheader("🔑 Login")
    email = st.text_input("Gmail", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    
    if st.button("Login"):
        hashed_pw = hash_password(password)
        conn = get_snowflake_conn()
        cs = conn.cursor()
        cs.execute(
            "SELECT COUNT(*) FROM AUTH_USERS WHERE GMAIL=%s AND HASHED_PASSWORD=%s",
            (email, hashed_pw)
        )
        result = cs.fetchone()
        cs.close()
        conn.close()
        
        if result[0] == 1:
            st.success(f"Logged in as {email}")
            st.session_state["user"] = email
        else:
            st.error("❌ Invalid Gmail or password")

# -------------------------
# Auth UI
# -------------------------
def auth_ui():
    mode = st.radio("Select mode:", ["Login", "Register"])
    if mode == "Login":
        login_user()
    else:
        register_user()
