# app.py
import streamlit as st
import sqlite3
import tempfile
import os
from datetime import datetime
from ocr_module import analyze_image

DB_PATH = "readings.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    filepath TEXT,
                    label TEXT,
                    value REAL,
                    raw_text TEXT,
                    annotated_path TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

def save_to_db(category, filepath, label, value, raw_text, annotated_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO readings (category, filepath, label, value, raw_text, annotated_path, timestamp) VALUES (?,?,?,?,?,?,?)",
              (category, filepath, label, value, raw_text, annotated_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

st.set_page_config(page_title="Control Panel OCR", layout="centered")
st.title("📊 Control Panel OCR Application")

init_db()

category = st.selectbox("Select Input Type", ["genset", "mri", "electrical"])

st.write("Upload an image or capture using your camera (if available).")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Take a picture")

file_obj = None
if uploaded_file is not None:
    file_obj = uploaded_file
elif camera_file is not None:
    file_obj = camera_file

if file_obj is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_obj.getbuffer())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded image preview", use_column_width=True)

    if st.button("Process & Save"):
        with st.spinner("Running OCR..."):
            try:
                res = analyze_image(tmp_path, show=False)
                if res:
                    st.success(f"Detected: {res['label']} → {res['value']}")
                    st.write("Raw OCR text:", res['raw_text'])
                    # Show annotated image
                    st.image(res['annotated_path'], caption="Annotated", use_column_width=True)
                    # Save to DB
                    save_to_db(category, tmp_path, res['label'], res['value'], res['raw_text'], res['annotated_path'])
                    st.info("Saved to database.")
                else:
                    st.error("No OCR result.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

st.markdown("---")
if st.checkbox("Show saved readings"):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, category, label, value, timestamp FROM readings ORDER BY id DESC").fetchall()
    conn.close()
    st.table(rows)
