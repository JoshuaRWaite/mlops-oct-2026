"""Step 2: a tiny web page for uploading the zip of house photos."""
import shutil
import zipfile
from pathlib import Path

import streamlit as st

from config import UPLOAD_DIR

EXTENSIONS = {".png", ".jpg", ".jpeg"}

st.title("House Image Upload")
st.write("Upload a **.zip** of house photos (png / jpg). Subfolders are fine.")

uploaded = st.file_uploader("Choose a ZIP file", type="zip")

# Streamlit re-runs this whole script on every interaction, so only extract once per upload.
if uploaded is not None and st.session_state.get("extracted_id") != uploaded.file_id:
    target = Path(UPLOAD_DIR)
    if target.exists():
        shutil.rmtree(target)  # replace the previous upload instead of mixing
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(uploaded) as zf:
            zf.extractall(target)
        st.session_state["extracted_id"] = uploaded.file_id
    except zipfile.BadZipFile:
        st.error("That doesn't look like a valid zip file.")

if uploaded is not None and st.session_state.get("extracted_id") == uploaded.file_id:
    images = sorted(
        p for p in Path(UPLOAD_DIR).rglob("*")
        if p.suffix.lower() in EXTENSIONS and "__MACOSX" not in p.parts and not p.name.startswith("._")
    )
    if images:
        st.success(f"Extracted {len(images)} images into `{UPLOAD_DIR}/`.")
        st.caption("Next step: run `python process_images.py` in the terminal.")
        st.image([str(p) for p in images[:4]], width=150)
    else:
        st.error("The zip was extracted, but it contains no .png / .jpg / .jpeg files.")
