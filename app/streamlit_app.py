import streamlit as st
import os
from pathlib import Path
import pandas as pd
import base64

# For markdown rendering
from io import StringIO

# For HTML rendering
from streamlit.components.v1 import html

# For image rendering
from PIL import Image

# Set the data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Supported file types and their extensions
FILE_TYPES = {
    "HTML": [".html", ".htm"],
    "PNG": [".png"],
    "Markdown": [".md"],
    "CSV": [".csv"],
}

# Optionally support JSON, Parquet, etc.
# FILE_TYPES["JSON"] = [".json"]
# FILE_TYPES["Parquet"] = [".parquet"]

def scan_files(base_dir, file_types):
    """Recursively scan for files matching the given types."""
    found = {ftype: [] for ftype in file_types}
    for root, _, files in os.walk(base_dir):
        for file in files:
            fpath = Path(root) / file
            for ftype, exts in file_types.items():
                if any(file.lower().endswith(ext) for ext in exts):
                    found[ftype].append(fpath)
    return found

def render_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Render HTML in an iframe
    st.markdown(f"#### {filepath.name}")
    html(html_content, height=600, scrolling=True)

def render_png_file(filepath):
    st.markdown(f"#### {filepath.name}")
    image = Image.open(filepath)
    st.image(image, use_column_width=True)

def render_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        md_content = f.read()
    st.markdown(f"#### {filepath.name}")
    st.markdown(md_content, unsafe_allow_html=True)

def render_csv_file(filepath):
    st.markdown(f"#### {filepath.name}")
    df = pd.read_csv(filepath)
    st.dataframe(df)
    # Download button
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filepath.name}">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Main app
st.set_page_config(page_title="Clinical Trial Data Dashboard", layout="wide")
st.title("📊 Clinical Trial Data Dashboard")
st.markdown("""
This dashboard automatically discovers and visualizes outputs from the data directory. Select a file type and file from the sidebar to view its contents.
""")

# Scan files
files_by_type = scan_files(DATA_DIR, FILE_TYPES)

# Sidebar navigation
st.sidebar.header("Browse Outputs")
file_type = st.sidebar.selectbox("Select file type", list(FILE_TYPES.keys()))

files = files_by_type[file_type]
if not files:
    st.sidebar.info(f"No {file_type} files found.")
    st.info(f"No {file_type} files found in the data directory.")
    st.stop()

# Group files by subfolder for easier navigation
files_by_folder = {}
for f in files:
    folder = str(f.parent.relative_to(DATA_DIR))
    files_by_folder.setdefault(folder, []).append(f)

folder = st.sidebar.selectbox("Select folder", sorted(files_by_folder.keys()))
file_options = files_by_folder[folder]
file_selected = st.sidebar.selectbox("Select file", [f.name for f in file_options])
file_path = next(f for f in file_options if f.name == file_selected)

# Render the selected file
if file_type == "HTML":
    render_html_file(file_path)
elif file_type == "PNG":
    render_png_file(file_path)
elif file_type == "Markdown":
    render_markdown_file(file_path)
elif file_type == "CSV":
    render_csv_file(file_path)
else:
    st.warning("Unsupported file type.")
