"""Streamlit dashboard for visualizing clinical trials data."""

import glob
import os
from pathlib import Path
import tempfile
from google.cloud import storage
from src.pipeline.utils import download_from_gcs

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

# Set page configuration
st.set_page_config(
    page_title="Clinical Trials Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GCS_BUCKET = "clinical-trial-pipeline-artifacts-bucket"


def get_latest_data() -> pd.DataFrame:
    """Get the latest processed clinical trials data from GCS.
    
    Returns:
        DataFrame with clinical trial data
    """
    # List all parquet files in the bucket under runs/*/trials_enriched_*.parquet
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix="runs/"))
    parquet_blobs = [b for b in blobs if b.name.endswith(".parquet") and "trials_enriched_" in b.name]
    if not parquet_blobs:
        st.error("No processed data found in GCS. Please run the pipeline first.")
        st.stop()
    # Get the most recent file by GCS updated time
    latest_blob = max(parquet_blobs, key=lambda b: b.updated)
    st.sidebar.info(f"Using data from: {latest_blob.name}")
    # Download to a temp file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        download_from_gcs(latest_blob.name, tmp_file.name, bucket_name=GCS_BUCKET)
        df = pd.read_parquet(tmp_file.name)
    return df


def list_runs(bucket):
    # List all unique run folders under runs/
    blobs = list(bucket.list_blobs(prefix="runs/"))
    run_folders = set()
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) > 1 and parts[1]:
            run_folders.add(parts[1])
    return sorted(run_folders, reverse=True)

def list_artifacts(bucket, run):
    # List all files for a given run
    prefix = f"runs/{run}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    figures = []
    data_files = []
    release_files = []
    for b in blobs:
        if b.name.endswith("/"):
            continue
        rel_path = b.name[len(prefix):]
        if rel_path.startswith("figures/"):
            figures.append(b)
        elif rel_path.startswith("release/"):
            release_files.append(b)
        else:
            data_files.append(b)
    return figures, data_files, release_files

def display_artifact(bucket, blob, filetype_hint=None):
    import tempfile
    import streamlit as st
    from PIL import Image
    # Download to temp file
    suffix = Path(blob.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        download_from_gcs(blob.name, tmp_file.name, bucket_name=bucket.name)
        if filetype_hint == "image" or suffix in [".png", ".jpg", ".jpeg"]:
            st.image(tmp_file.name, caption=blob.name)
        elif filetype_hint == "html" or suffix == ".html":
            with open(tmp_file.name, "r", encoding="utf-8") as f:
                html = f.read()
            # Inject CSS for background and text color
            html = (
                "<style>"
                "body, table {background: #fff !important; color: #222 !important;}"
                "table {border-collapse: collapse; width: 100%;}"
                "th, td {border: 1px solid #ddd; padding: 8px;}"
                "</style>"
            ) + html
            st.components.v1.html(html, height=600, scrolling=True)
        elif filetype_hint == "csv" or suffix == ".csv":
            df = pd.read_csv(tmp_file.name)
            st.dataframe(df)
        elif filetype_hint == "parquet" or suffix == ".parquet":
            df = pd.read_parquet(tmp_file.name)
            st.dataframe(df)
        elif filetype_hint == "md" or suffix in [".md", ".markdown"]:
            with open(tmp_file.name, "r", encoding="utf-8") as f:
                md = f.read()
            st.markdown(md)
        else:
            st.download_button(f"Download {blob.name}", tmp_file.name)

def main():
    st.title("Clinical Trials Pipeline Artifacts Browser")
    st.markdown("""
    Browse and view all pipeline run artifacts stored in Google Cloud Storage.
    """)
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    # List available runs
    runs = list_runs(bucket)
    if not runs:
        st.error("No pipeline runs found in GCS bucket.")
        st.stop()
    selected_run = st.selectbox("Select pipeline run (timestamp):", runs)
    st.info(f"Artifacts for run: {selected_run}")
    figures, data_files, release_files = list_artifacts(bucket, selected_run)

    st.subheader("Figures (Plots)")
    if figures:
        for blob in sorted(figures, key=lambda b: b.name):
            col1, col2 = st.columns([3,1])
            with col1:
                if blob.name.endswith(".png"):
                    display_artifact(bucket, blob, filetype_hint="image")
                elif blob.name.endswith(".html"):
                    display_artifact(bucket, blob, filetype_hint="html")
                else:
                    st.write(blob.name)
            with col2:
                st.download_button(f"Download {Path(blob.name).name}", data=blob.download_as_bytes(), file_name=Path(blob.name).name, key=f"download-{blob.name}")
    else:
        st.write("No figures found for this run.")
    st.subheader("Data Files (CSV, Parquet)")
    if data_files:
        for blob in sorted(data_files, key=lambda b: b.name):
            col1, col2 = st.columns([3,1])
            with col1:
                if blob.name.endswith(".csv"):
                    display_artifact(bucket, blob, filetype_hint="csv")
                elif blob.name.endswith(".parquet"):
                    display_artifact(bucket, blob, filetype_hint="parquet")
                else:
                    st.write(blob.name)
            with col2:
                st.download_button(f"Download {Path(blob.name).name}", data=blob.download_as_bytes(), file_name=Path(blob.name).name, key=f"download-{blob.name}")
    else:
        st.write("No data files found for this run.")
    st.subheader("Release Files (Markdown, CSV, etc.)")
    if release_files:
        for blob in sorted(release_files, key=lambda b: b.name):
            col1, col2 = st.columns([3,1])
            with col1:
                if blob.name.endswith(".md"):
                    display_artifact(bucket, blob, filetype_hint="md")
                elif blob.name.endswith(".csv"):
                    display_artifact(bucket, blob, filetype_hint="csv")
                else:
                    st.write(blob.name)
            with col2:
                st.download_button(f"Download {Path(blob.name).name}", data=blob.download_as_bytes(), file_name=Path(blob.name).name, key=f"download-{blob.name}")
    else:
        st.write("No release files found for this run.")


if __name__ == "__main__":
    main() 