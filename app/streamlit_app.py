"""Streamlit dashboard for visualizing clinical trials data."""

import glob
import os
from pathlib import Path

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


def get_latest_data() -> pd.DataFrame:
    """Get the latest processed clinical trials data.
    
    Returns:
        DataFrame with clinical trial data
    """
    # Find all parquet files in the processed data directory
    data_dir = Path("../data/processed")
    parquet_files = list(data_dir.glob("trials_enriched_*.parquet"))
    
    if not parquet_files:
        st.error("No processed data found. Please run the pipeline first.")
        st.stop()
    
    # Get the most recent file
    latest_file = max(parquet_files, key=os.path.getctime)
    st.sidebar.info(f"Using data from: {latest_file.name}")
    
    # Load the data
    return pd.read_parquet(latest_file)


def main():
    """Main function for the Streamlit app."""
    st.title("Clinical Trials Dashboard")
    st.markdown(
        """
        This dashboard visualizes clinical trial data extracted from ClinicalTrials.gov
        and enriched with drug modality and target information.
        """
    )
    
    # Load data
    df = get_latest_data()
    
    # Get dataset information
    disease_list = df["conditions"].str.split(", ").explode().unique().tolist()
    main_disease = disease_list[0] if disease_list else "Unknown"
    
    # Display summary metrics
    st.header(f"Summary for {main_disease} Clinical Trials")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trials", len(df))
        
    with col2:
        completed_trials = df[df["status"] == "Completed"].shape[0]
        st.metric("Completed Trials", completed_trials)
        
    with col3:
        ongoing_trials = df[df["status"].isin(["Recruiting", "Active, not recruiting"])].shape[0]
        st.metric("Ongoing Trials", ongoing_trials)
        
    with col4:
        avg_enrollment = int(df["enrollment"].mean())
        st.metric("Avg. Enrollment", avg_enrollment)

    # --- New: Key Quantitative Visualizations ---
    st.subheader("Key Quantitative Visualizations")
    from src.pipeline.analysis import create_plots
    plots = create_plots(df)
    # Show top primary outcomes
    if "top_primary_outcomes" in plots:
        st.plotly_chart(plots["top_primary_outcomes"], use_container_width=True)
    # Show top secondary outcomes
    if "top_secondary_outcomes" in plots:
        st.plotly_chart(plots["top_secondary_outcomes"], use_container_width=True)
    # Show age boxplot
    if "age_boxplot" in plots:
        st.plotly_chart(plots["age_boxplot"], use_container_width=True)
    # Show age histogram
    if "age_histogram" in plots:
        st.plotly_chart(plots["age_histogram"], use_container_width=True)
    
    # Display trial phases distribution
    st.header("Trial Phases")
    phase_counts = df["study_phase"].value_counts().reset_index()
    phase_counts.columns = ["Phase", "Count"]
    
    fig_phases = px.bar(
        phase_counts,
        x="Phase",
        y="Count",
        title="Distribution of Clinical Trial Phases",
        color="Phase",
    )
    st.plotly_chart(fig_phases, use_container_width=True)
    
    # Display interactive filters
    st.header("Trial Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_phases = st.multiselect(
            "Select Trial Phases",
            df["study_phase"].unique().tolist(),
            default=df["study_phase"].unique().tolist()[:3],
        )
        
    with col2:
        # Extract unique modalities from the exploded list
        all_modalities = []
        for modalities in df["modalities"]:
            if isinstance(modalities, list):
                all_modalities.extend(modalities)
        unique_modalities = list(set(all_modalities))
        
        selected_modalities = st.multiselect(
            "Select Drug Modalities",
            unique_modalities,
            default=unique_modalities[:3] if len(unique_modalities) > 3 else unique_modalities,
        )
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_phases:
        filtered_df = filtered_df[filtered_df["study_phase"].isin(selected_phases)]
        
    if selected_modalities:
        filtered_df = filtered_df[filtered_df["modalities"].apply(
            lambda x: any(m in selected_modalities for m in (x or []))
        )]
    
    # Display filtered results
    st.write(f"Showing {len(filtered_df)} trials")
    st.dataframe(
        filtered_df[["nct_id", "brief_title", "study_phase", "start_date", "status", "enrollment"]],
        hide_index=True,
    )
    
    # Display enrollment vs. duration scatter plot
    st.header("Enrollment vs. Duration")
    
    scatter_df = filtered_df.dropna(subset=["enrollment", "duration_days"])
    
    fig_scatter = px.scatter(
        scatter_df,
        x="enrollment",
        y="duration_days",
        color="study_phase",
        hover_name="brief_title",
        size="enrollment",
        size_max=50,
        opacity=0.7,
        title="Trial Enrollment vs. Duration",
        labels={
            "enrollment": "Number of Participants",
            "duration_days": "Duration (days)",
            "study_phase": "Study Phase",
        },
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Display modality trends over time
    st.header("Modality Trends Over Time")
    
    # Prepare data: extract year from start date and explode modalities
    trend_df = filtered_df.copy()
    trend_df["year"] = pd.to_datetime(trend_df["start_date"]).dt.year
    trend_df = trend_df.dropna(subset=["year", "modalities"])
    
    # Explode the modalities list to get one row per modality per trial
    exploded_df = trend_df.explode("modalities")
    
    # Count trials by year and modality
    yearly_counts = exploded_df.groupby(["year", "modalities"]).size().reset_index()
    yearly_counts.columns = ["Year", "Modality", "Count"]
    
    # Create area chart
    if not yearly_counts.empty:
        fig_trends = px.area(
            yearly_counts,
            x="Year",
            y="Count",
            color="Modality",
            title="Clinical Trials by Modality Over Time",
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.info("No trend data available for the selected filters.")
    
    # Show raw data expander
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)


if __name__ == "__main__":
    main() 