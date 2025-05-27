"""Analysis module for generating tables, plots, and insights.

This module contains functions for analyzing clinical trial data,
generating statistical summaries, and creating visualizations.
"""

import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure as PlotlyFigure
import plotly.graph_objects as go
import re
from collections import Counter, defaultdict
import tempfile
from src.pipeline.utils import upload_to_gcs


from src.pipeline.config import settings
from src.pipeline.utils import get_timestamp, log_execution_time, logger
from src.pipeline.gemini_utils import generate_pipeline_insights


def get_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        Year as integer or None
    """
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except (ValueError, AttributeError, IndexError):
        return None


def generate_summary_statistics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate summary statistics for clinical trials data.
    
    Args:
        df: DataFrame with clinical trial data
        
    Returns:
        Dictionary with summary statistics
    """
    # STEP 1: Generate summary statistics
    logger.info("Generating summary statistics")
    
    stats = {}
    
    # Overall counts
    stats["total_trials"] = len(df)
    
    # Make sure required columns exist to prevent KeyError
    if 'overall_status' in df.columns:
        status_col = df['overall_status'].astype(str).str.strip().str.lower()
        logger.info(f"Unique overall_status values: {status_col.unique()}")
        stats["completed_trials"] = (status_col == "completed").sum()
        stats["ongoing_trials"] = status_col.isin(["recruiting", "active, not recruiting"]).sum()
    else:
        logger.warning("Column 'overall_status' not found in DataFrame. Using zeros for status counts.")
        stats["completed_trials"] = 0
        stats["ongoing_trials"] = 0
    
    # Enrollment statistics - check if column exists
    if 'enrollment_count' in df.columns:
        enrollment_column = 'enrollment_count'
    elif 'enrollment' in df.columns:
        enrollment_column = 'enrollment'
    else:
        enrollment_column = None
        
    if enrollment_column and not df[enrollment_column].empty:
        enrollment_stats = df[enrollment_column].describe()
        stats["enrollment"] = {
            "mean": enrollment_stats["mean"],
            "median": enrollment_stats["50%"],
            "min": enrollment_stats["min"],
            "max": enrollment_stats["max"],
            "q1": enrollment_stats["25%"],
            "q3": enrollment_stats["75%"],
        }
    else:
        logger.warning("Enrollment data not available or empty")
        stats["enrollment"] = {
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "q1": 0,
            "q3": 0,
        }
    
    # Duration statistics - check if column exists
    if 'duration_days' in df.columns and not df['duration_days'].dropna().empty:
        duration_stats = df["duration_days"].dropna().describe()
        stats["duration_days"] = {
            "mean": duration_stats["mean"],
            "median": duration_stats["50%"],
            "min": duration_stats["min"],
            "max": duration_stats["max"],
            "q1": duration_stats["25%"],
            "q3": duration_stats["75%"],
        }
    else:
        logger.warning("Duration data not available or empty")
        stats["duration_days"] = {
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "q1": 0,
            "q3": 0,
        }
    
    # Counts by phase - check if column exists
    phase_column = None
    for col in ['study_phase', 'phase']:
        if col in df.columns:
            phase_column = col
            break
            
    if phase_column and not df[phase_column].empty:
        stats["phase_counts"] = df[phase_column].value_counts().to_dict()
    else:
        logger.warning("Phase data not available or empty")
        stats["phase_counts"] = {}
    
    return stats


def generate_modality_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate counts of trials by modality (case-insensitive), excluding placebo and unknown."""
    logger.info("Generating modality counts")
    exploded_df = df.explode("modalities")
    # Standardize case
    if 'modalities' in exploded_df.columns:
        exploded_df["modalities"] = exploded_df["modalities"].str.lower()
    # Exclude placebo and unknown
    modality_counts = exploded_df["modalities"].value_counts().reset_index()
    modality_counts.columns = ["modality", "count"]
    modality_counts = modality_counts[~modality_counts["modality"].isin(["unknown", "placebo"])]
    modality_counts = modality_counts.sort_values("count", ascending=False).head(30)
    return modality_counts


def generate_target_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate counts of trials by target (case-insensitive), excluding placebo and unknown."""
    logger.info("Generating target counts")
    exploded_df = df.explode("targets")
    # Standardize case
    if 'targets' in exploded_df.columns:
        exploded_df["targets"] = exploded_df["targets"].str.lower()
    # Exclude placebo and unknown
    target_counts = exploded_df["targets"].value_counts().reset_index()
    target_counts.columns = ["target", "count"]
    target_counts = target_counts[~target_counts["target"].isin(["unknown", "placebo"])]
    target_counts = target_counts.head(30)
    return target_counts


def generate_yearly_modality_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate data for yearly modality trends.
    
    Args:
        df: DataFrame with clinical trial data
        
    Returns:
        DataFrame with yearly modality data
    """
    # STEP 4: Generate yearly modality data
    logger.info("Generating yearly modality data")
    
    # Check if required columns exist
    if 'start_date' not in df.columns or 'modalities' not in df.columns or df.empty:
        logger.warning("Cannot generate yearly modality data: missing required columns or empty DataFrame")
        # Return an empty DataFrame with expected structure
        return pd.DataFrame(columns=['year'])
    
    # Extract year from start date
    df_with_year = df.copy()
    df_with_year["year"] = df_with_year["start_date"].apply(get_year_from_date)
    
    # Filter rows with valid year and modalities
    df_with_year = df_with_year.dropna(subset=["year", "modalities"])
    
    # If after filtering we have no data, return empty DataFrame
    if df_with_year.empty:
        logger.warning("No valid data for yearly modality analysis after filtering")
        return pd.DataFrame(columns=['year'])
    
    # Explode modalities to get one row per modality per trial
    exploded_df = df_with_year.explode("modalities")
    
    # Group by year and modality to count trials
    yearly_modality = exploded_df.groupby(["year", "modalities"]).size().reset_index()
    yearly_modality.columns = ["year", "modality", "count"]
    
    # Pivot to get modalities as columns
    pivot_df = yearly_modality.pivot_table(
        index="year", columns="modality", values="count", fill_value=0
    ).reset_index()
    
    return pivot_df


def plot_top_sponsors(df: pd.DataFrame, output_dir: Optional[Path] = None, top_n: int = 30, timestamp: Optional[str] = None):
    """Bar chart of top sponsors by number of trials."""
    if 'lead_sponsor' not in df.columns or df['lead_sponsor'].dropna().empty:
        logger.warning("'lead_sponsor' column missing or empty, skipping top sponsors plot.")
        return
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    sponsor_counts = df['lead_sponsor'].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    sponsor_counts.plot(kind='bar')
    plt.title(f"Top {top_n} Sponsors by Number of Trials")
    plt.xlabel("Sponsor")
    plt.ylabel("Number of Trials")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=f"_top_{top_n}_sponsors_{timestamp}.png", delete=True) as tmp_png:
        plt.savefig(tmp_png.name, dpi=300)
        upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/top_{top_n}_sponsors_{timestamp}.png")
    plt.close()


def plot_status_distribution(df: pd.DataFrame, output_dir: Optional[Path] = None, timestamp: Optional[str] = None):
    """Pie chart of trial status distribution."""
    if 'overall_status' not in df.columns or df['overall_status'].dropna().empty:
        logger.warning("'overall_status' column missing or empty, skipping status distribution plot.")
        return
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    status_counts = df['overall_status'].value_counts()
    plt.figure(figsize=(8, 6))
    status_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title("Trial Status Distribution")
    plt.ylabel("")
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=f"_trial_status_distribution_{timestamp}.png", delete=True) as tmp_png:
        plt.savefig(tmp_png.name, dpi=300)
        upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/trial_status_distribution_{timestamp}.png")
    plt.close()


def plot_enrollment_by_sponsor_plotly(df, output_dir=None, top_n=30, timestamp=None):
    """Interactive horizontal boxplot of enrollment sizes by top sponsors (Plotly HTML only)."""
    if 'lead_sponsor' not in df.columns or 'enrollment_count' not in df.columns:
        logger.warning("'lead_sponsor' or 'enrollment_count' column missing, skipping enrollment by sponsor plot.")
        return
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    top_sponsors = df['lead_sponsor'].value_counts().head(top_n).index
    filtered = df[df['lead_sponsor'].isin(top_sponsors) & df['enrollment_count'].notna()]
    if filtered.empty:
        logger.warning("No data for enrollment by sponsor plot after filtering.")
        return
    # Sort sponsors by median enrollment for better visual order
    sponsor_order = (
        filtered.groupby('lead_sponsor')['enrollment_count']
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig = px.box(
        filtered,
        y='lead_sponsor',
        x='enrollment_count',
        orientation='h',
        category_orders={'lead_sponsor': sponsor_order},
        title=f'Enrollment Size by Top {top_n} Sponsors',
        labels={'lead_sponsor': 'Sponsor', 'enrollment_count': 'Enrollment Count'},
        height=40 * len(sponsor_order) + 200  # Dynamic height for readability
    )
    fig.update_layout(yaxis_title="Sponsor", xaxis_title="Enrollment Count")
    with tempfile.NamedTemporaryFile(suffix=f"_enrollment_by_top_{top_n}_sponsors_{timestamp}.html", delete=True) as tmp_html:
        fig.write_html(tmp_html.name)
        upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/enrollment_by_top_{top_n}_sponsors_{timestamp}.html")


def generate_sankey_data(df: pd.DataFrame, top_n: int = 30):
    """Prepare data for a Sankey diagram: Sponsor -> Modality -> Target."""
    if not all(col in df.columns for col in ['lead_sponsor', 'modalities', 'targets']):
        return None, None, None
    # Flatten the data for Sankey
    rows = []
    for _, row in df.iterrows():
        sponsor = row['lead_sponsor']
        modalities = row['modalities'] if isinstance(row['modalities'], list) else [row['modalities']]
        targets = row['targets'] if isinstance(row['targets'], list) else [row['targets']]
        for m in modalities:
            for t in targets:
                rows.append((sponsor, m, t))
    sankey_df = pd.DataFrame(rows, columns=['sponsor', 'modality', 'target'])
    # Filter to top N sponsors, modalities, targets
    top_sponsors = sankey_df['sponsor'].value_counts().head(top_n).index
    top_modalities = sankey_df['modality'].value_counts().head(top_n).index
    top_targets = sankey_df['target'].value_counts().head(top_n).index
    sankey_df = sankey_df[
        sankey_df['sponsor'].isin(top_sponsors) &
        sankey_df['modality'].isin(top_modalities) &
        sankey_df['target'].isin(top_targets)
    ]
    return sankey_df, top_sponsors, top_modalities, top_targets


def normalize_outcome_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase, remove punctuation, strip whitespace
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)  # Remove punctuation
    text = text.strip()
    return text


def plot_top_outcomes_normalized(df: pd.DataFrame, outcome_col: str, title: str, top_n: int = 10):
    """Horizontal bar chart of top N normalized outcome measures (primary or secondary), excluding placebo."""
    # Flatten and normalize the list of outcomes
    all_outcomes = [normalize_outcome_text(item) for sublist in df[outcome_col].dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]
    all_outcomes = [x for x in all_outcomes if x and x != "placebo"]
    if not all_outcomes:
        return None
    outcome_counts = Counter(all_outcomes).most_common(top_n)
    outcome_df = pd.DataFrame(outcome_counts, columns=["Outcome Measure", "Count"])
    fig = px.bar(
        outcome_df,
        x="Count",
        y="Outcome Measure",
        orientation="h",
        title=title + " (Normalized)",
        labels={"Outcome Measure": "Outcome Measure", "Count": "Count"},
    )
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    return fig


def plot_age_distribution(df: pd.DataFrame):
    """Boxplot and histogram for minimum and maximum age (in years)."""
    age_df = df[['minimum_age', 'maximum_age']].copy()
    # Melt for boxplot
    melted = age_df.melt(var_name='Age Type', value_name='Age (years)')
    melted = melted.dropna(subset=['Age (years)'])
    if melted.empty:
        return None, None
    box = px.box(melted, x='Age Type', y='Age (years)', title='Age Distribution (Boxplot)')
    hist = px.histogram(melted, x='Age (years)', color='Age Type', barmode='overlay', nbins=20, title='Age Distribution (Histogram)')
    return box, hist


def plot_age_group_distribution(df: pd.DataFrame):
    """Bar chart for age group (categorical) distribution."""
    if 'age_groups' not in df.columns:
        return None
    # Explode the list of age groups for counting
    exploded = df.explode('age_groups')
    group_counts = exploded['age_groups'].value_counts().reset_index()
    group_counts.columns = ['Age Group', 'Count']
    if group_counts.empty:
        return None
    fig = px.bar(
        group_counts,
        x='Age Group',
        y='Count',
        title='Distribution of Age Groups in Trials',
        labels={'Age Group': 'Age Group', 'Count': 'Number of Trials'},
    )
    fig.update_layout(xaxis_tickangle=-30, height=500)
    return fig


def plot_modality_by_phase_distribution(df: pd.DataFrame, output_dir: Path, timestamp: str):
    """
    Create a stacked bar chart: X=Trial Phase, Y=Count of Trials, Color=Modality.
    """
    if 'study_phase' not in df.columns or 'modalities' not in df.columns:
        logger.warning("Cannot create modality-by-phase chart: missing columns.")
        return
    exploded = df.explode('modalities')
    exploded = exploded.dropna(subset=['study_phase', 'modalities'])
    if exploded.empty:
        logger.warning("No data for modality-by-phase chart after filtering.")
        return
    # Normalize phase values
    def map_phase(phase):
        if not isinstance(phase, str):
            return "Other"
        p = phase.strip().upper()
        mapping = {
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "PHASE1/PHASE2": "Phase 1/2",
            "PHASE2/PHASE3": "Phase 2/3",
            "PHASE1/PHASE2/PHASE3": "Phase 1/2/3",
            "NOT APPLICABLE": "N/A",
            "N/A": "N/A",
            "EARLY PHASE 1": "Early Phase 1",
        }
        return mapping.get(p, "Other")
    exploded['study_phase'] = exploded['study_phase'].map(map_phase)
    # Lowercase and clean modalities
    exploded['modalities'] = exploded['modalities'].str.lower().str.strip()
    exploded = exploded[exploded['modalities'] != "unknown"]
    grouped = exploded.groupby(['study_phase', 'modalities']).size().reset_index(name='count')
    phase_order = [
        'Phase 1', 'Phase 1/2', 'Phase 2', 'Phase 2/3', 'Phase 3', 'Phase 4', 'N/A', 'Early Phase 1', 'Other', 'Phase 1/2/3'
    ]
    grouped['study_phase'] = pd.Categorical(grouped['study_phase'], categories=phase_order, ordered=True)
    fig = px.bar(
        grouped,
        x='study_phase',
        y='count',
        color='modalities',
        title='Modality-by-Phase Distribution of Clinical Trials',
        labels={'study_phase': 'Trial Phase', 'count': 'Number of Trials', 'modalities': 'Modality'},
        category_orders={'study_phase': phase_order}
    )
    fig.update_layout(barmode='stack', height=600)
    with tempfile.NamedTemporaryFile(suffix=f"_modality_by_phase_distribution_{timestamp}.html", delete=True) as tmp_html:
        fig.write_html(tmp_html.name)
        upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/modality_by_phase_distribution_{timestamp}.html")


@log_execution_time
def create_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None, timestamp: Optional[str] = None
) -> Dict[str, PlotlyFigure]:
    """Create plots from clinical trial data.
    
    Args:
        df: DataFrame with clinical trial data
        output_dir: Directory to save plots (defaults to settings)
        timestamp: Run-level timestamp for file naming and GCS folder
        
    Returns:
        Dictionary of Plotly figures
    """
    logger.info("Creating plots")
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    plots = {}
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    caption = f"Source: clinicaltrials.gov, retrieved {run_date}"
    if df.empty:
        logger.warning("Cannot create plots: DataFrame is empty")
        return plots
    # Modality-by-Phase Distribution Chart
    try:
        plot_modality_by_phase_distribution(df, output_dir, timestamp)
    except Exception as e:
        logger.error(f"Error creating modality-by-phase distribution chart: {e}")
    # Stacked area chart of modality shares over time
    try:
        yearly_modality_data = generate_yearly_modality_data(df)
        if not yearly_modality_data.empty and len(yearly_modality_data.columns) > 1:
            modality_columns = [col for col in yearly_modality_data.columns if col != "year"]
            melted_df = pd.melt(
                yearly_modality_data,
                id_vars=["year"],
                value_vars=modality_columns,
                var_name="modality",
                value_name="count",
            )
            fig_modality = px.area(
                melted_df,
                x="year",
                y="count",
                color="modality",
                title="Clinical Trials by Modality Over Time",
                labels={"year": "Year", "count": "Number of Trials", "modality": "Modality"},
            )
            fig_modality.update_layout(
                autosize=True,
                height=600,
                legend_title="Modality",
                annotations=[
                    dict(
                        text=caption,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.15,
                        font=dict(size=10),
                    )
                ],
            )
            with tempfile.NamedTemporaryFile(suffix=f"_modality_over_time_{timestamp}.html", delete=True) as tmp_html:
                fig_modality.write_html(tmp_html.name)
                upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/modality_over_time_{timestamp}.html")
        else:
            logger.warning("Skipping modality over time plot: insufficient data")
    except Exception as e:
        logger.error(f"Error creating modality over time plot: {e}")
    # Boxplot of trial durations by phase
    try:
        if 'duration_days' in df.columns and 'study_phase' in df.columns:
            duration_data = df.dropna(subset=["duration_days", "study_phase"]).copy()
            if not duration_data.empty:
                def map_phase(phase):
                    if not isinstance(phase, str):
                        return "Other"
                    p = phase.strip().upper()
                    mapping = {
                        "PHASE1": "Phase 1",
                        "PHASE2": "Phase 2",
                        "PHASE3": "Phase 3",
                        "PHASE4": "Phase 4",
                        "PHASE1/PHASE2": "Phase 1/2",
                        "PHASE2/PHASE3": "Phase 2/3",
                        "PHASE1/PHASE2/PHASE3": "Phase 1/2/3",
                        "NOT APPLICABLE": "N/A",
                        "N/A": "N/A",
                        "EARLY PHASE 1": "Early Phase 1",
                    }
                    return mapping.get(p, "Other")
                duration_data["simplified_phase"] = duration_data["study_phase"].map(map_phase)
                phase_order = ["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "N/A", "Early Phase 1", "Other", "Phase 1/2/3"]
                fig_duration = px.box(
                    duration_data,
                    x="simplified_phase",
                    y="duration_days",
                    category_orders={"simplified_phase": phase_order},
                    title="Trial Duration by Phase",
                    labels={
                        "simplified_phase": "Study Phase",
                        "duration_days": "Duration (days)",
                    },
                )
                fig_duration.update_layout(
                    autosize=True,
                    height=600,
                    annotations=[
                        dict(
                            text=caption,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.15,
                            font=dict(size=10),
                        )
                    ],
                )
                with tempfile.NamedTemporaryFile(suffix=f"_duration_by_phase_{timestamp}.html", delete=True) as tmp_html:
                    fig_duration.write_html(tmp_html.name)
                    upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/duration_by_phase_{timestamp}.html")
            else:
                logger.warning("Skipping duration by phase plot: no valid data after filtering")
        else:
            logger.warning("Skipping duration by phase plot: required columns missing")
    except Exception as e:
        logger.error(f"Error creating duration by phase plot: {e}")
    # Top Sponsors (Plotly)
    try:
        if 'lead_sponsor' in df.columns and not df['lead_sponsor'].dropna().empty:
            sponsor_counts = df['lead_sponsor'].value_counts().head(30).reset_index()
            sponsor_counts.columns = ['sponsor', 'count']
            fig_sponsors = px.bar(
                sponsor_counts,
                x='sponsor',
                y='count',
                title='Top 30 Sponsors by Number of Trials',
                labels={'sponsor': 'Sponsor', 'count': 'Number of Trials'},
            )
            fig_sponsors.update_layout(
                autosize=True,
                height=600,
                xaxis_tickangle=-45,
                annotations=[
                    dict(
                        text=caption,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.15,
                        font=dict(size=10),
                    )
                ],
            )
            with tempfile.NamedTemporaryFile(suffix=f"_top_sponsors_{timestamp}.html", delete=True) as tmp_html:
                fig_sponsors.write_html(tmp_html.name)
                upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/top_sponsors_{timestamp}.html")
        else:
            logger.warning("Skipping top sponsors plot: 'lead_sponsor' column missing or empty.")
    except Exception as e:
        logger.error(f"Error creating top sponsors plot: {e}")
    # Sankey chart (Plotly)
    try:
        sankey_df, top_sponsors, top_modalities, top_targets = generate_sankey_data(df, top_n=30)
        if sankey_df is not None and not sankey_df.empty:
            sponsor_nodes = list(top_sponsors)
            modality_nodes = list(top_modalities)
            target_nodes = list(top_targets)
            nodes = sponsor_nodes + modality_nodes + target_nodes
            node_indices = {name: i for i, name in enumerate(nodes)}
            sponsor_modality = sankey_df.groupby(['sponsor', 'modality']).size().reset_index(name='count')
            modality_target = sankey_df.groupby(['modality', 'target']).size().reset_index(name='count')
            links_sponsor_modality = dict(
                source=[node_indices[row['sponsor']] for _, row in sponsor_modality.iterrows()],
                target=[node_indices[row['modality']] for _, row in sponsor_modality.iterrows()],
                value=[row['count'] for _, row in sponsor_modality.iterrows()]
            )
            links_modality_target = dict(
                source=[node_indices[row['modality']] for _, row in modality_target.iterrows()],
                target=[node_indices[row['target']] for _, row in modality_target.iterrows()],
                value=[row['count'] for _, row in modality_target.iterrows()]
            )
            link = dict(
                source=links_sponsor_modality['source'] + links_modality_target['source'],
                target=links_sponsor_modality['target'] + links_modality_target['target'],
                value=links_sponsor_modality['value'] + links_modality_target['value']
            )
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                ),
                link=link
            )])
            fig_sankey.update_layout(
                title_text="Sponsor → Modality → Target Relationships (Top 30 Each)",
                font_size=12,
                height=700
            )
            with tempfile.NamedTemporaryFile(suffix=f"_sankey_sponsor_modality_target_{timestamp}.html", delete=True) as tmp_html:
                fig_sankey.write_html(tmp_html.name)
                upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/sankey_sponsor_modality_target_{timestamp}.html")
        else:
            logger.warning("Skipping Sankey plot: insufficient data")
    except Exception as e:
        logger.error(f"Error creating Sankey plot: {e}")
    # Enrollment by Sponsor (Plotly HTML only)
    try:
        plot_enrollment_by_sponsor_plotly(df, output_dir, top_n=30, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Error creating enrollment by sponsor plot: {e}")
    logger.info(f"Created {len(plots)} plots")
    return plots


def generate_static_matplotlib_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None, timestamp: Optional[str] = None
) -> None:
    """Generate static matplotlib plots for reporting.
    
    Args:
        df: DataFrame with clinical trial data
        output_dir: Directory to save plots (defaults to settings)
        timestamp: Run-level timestamp for file naming and GCS folder
    """
    logger.info("Generating static matplotlib plots")
    if df.empty:
        logger.warning("Cannot create static plots: DataFrame is empty")
        return
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    caption = f"Source: clinicaltrials.gov, retrieved {run_date}"
    # 1. Bar chart of top modalities
    try:
        if 'modalities' in df.columns:
            modality_counts = generate_modality_counts(df)
            if not modality_counts.empty:
                modality_counts = modality_counts[modality_counts["modality"] != "unknown"]
                modality_counts = modality_counts.sort_values("count", ascending=False).head(30)
                plt.figure(figsize=(10, 6))
                bars = plt.bar(modality_counts["modality"], modality_counts["count"])
                plt.title("Top 30 Modalities in Clinical Trials")
                plt.xlabel("Modality")
                plt.ylabel("Number of Trials")
                plt.xticks(rotation=45, ha="right")
                plt.figtext(0.5, 0.01, caption, ha="center", fontsize=9)
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix=f"_top_modalities_{timestamp}.png", delete=True) as tmp_png:
                    plt.savefig(tmp_png.name, dpi=300)
                    upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/top_modalities_{timestamp}.png")
                plt.close()
            else:
                logger.warning("Skipping top modalities plot: no valid data after filtering")
        else:
            logger.warning("Skipping top modalities plot: 'modalities' column missing")
    except Exception as e:
        logger.error(f"Error creating top modalities plot: {e}")
    # 2. Pie chart of trial phases
    try:
        phase_column = None
        for col in ['study_phase', 'phase']:
            if col in df.columns:
                phase_column = col
                break
        if phase_column:
            phase_counts = df[phase_column].value_counts()
            if not phase_counts.empty:
                plt.figure(figsize=(10, 8))
                plt.pie(
                    phase_counts,
                    labels=phase_counts.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    shadow=False,
                )
                plt.title("Distribution of Trial Phases")
                plt.figtext(0.5, 0.01, caption, ha="center", fontsize=9)
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix=f"_phase_distribution_{timestamp}.png", delete=True) as tmp_png:
                    plt.savefig(tmp_png.name, dpi=300)
                    upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/phase_distribution_{timestamp}.png")
                plt.close()
            else:
                logger.warning("Skipping phase distribution plot: no valid data after filtering")
        else:
            logger.warning("Skipping phase distribution plot: phase column missing")
    except Exception as e:
        logger.error(f"Error creating phase distribution plot: {e}")
    # 3. Bar chart of top targets
    try:
        if 'targets' in df.columns:
            target_counts = generate_target_counts(df)
            if not target_counts.empty:
                plt.figure(figsize=(12, 8))
                bars = plt.barh(
                    target_counts["target"][::-1], target_counts["count"][::-1]
                )
                plt.title("Top 30 Protein Targets in Clinical Trials")
                plt.xlabel("Number of Trials")
                plt.ylabel("Target")
                plt.figtext(0.5, 0.01, caption, ha="center", fontsize=9)
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix=f"_top_targets_{timestamp}.png", delete=True) as tmp_png:
                    plt.savefig(tmp_png.name, dpi=300)
                    upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/top_targets_{timestamp}.png")
                plt.close()
            else:
                logger.warning("Skipping top targets plot: no valid data after filtering")
        else:
            logger.warning("Skipping top targets plot: 'targets' column missing")
    except Exception as e:
        logger.error(f"Error creating top targets plot: {e}")
    # Add new static plots
    plot_top_sponsors(df, output_dir, top_n=30, timestamp=timestamp)
    plot_status_distribution(df, output_dir, timestamp=timestamp)
    logger.info("Static plots generated")


@log_execution_time
def analyze_trials(
    df: pd.DataFrame, timestamp: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """Analyze clinical trial data and generate insights.
    
    Args:
        df: DataFrame with clinical trial data
        timestamp: Timestamp string for file naming
        
    Returns:
        Tuple of (summary stats dictionary, insights text)
    """
    # STEP 8: Run full analysis pipeline
    logger.info("Running full analysis pipeline")
    
    if timestamp is None:
        timestamp = get_timestamp()
    
    # Handle empty DataFrame case
    if df.empty:
        logger.warning("Cannot perform full analysis: DataFrame is empty")
        empty_stats = {
            "total_trials": 0,
            "completed_trials": 0,
            "ongoing_trials": 0,
            "enrollment": {
                "mean": 0, "median": 0, "min": 0, "max": 0, "q1": 0, "q3": 0
            },
            "duration_days": {
                "mean": 0, "median": 0, "min": 0, "max": 0, "q1": 0, "q3": 0
            },
            "phase_counts": {}
        }
        
        empty_insights = """
        ## No Clinical Trials Data Available
        
        No clinical trials data was found for the requested criteria. This could be due to:
        
        - API connection issues with ClinicalTrials.gov
        - No trials matching the specified search criteria
        - Data filtering during processing that removed all entries
        
        Please check the logs for more details.
        """
        
        # Save empty insights to file
        insights_path = settings.paths.processed_data / f"insights_{timestamp}.md"
        settings.paths.processed_data.mkdir(parents=True, exist_ok=True)
        with open(insights_path, "w") as f:
            f.write(empty_insights)
        
        # Save empty stats to file
        stats_path = settings.paths.processed_data / f"stats_{timestamp}.json"
        import json
        with open(stats_path, "w") as f:
            json.dump(empty_stats, f, indent=2)
            
        return empty_stats, empty_insights
    
    # Generate summary statistics
    try:
        summary_stats = generate_summary_statistics(df)
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")
        summary_stats = {
            "total_trials": len(df),
            "completed_trials": 0,
            "ongoing_trials": 0,
            "enrollment": {
                "mean": 0, "median": 0, "min": 0, "max": 0, "q1": 0, "q3": 0
            },
            "duration_days": {
                "mean": 0, "median": 0, "min": 0, "max": 0, "q1": 0, "q3": 0
            },
            "phase_counts": {}
        }
    
    # --- NEW: Quantitative summary extraction and visualizations ---
    output_dir = settings.paths.figures
    if timestamp is None:
        timestamp = get_timestamp()
    # Age
    if 'minimum_age' in df.columns and 'maximum_age' in df.columns:
        plot_age_quartiles_box(df, output_dir, timestamp)
    plot_enrollment_quartiles_box(df, output_dir, timestamp)
    # --- END NEW ---

    # --- Outcome ranking and CSV output ---
    def normalize_outcome(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[\W_]+", " ", text)
        text = text.strip()
        return text
    def fuzzy_dedupe(outcomes, threshold=0.85):
        from difflib import SequenceMatcher
        deduped = []
        for o in outcomes:
            if not any(SequenceMatcher(None, o, d).ratio() > threshold for d in deduped):
                deduped.append(o)
        return deduped
    def extract_and_rank_outcomes(df, col, top_n=40):
        from collections import Counter
        all_outcomes = [normalize_outcome(item) for sublist in df[col].dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]
        all_outcomes = [x for x in all_outcomes if x and x != "placebo"]
        counts = Counter(all_outcomes)
        ranked = counts.most_common(top_n * 2)  # get more for deduping
        deduped = fuzzy_dedupe([x[0] for x in ranked])[:top_n]
        top_counts = [(o, counts[o]) for o in deduped]
        return top_counts, deduped
    if 'primary_outcomes' in df.columns:
        top_primary, deduped_primary = extract_and_rank_outcomes(df, 'primary_outcomes', top_n=40)
    else:
        top_primary, deduped_primary = [], []
    if 'secondary_outcomes' in df.columns:
        top_secondary, deduped_secondary = extract_and_rank_outcomes(df, 'secondary_outcomes', top_n=40)
    else:
        top_secondary, deduped_secondary = [], []
    # Prepare CSV rows
    rows = []
    for _, row in df.iterrows():
        nct_id = row.get('nct_id', '')
        brief_title = row.get('brief_title', '')
        for outcome in (row.get('primary_outcomes', []) or []):
            norm = normalize_outcome(outcome)
            if norm in deduped_primary:
                rows.append({
                    'nct_id': nct_id,
                    'brief_title': brief_title,
                    'type': 'primary',
                    'outcome': outcome
                })
        for outcome in (row.get('secondary_outcomes', []) or []):
            norm = normalize_outcome(outcome)
            if norm in deduped_secondary:
                rows.append({
                    'nct_id': nct_id,
                    'brief_title': brief_title,
                    'type': 'secondary',
                    'outcome': outcome
                })
    import csv
    csv_path = settings.paths.processed_data / f"top_outcomes_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['nct_id', 'brief_title', 'type', 'outcome'])
        writer.writeheader()
        writer.writerows(rows)
    upload_to_gcs(str(csv_path), f"runs/{timestamp}/top_outcomes_{timestamp}.csv")
    # --- END NEW ---

    # --- Restore plot generation and upload (from old version) ---
    try:
        create_plots(df, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
    try:
        generate_static_matplotlib_plots(df, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Error generating static plots: {e}")

    # --- Restore summary variables for markdown ---
    modalities = get_unique_flat_list(df, 'modalities') if 'modalities' in df.columns else []
    targets = get_unique_flat_list(df, 'targets') if 'targets' in df.columns else []
    sponsors = get_unique_sponsors(df)
    def _flatten_outcomes(df, col):
        outcomes = set()
        for val in df[col].dropna():
            if isinstance(val, list):
                for o in val:
                    if isinstance(o, dict) and o.get('measure'):
                        outcomes.add(o['measure'])
                    elif isinstance(o, str):
                        outcomes.add(o)
            elif isinstance(val, dict) and val.get('measure'):
                outcomes.add(val['measure'])
            elif isinstance(val, str):
                outcomes.add(val)
        return sorted(outcomes)
    primary_outcomes = _flatten_outcomes(df, 'primary_outcomes') if 'primary_outcomes' in df.columns else []
    secondary_outcomes = _flatten_outcomes(df, 'secondary_outcomes') if 'secondary_outcomes' in df.columns else []
    # --- END NEW ---

    # --- LLM insights: pass top 10 normalized outcomes and counts ---
    try:
        insights = generate_llm_insights(
            df,
            top_primary_clusters=[{'label': o, 'count': c, 'examples': [o]} for o, c in top_primary],
            top_secondary_clusters=[{'label': o, 'count': c, 'examples': [o]} for o, c in top_secondary],
            primary_outcomes=deduped_primary,
            secondary_outcomes=deduped_secondary,
            categorized_outcomes=None
        )
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights = f"""
        ## Clinical Trials Analysis 
        
        Analysis completed with {len(df)} trials, but there was an error generating detailed insights.
        Please check the logs for more information.
        
        Error: {str(e)}
        """
    # --- END NEW ---

    # --- Quantitative summary markdown (unchanged) ---
    quantitative_md = f"""
## Quantitative Summary (Manual)

- **Number of explored modalities:** {len(modalities)}
- **List of modalities:** {', '.join(modalities) if modalities else 'N/A'}
- **Number of biological targets:** {len(targets)}
- **List of targets:** {', '.join(targets) if targets else 'N/A'}
- **Number of sponsors:** {len(sponsors)}
- **List of sponsors:** {', '.join(sponsors) if sponsors else 'N/A'}
"""
    if 'minimum_age' in df.columns:
        min_age_quartiles = get_age_quartiles(df, 'minimum_age')
        if min_age_quartiles:
            quantitative_md += f"""
- **Minimum Age (years):** min={min_age_quartiles['min']:.2f}, Q1={min_age_quartiles['q1']:.2f}, median={min_age_quartiles['median']:.2f}, Q3={min_age_quartiles['q3']:.2f}, max={min_age_quartiles['max']:.2f}, mean={min_age_quartiles['mean']:.2f} (n={min_age_quartiles['count']})
"""
    if 'maximum_age' in df.columns:
        max_age_quartiles = get_age_quartiles(df, 'maximum_age')
        if max_age_quartiles:
            quantitative_md += f"""
- **Maximum Age (years):** min={max_age_quartiles['min']:.2f}, Q1={max_age_quartiles['q1']:.2f}, median={max_age_quartiles['median']:.2f}, Q3={max_age_quartiles['q3']:.2f}, max={max_age_quartiles['max']:.2f}, mean={max_age_quartiles['mean']:.2f} (n={max_age_quartiles['count']})
"""
    # Enrollment and duration quartiles (already in summary_stats)
    if 'enrollment' in summary_stats:
        e = summary_stats['enrollment']
        quantitative_md += f"""
- **Enrollment (number of patients):** min={e['min']:.2f}, Q1={e['q1']:.2f}, median={e['median']:.2f}, Q3={e['q3']:.2f}, max={e['max']:.2f}, mean={e['mean']:.2f}
"""
    if 'duration_days' in summary_stats:
        d = summary_stats['duration_days']
        quantitative_md += f"""
- **Trial Duration (days):** min={d['min']:.2f}, Q1={d['q1']:.2f}, median={d['median']:.2f}, Q3={d['q3']:.2f}, max={d['max']:.2f}, mean={d['mean']:.2f}
"""
    # Age group (stdAges) distribution
    if 'age_groups' in df.columns:
        from collections import Counter
        all_groups = [g for sublist in df['age_groups'].dropna() for g in (sublist if isinstance(sublist, list) else [sublist])]
        group_counts = Counter(all_groups)
        if group_counts:
            quantitative_md += "\n- **Age group distribution:**\n"
            for group, count in group_counts.items():
                quantitative_md += f"    - {group}: {count} studies\n"
    # --- END NEW ---

    # Compose final insights (manual and LLM sections)
    final_insights = quantitative_md + "\n" + insights
    # Create directories if they don't exist
    settings.paths.processed_data.mkdir(parents=True, exist_ok=True)
    # Save insights to file
    insights_path = settings.paths.processed_data / f"insights_{timestamp}.md"
    with open(insights_path, "w") as f:
        f.write(final_insights)
    logger.info(f"Analysis completed and saved to {insights_path}")

    # --- Always generate treatment details HTML table ---
    try:
        generate_treatment_details_table(df, settings.paths.figures, timestamp)
    except Exception as e:
        logger.error(f"Error generating treatment details table: {e}")

    return summary_stats, final_insights


# --- NEW: Helper functions for extracting unique lists and quartiles ---
def get_unique_flat_list(df, col):
    """Extract a flat set of unique items from a column of lists or lists of lists."""
    items = set()
    for val in df[col].dropna():
        if isinstance(val, list):
            for v in val:
                if isinstance(v, list):
                    for vv in v:
                        items.add(vv)
                else:
                    items.add(v)
        else:
            items.add(val)
    return sorted(items)

def get_age_quartiles(df, col):
    """Calculate quartiles, min, max, mean for an age column (in years)."""
    import numpy as np
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if vals.empty:
        return None
    return {
        'min': float(vals.min()),
        'q1': float(np.percentile(vals, 25)),
        'median': float(np.percentile(vals, 50)),
        'q3': float(np.percentile(vals, 75)),
        'max': float(vals.max()),
        'mean': float(vals.mean()),
        'count': int(vals.count()),
    }

def get_unique_sponsors(df):
    """Get unique sponsors from lead_sponsor and collaborators."""
    sponsors = set(df['lead_sponsor'].dropna().unique()) if 'lead_sponsor' in df.columns else set()
    if 'collaborators' in df.columns:
        for val in df['collaborators'].dropna():
            if isinstance(val, list):
                sponsors.update(val)
            elif isinstance(val, str):
                sponsors.add(val)
    return sorted(sponsors)

def plot_age_quartiles_box(df, output_dir, timestamp):
    """
    Create a single boxplot for minimum and maximum age (in years) and save as PNG.
    """
    import matplotlib.pyplot as plt
    age_df = df[['minimum_age', 'maximum_age']].copy()
    melted = age_df.melt(var_name='Age Type', value_name='Age (years)')
    melted = melted.dropna(subset=['Age (years)'])
    if melted.empty:
        return
    plt.figure(figsize=(7, 5))
    melted.boxplot(by='Age Type', column='Age (years)', grid=False)
    plt.title('Age Distribution by Type (Boxplot)')
    plt.suptitle("")
    plt.xlabel("Age Type")
    plt.ylabel("Age (years)")
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=f"_age_quartiles_box_{timestamp}.png", delete=True) as tmp_png:
        plt.savefig(tmp_png.name, dpi=300)
        upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/age_quartiles_box_{timestamp}.png")
    plt.close()

def plot_enrollment_quartiles_box(df, output_dir, timestamp):
    """
    Create a single boxplot for enrollment count and save as PNG.
    """
    import matplotlib.pyplot as plt
    col = 'enrollment_count' if 'enrollment_count' in df.columns else 'enrollment'
    if col not in df.columns:
        return
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if vals.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot(vals, vert=False)
    plt.title('Enrollment Distribution (Boxplot)')
    plt.xlabel('Enrollment Count')
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=f"_enrollment_quartiles_box_{timestamp}.png", delete=True) as tmp_png:
        plt.savefig(tmp_png.name, dpi=300)
        upload_to_gcs(tmp_png.name, f"runs/{timestamp}/figures/enrollment_quartiles_box_{timestamp}.png")
    plt.close()

def generate_treatment_details_table(df: pd.DataFrame, output_dir: Optional[Path] = None, timestamp: Optional[str] = None) -> Optional[Path]:
    """Generate a structured HTML table of treatment details for all trials."""
    if 'treatment_details' not in df.columns or df['treatment_details'].dropna().empty:
        logger.warning("No treatment details found in DataFrame.")
        return None
    rows = []
    for _, row in df.iterrows():
        nct_id = row.get('nct_id')
        details = row.get('treatment_details', [])
        for d in details:
            rows.append({
                'NCT ID': nct_id,
                'Intervention Name': d.get('intervention_name'),
                'Type': d.get('intervention_type'),
                'Arm Group': d.get('arm_group_label'),
                'Arm Group Description': d.get('arm_group_description'),
                'Intervention Description': d.get('intervention_description'),
            })
    if not rows:
        logger.warning("No treatment details to tabulate.")
        return None
    table_df = pd.DataFrame(rows)
    if output_dir is None:
        output_dir = settings.paths.figures
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = get_timestamp()
    with tempfile.NamedTemporaryFile(suffix=f"_treatment_details_{timestamp}.html", delete=True) as tmp_html:
        table_df.to_html(tmp_html.name, index=False, escape=False)
        upload_to_gcs(tmp_html.name, f"runs/{timestamp}/figures/treatment_details_{timestamp}.html")
    return f"runs/{timestamp}/figures/treatment_details_{timestamp}.html"

def generate_llm_insights(df: pd.DataFrame, top_primary_clusters=None, top_secondary_clusters=None, primary_outcomes=None, secondary_outcomes=None, categorized_outcomes=None) -> str:
    """
    Generate a detailed insights report using Gemini LLM, summarizing key statistics, trends, and findings from the clinical trials data.
    Falls back to manual bullet points if LLM call fails.
    """
    import json
    from src.pipeline.gemini_utils import generate_pipeline_insights
    if df.empty:
        return "No clinical trials data available for insights generation."
    # Prepare summary statistics
    stats = generate_summary_statistics(df)
    # --- NEW: Convert stats to native types for JSON serialization ---
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj
    stats_native = convert_to_native(stats)
    # --- END NEW ---
    # Top sponsors
    top_sponsors = df['lead_sponsor'].value_counts().head(30).to_dict() if 'lead_sponsor' in df.columns else {}
    # Top modalities
    modalities = []
    if 'modalities' in df.columns:
        modalities = pd.Series([m for sublist in df['modalities'].dropna() for m in (sublist if isinstance(sublist, list) else [sublist])]).value_counts().head(30).to_dict()
    # Top targets
    targets = []
    if 'targets' in df.columns:
        targets = pd.Series([t for sublist in df['targets'].dropna() for t in (sublist if isinstance(sublist, list) else [sublist])]).value_counts().head(30).to_dict()
    # Yearly trend
    yearly_counts = df['start_date'].dropna().apply(get_year_from_date).value_counts().sort_index().to_dict() if 'start_date' in df.columns else {}
    # --- NEW: Add context from environment variables ---
    from src.pipeline.config import settings
    disease = getattr(settings, 'disease', 'N/A')
    year_start = getattr(settings, 'year_start', 'N/A')
    year_end = getattr(settings, 'year_end', 'N/A')
    context_section = f"""
Clinical Trials Context:
- Disease: {disease}
- Start Year: {year_start}
- End Year: {year_end}
"""
    # --- END NEW ---
    # --- NEW: Add top outcome clusters and outcomes to the prompt ---
    def format_clusters_for_prompt(clusters, label):
        if not clusters:
            return f"No {label} clusters found."
        lines = [f"Top {label} Clusters:"]
        for c in clusters:
            lines.append(f"- {c['label']} (n={c['count']}): e.g. {', '.join(c['examples'])}")
        return '\n'.join(lines)
    primary_cluster_section = format_clusters_for_prompt(top_primary_clusters, 'Primary Outcome')
    secondary_cluster_section = format_clusters_for_prompt(top_secondary_clusters, 'Secondary Outcome')
    # --- NEW: Add categorized outcomes to the prompt ---
    def format_categorized_section(categorized_outcomes, outcome_type):
        if not categorized_outcomes or outcome_type not in categorized_outcomes:
            return f"No {outcome_type} outcomes found."
        lines = [f"### {outcome_type.capitalize()} Outcomes by Category:"]
        for cat, outcomes in categorized_outcomes[outcome_type].items():
            lines.append(f"- **{cat}** ({len(outcomes)}):")
            for o in outcomes[:5]:
                lines.append(f"    - {o}")
            if len(outcomes) > 5:
                lines.append(f"    ...and {len(outcomes)-5} more.")
        return '\n'.join(lines)
    if categorized_outcomes:
        categorized_section = (
            format_categorized_section(categorized_outcomes, 'primary') + '\n\n' +
            format_categorized_section(categorized_outcomes, 'secondary')
        )
    else:
        categorized_section = ""
    # --- NEW: Add outcomes to the prompt ---
    primary_outcomes_section = f"Top Primary Outcomes (raw):\n- " + "\n- ".join(primary_outcomes) if primary_outcomes else "No primary outcomes found."
    secondary_outcomes_section = f"Top Secondary Outcomes (raw):\n- " + "\n- ".join(secondary_outcomes) if secondary_outcomes else "No secondary outcomes found."
    # --- END NEW ---
    # Compose a detailed prompt
    prompt = f"""
{context_section}
You are an expert biotech analyst. Analyze the following clinical trials data and generate a detailed, insightful report for a hedge fund or investor audience. Highlight trends, sponsor activity, therapeutic focus, and any notable findings.

Key statistics:
{json.dumps(stats_native, indent=2)}

Top sponsors (by number of trials):
{json.dumps(top_sponsors, indent=2)}

Top modalities (by number of trials):
{json.dumps(modalities, indent=2)}

Top targets (by number of trials):
{json.dumps(targets, indent=2)}

Number of new trials started per year:
{json.dumps(yearly_counts, indent=2)}

{categorized_section}

{primary_outcomes_section}

{secondary_outcomes_section}

### Outcome Analysis – Definitions for Categorization

For the Primary and Secondary Outcomes analysis section, please perform the following:

---

### Instructions:

- Start with a **2–3 sentence summary** of trends in the outcomes (e.g., whether trials focus mostly on safety, biomarkers, etc.).
- Then create a **markdown table** for each of Primary and Secondary Outcomes, using the structure below.

---

### Outcome Analysis – Definitions for Categorization

For the Primary and Secondary Outcomes analysis section, use the following definitions to categorize outcomes:

- **Safety/Tolerability**: Adverse events, tolerability issues, withdrawals, vital signs, lab values, ECGs.
- **Pharmacokinetics (PK)**: Drug concentration metrics (Cmax, Tmax, AUC), absorption, half-life.
- **Biomarkers**: Biological indicators of disease or treatment effect (e.g., mHTT in CSF).
- **Efficacy: motor**: Movement, coordination, chorea, or motor scores (e.g., UHDRS).
- **Efficacy: cognitive**: Cognitive function tests or brain imaging (e.g., vMRI, memory tests).
- **Efficacy: behavioral**: Psychiatric/mood assessments (e.g., C-SSRS).
- **Efficacy: QoL**: Quality of life, functional status, patient-reported well-being.

Please avoid listing the same example in both primary and secondary tables. If an outcome cannot be clearly categorized, explain briefly in a footnote.

### Primary Outcomes

Before the table, write a 2–3 sentence summary of trends in primary outcomes.

Then create a markdown table with:

| Outcome Category | Examples |
|------------------|----------|
| ...              | ...      |

### Secondary Outcomes

Before the table, write a 2–3 sentence summary of trends in secondary outcomes.

Then create a markdown table with:

| Outcome Category | Examples |
|------------------|----------|
| ...              | ...      |
"""

    try:
        llm_report = generate_pipeline_insights(prompt)
        if llm_report and isinstance(llm_report, str):
            return llm_report
        else:
            return "[LLM did not return a valid report. Please check the LLM integration.]"
    except Exception as e:
        return f"[LLM insights generation failed: {str(e)}]" 