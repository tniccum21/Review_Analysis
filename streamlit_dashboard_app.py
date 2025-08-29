#!/usr/bin/env python3
"""
Streamlit Review Analysis App - Dashboard Module
This app reads analysis results and provides an interactive dashboard for visualization.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
from collections import Counter
import json
import requests
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud library not installed. Run: pip install wordcloud matplotlib")

st.set_page_config(
    page_title="Review Analyzer - Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; margin-bottom: 2rem; }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #dee2e6;
    }
    .filter-section {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'products_data' not in st.session_state:
        st.session_state.products_data = None
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'gender': 'All',
            'product_class': 'All',
            'product_subclass': 'All',
            'size': 'All',
            'end_use': 'All',
            'time_frame': 'All Time'
        }
        # Initialize widget values to match filter values
        st.session_state.gender_filter = 'All'
        st.session_state.class_filter = 'All'
        st.session_state.subclass_filter = 'All'
        st.session_state.size_filter = 'All'
        st.session_state.enduse_filter = 'All'

def load_analysis_results():
    """Load available analysis result files"""
    st.markdown('<div class="section-header">📁 1. Select Analysis Results</div>', unsafe_allow_html=True)
    
    # Find all analysis result CSV files
    csv_files = glob.glob("analysis_results_*.csv") + glob.glob("review_analysis_*.csv")
    csv_files.sort(reverse=True)  # Most recent first
    
    if not csv_files:
        st.warning("No analysis result files found. Please run the analysis app first.")
        st.info("Expected file pattern: analysis_results_*.csv or review_analysis_*.csv")
        
        # Allow manual file upload as fallback
        uploaded_file = st.file_uploader(
            "Or upload an analysis results CSV file:",
            type=['csv']
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, keep_default_na=False, na_values=[''])
                st.session_state.analysis_data = df
                st.session_state.selected_file = uploaded_file.name
                st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
                return True
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return False
        return False
    
    # File selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_file = st.selectbox(
            "Select analysis results file:",
            options=csv_files,
            format_func=lambda x: f"{x} ({os.path.getsize(x)/1024:.1f} KB)"
        )
    
    with col2:
        if st.button("🔄 Refresh File List"):
            st.rerun()
    
    if selected_file and selected_file != st.session_state.selected_file:
        try:
            df = pd.read_csv(selected_file, keep_default_na=False, na_values=[''])
            st.session_state.analysis_data = df
            st.session_state.selected_file = selected_file
            st.success(f"✅ Loaded {len(df)} rows from {selected_file}")
        except Exception as e:
            st.error(f"Error loading {selected_file}: {e}")
            return False
    
    if st.session_state.analysis_data is not None:
        df = st.session_state.analysis_data
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        
        return True
    
    return False

def load_product_metadata():
    """Try to load product metadata if available"""
    if os.path.exists('products.csv'):
        try:
            # Fix DtypeWarning by specifying low_memory=False to infer dtypes properly
            products_df = pd.read_csv('products.csv', low_memory=False)
            st.session_state.products_data = products_df
            return True
        except Exception as e:
            st.warning(f"Could not load product metadata: {e}")
    return False

def display_summary_metrics(df: pd.DataFrame):
    """Display key summary metrics"""
    st.markdown('<div class="section-header">📈 Summary Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        if 'rating' in df.columns:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
        else:
            st.metric("Avg Rating", "N/A")
    
    with col3:
        if 'sentiment' in df.columns:
            pos_pct = (df['sentiment'].value_counts(normalize=True).get('Positive', 0)) * 100
            st.metric("Positive %", f"{pos_pct:.1f}%")
        else:
            st.metric("Positive %", "N/A")
    
    with col4:
        if 'sentiment' in df.columns:
            neg_pct = (df['sentiment'].value_counts(normalize=True).get('Negative', 0)) * 100
            st.metric("Negative %", f"{neg_pct:.1f}%")
        else:
            st.metric("Negative %", "N/A")
    
    with col5:
        if 'problems_mentioned' in df.columns:
            # Count rows that have problems (not 'None' and not empty)
            has_problems = sum(1 for p in df['problems_mentioned'] 
                             if pd.notna(p) and str(p).strip() != '' and str(p).strip() != 'None')
            prob_pct = (has_problems / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Has Problems", f"{prob_pct:.1f}%")
        else:
            st.metric("Has Problems", "N/A")
    
    with col6:
        if 'positive_mentions' in df.columns:
            # Count rows that have positive mentions (not 'None' and not empty)
            has_positives = sum(1 for p in df['positive_mentions'] 
                              if pd.notna(p) and str(p).strip() != '' and str(p).strip() != 'None')
            pos_mentions_pct = (has_positives / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Has Positives", f"{pos_mentions_pct:.1f}%")
        else:
            st.metric("Has Positives", "N/A")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply time frame filter first
    if 'date' in filtered_df.columns and 'time_frame' in st.session_state.filters:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
        time_frame = st.session_state.filters.get('time_frame', 'All Time')
        
        if time_frame != 'All Time' and not filtered_df['date'].isna().all():
            today = pd.Timestamp.now()
            if time_frame == 'Last 30 Days':
                filtered_df = filtered_df[filtered_df['date'] >= today - pd.Timedelta(days=30)]
            elif time_frame == 'Last 90 Days':
                filtered_df = filtered_df[filtered_df['date'] >= today - pd.Timedelta(days=90)]
            elif time_frame == 'Last 6 Months':
                filtered_df = filtered_df[filtered_df['date'] >= today - pd.Timedelta(days=180)]
            elif time_frame == 'Last Year':
                filtered_df = filtered_df[filtered_df['date'] >= today - pd.Timedelta(days=365)]
            elif time_frame == 'Custom Range':
                start_date = pd.to_datetime(st.session_state.filters.get('start_date'))
                end_date = pd.to_datetime(st.session_state.filters.get('end_date'))
                filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Product metadata filters (if product data is loaded and merged)
    if 'GENDER_CODE' in filtered_df.columns and st.session_state.filters['gender'] != 'All':
        filtered_df = filtered_df[filtered_df['GENDER_CODE'] == st.session_state.filters['gender']]
    
    if 'PRODUCT_CLASS_CODE' in filtered_df.columns and st.session_state.filters['product_class'] != 'All':
        filtered_df = filtered_df[filtered_df['PRODUCT_CLASS_CODE'] == st.session_state.filters['product_class']]
    
    if 'PRODUCT_SUB_CLASS_CODE' in filtered_df.columns and st.session_state.filters['product_subclass'] != 'All':
        filtered_df = filtered_df[filtered_df['PRODUCT_SUB_CLASS_CODE'] == st.session_state.filters['product_subclass']]
    
    if 'SIZE_CODE' in filtered_df.columns and st.session_state.filters.get('size', 'All') != 'All':
        filtered_df = filtered_df[filtered_df['SIZE_CODE'] == st.session_state.filters['size']]
    
    if 'END_USE_CODE' in filtered_df.columns and st.session_state.filters['end_use'] != 'All':
        filtered_df = filtered_df[filtered_df['END_USE_CODE'] == st.session_state.filters['end_use']]
    
    return filtered_df

def on_filter_change():
    """Update filter values in session state when any filter changes"""
    if 'gender_filter' in st.session_state:
        st.session_state.filters['gender'] = st.session_state.gender_filter
    if 'class_filter' in st.session_state:
        st.session_state.filters['product_class'] = st.session_state.class_filter
    if 'subclass_filter' in st.session_state:
        st.session_state.filters['product_subclass'] = st.session_state.subclass_filter
    if 'size_filter' in st.session_state:
        st.session_state.filters['size'] = st.session_state.size_filter
    if 'enduse_filter' in st.session_state:
        st.session_state.filters['end_use'] = st.session_state.enduse_filter

def create_filter_section(df: pd.DataFrame):
    """Create filter controls - always visible"""
    st.markdown('<div class="section-header">🔍 Filters</div>', unsafe_allow_html=True)
    
    # Add time frame filter
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_df = df.dropna(subset=['date'])
        if not date_df.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                time_frame = st.selectbox(
                    "Time Frame:",
                    options=['All Time', 'Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Last Year', 'Custom Range'],
                    key="time_frame_filter"
                )
                st.session_state.filters['time_frame'] = time_frame
            
            with col2:
                if time_frame == 'Custom Range':
                    date_col1, date_col2 = st.columns(2)
                    with date_col1:
                        start_date = st.date_input(
                            "Start Date:",
                            value=date_df['date'].min(),
                            min_value=date_df['date'].min(),
                            max_value=date_df['date'].max(),
                            key="start_date_filter"
                        )
                        st.session_state.filters['start_date'] = start_date
                    with date_col2:
                        end_date = st.date_input(
                            "End Date:",
                            value=date_df['date'].max(),
                            min_value=date_df['date'].min(),
                            max_value=date_df['date'].max(),
                            key="end_date_filter"
                        )
                        st.session_state.filters['end_date'] = end_date
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Always show filter dropdowns, whether product data exists or not
    with col1:
        if 'GENDER_CODE' in df.columns:
            gender_options = ['All'] + sorted(df['GENDER_CODE'].dropna().unique().tolist())
            current_val = st.session_state.filters.get('gender', 'All')
            if current_val in gender_options:
                st.selectbox(
                    "Gender:",
                    options=gender_options,
                    index=gender_options.index(current_val),
                    key="gender_filter",
                    on_change=on_filter_change
                )
            else:
                st.selectbox(
                    "Gender:",
                    options=gender_options,
                    index=0,
                    key="gender_filter",
                    on_change=on_filter_change
                )
        else:
            st.session_state.filters['gender'] = st.selectbox(
                "Gender:",
                options=['All'],
                disabled=True,
                key="gender_filter"
            )
    
    with col2:
        if 'PRODUCT_CLASS_CODE' in df.columns:
            class_options = ['All'] + sorted(df['PRODUCT_CLASS_CODE'].dropna().unique().tolist())
            current_val = st.session_state.filters.get('product_class', 'All')
            if current_val in class_options:
                st.selectbox(
                    "Product Class:",
                    options=class_options,
                    index=class_options.index(current_val),
                    key="class_filter",
                    on_change=on_filter_change
                )
            else:
                st.selectbox(
                    "Product Class:",
                    options=class_options,
                    index=0,
                    key="class_filter",
                    on_change=on_filter_change
                )
        else:
            st.session_state.filters['product_class'] = st.selectbox(
                "Product Class:",
                options=['All'],
                disabled=True,
                key="class_filter"
            )
    
    with col3:
        if 'PRODUCT_SUB_CLASS_CODE' in df.columns:
            subclass_options = ['All'] + sorted(df['PRODUCT_SUB_CLASS_CODE'].dropna().unique().tolist())
            current_val = st.session_state.filters.get('product_subclass', 'All')
            if current_val in subclass_options:
                st.selectbox(
                    "Product Sub-Class:",
                    options=subclass_options,
                    index=subclass_options.index(current_val),
                    key="subclass_filter",
                    on_change=on_filter_change
                )
            else:
                st.selectbox(
                    "Product Sub-Class:",
                    options=subclass_options,
                    index=0,
                    key="subclass_filter",
                    on_change=on_filter_change
                )
        else:
            st.session_state.filters['product_subclass'] = st.selectbox(
                "Product Sub-Class:",
                options=['All'],
                disabled=True,
                key="subclass_filter"
            )
    
    with col4:
        if 'SIZE_CODE' in df.columns:
            size_options = ['All'] + sorted(df['SIZE_CODE'].dropna().unique().tolist())
            current_val = st.session_state.filters.get('size', 'All')
            if current_val in size_options:
                st.selectbox(
                    "Size:",
                    options=size_options,
                    index=size_options.index(current_val),
                    key="size_filter",
                    on_change=on_filter_change
                )
            else:
                st.selectbox(
                    "Size:",
                    options=size_options,
                    index=0,
                    key="size_filter",
                    on_change=on_filter_change
                )
        else:
            st.session_state.filters['size'] = st.selectbox(
                "Size:",
                options=['All'],
                disabled=True,
                key="size_filter"
            )
    
    with col5:
        if 'END_USE_CODE' in df.columns:
            end_use_options = ['All'] + sorted(df['END_USE_CODE'].dropna().unique().tolist())
            current_val = st.session_state.filters.get('end_use', 'All')
            if current_val in end_use_options:
                st.selectbox(
                    "End Use:",
                    options=end_use_options,
                    index=end_use_options.index(current_val),
                    key="enduse_filter",
                    on_change=on_filter_change
                )
            else:
                st.selectbox(
                    "End Use:",
                    options=end_use_options,
                    index=0,
                    key="enduse_filter",
                    on_change=on_filter_change
                )
        else:
            st.session_state.filters['end_use'] = st.selectbox(
                "End Use:",
                options=['All'],
                disabled=True,
                key="enduse_filter"
            )
    
    # Show info message if filters are disabled
    if 'GENDER_CODE' not in df.columns:
        st.info("📌 Product filters will be enabled when product metadata (products.csv) is available.")

def create_mentions_chart(df: pd.DataFrame):
    """Create What's Mentioned Most chart"""
    st.markdown("#### What's Mentioned Most?")
    
    # Calculate counts for problems and positives
    problems_counts = {}
    positives_counts = {}
    
    for _, row in df.iterrows():
        # Count problems
        if 'problems_mentioned' in row and pd.notna(row['problems_mentioned']) and row['problems_mentioned'] != 'None':
            for problem in str(row['problems_mentioned']).split('; '):
                problems_counts[problem] = problems_counts.get(problem, 0) + 1
        
        # Count positives
        if 'positive_mentions' in row and pd.notna(row['positive_mentions']) and row['positive_mentions'] != 'None':
            for positive in str(row['positive_mentions']).split('; '):
                positives_counts[positive] = positives_counts.get(positive, 0) + 1
    
    # Get all unique categories mentioned
    all_categories = list(set(list(problems_counts.keys()) + list(positives_counts.keys())))
    
    if not all_categories:
        st.info("No mention data available to display.")
        return
    
    # Calculate percentages and prepare data
    total_reviews = len(df)
    mentions_data = []
    
    for category in all_categories:
        problem_count = problems_counts.get(category, 0)
        positive_count = positives_counts.get(category, 0)
        problem_pct = (problem_count / total_reviews * 100) if total_reviews > 0 else 0
        positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
        
        mentions_data.append({
            'Category': category,
            'Negative %': problem_pct,
            'Positive %': positive_pct,
            'Total %': problem_pct + positive_pct
        })
    
    # Sort by total mentions and take top categories
    mentions_df = pd.DataFrame(mentions_data)
    mentions_df = mentions_df.sort_values('Total %', ascending=False).head(15)
    
    # Create the diverging bar chart
    fig = go.Figure()
    
    # Add negative mentions (left side - orange)
    fig.add_trace(go.Bar(
        y=mentions_df['Category'],
        x=-mentions_df['Negative %'],  # Negative to go left
        name='Negative Mentions',
        orientation='h',
        marker=dict(color='#ff7f0e'),
        text=[f"{x:.0f}%" for x in mentions_df['Negative %']],
        textposition='auto',
        hovertemplate='%{y}<br>Negative: %{text}<extra></extra>'
    ))
    
    # Add positive mentions (right side - blue)
    fig.add_trace(go.Bar(
        y=mentions_df['Category'],
        x=mentions_df['Positive %'],  # Positive to go right
        name='Positive Mentions',
        orientation='h',
        marker=dict(color='#1f77b4'),
        text=[f"{x:.0f}%" for x in mentions_df['Positive %']],
        textposition='auto',
        hovertemplate='%{y}<br>Positive: %{text}<extra></extra>'
    ))
    
    # Update layout
    max_val = max(mentions_df['Negative %'].max(), mentions_df['Positive %'].max()) if not mentions_df.empty else 10
    
    fig.update_layout(
        xaxis=dict(
            title='Percentage of Reviews',
            tickformat='.0f',
            ticksuffix='%',
            range=[-(max_val + 5), max_val + 5]
        ),
        yaxis=dict(
            title='',
            autorange='reversed'
        ),
        barmode='overlay',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    # Add a vertical line at x=0
    fig.add_vline(x=0, line_width=1, line_color='gray')
    
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_distribution_chart(df: pd.DataFrame):
    """Create sentiment distribution chart"""
    if 'sentiment' not in df.columns:
        return
    
    st.markdown("#### Sentiment Distribution")
    
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color_discrete_map={
            'Positive': '#1f77b4',
            'Negative': '#ff7f0e',
            'Neutral': '#aec7e8'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def create_rating_distribution_chart(df: pd.DataFrame):
    """Create average rating by month chart"""
    if 'rating' not in df.columns:
        st.info("No rating data available.")
        return
    
    st.markdown("#### Average Rating Over Time")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df_with_dates = df.dropna(subset=['date', 'rating'])
        
        if not df_with_dates.empty:
            # Group by month and calculate average rating
            df_with_dates['month'] = df_with_dates['date'].dt.to_period('M')
            monthly_ratings = df_with_dates.groupby('month')['rating'].agg(['mean', 'count']).reset_index()
            monthly_ratings['month'] = monthly_ratings['month'].dt.to_timestamp()
            
            # Calculate overall average
            overall_avg = df_with_dates['rating'].mean()
            
            fig = px.line(
                monthly_ratings,
                x='month',
                y='mean',
                markers=True,
                labels={'month': 'Month', 'mean': 'Rating'},
                title=None
            )
            
            fig.update_traces(
                line_color='#4285F4',  # Google blue
                marker=dict(size=6, color='#4285F4')
            )
            
            fig.update_layout(
                yaxis_range=[1, 5],
                yaxis_title='Rating',
                xaxis_title='Month',
                height=400,
                hovermode='x',
                showlegend=False
            )
            
            # Add horizontal trend line for overall average
            fig.add_hline(
                y=overall_avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"avg: {overall_avg:.1f}",
                annotation_position="right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date data available for monthly ratings.")
    else:
        st.info("No date column available for monthly analysis.")

def create_sentiment_area_chart(df: pd.DataFrame):
    """Create area chart showing sentiment percentages over time"""
    if 'date' not in df.columns or 'sentiment' not in df.columns:
        st.info("No date and sentiment data available for chart.")
        return
    
    st.markdown("#### Sentiment Trend (% of Monthly Total)")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_filtered = df.dropna(subset=['date', 'sentiment'])
    
    if df_filtered.empty:
        st.info("No data available for sentiment trend.")
        return
    
    # Group by month and sentiment
    df_filtered['month'] = df_filtered['date'].dt.to_period('M')
    monthly_sentiment = df_filtered.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    
    # Calculate percentages (already normalized to percentages, not decimals)
    monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    # Reset index for plotting
    monthly_sentiment_pct = monthly_sentiment_pct.reset_index()
    monthly_sentiment_pct['month'] = monthly_sentiment_pct['month'].dt.to_timestamp()
    
    # Melt for area chart - keeping the percentage values
    plot_data = monthly_sentiment_pct.melt(id_vars='month', var_name='Sentiment', value_name='Percentage')
    
    # Create the area chart WITHOUT groupnorm since we already calculated percentages
    fig = px.area(
        plot_data,
        x='month',
        y='Percentage',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#6baed6',  # Light blue
            'Negative': '#fd8d3c',  # Orange 
            'Neutral': '#74c476'    # Green
        },
        labels={'month': 'Month', 'Percentage': 'Percentage of Reviews'}
        # Removed groupnorm='percent' - this was causing the issue
    )
    
    fig.update_layout(
        yaxis_tickformat='.0f%%',  # Format as percentage
        yaxis_range=[0, 100],  # 0 to 100 for percentage
        yaxis_title='Percentage of Reviews',
        xaxis_title='Month',
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_problem_categories_chart(df: pd.DataFrame):
    """Create stacked bar chart showing problem categories over time"""
    if 'date' not in df.columns or 'problems_mentioned' not in df.columns:
        st.info("No date and problem data available for chart.")
        return
    
    st.markdown("#### Problem Categories (% of Monthly Total)")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_filtered = df.dropna(subset=['date'])
    
    if df_filtered.empty:
        st.info("No data available for problem categories.")
        return
    
    # Parse problem categories
    problem_rows = []
    for _, row in df_filtered.iterrows():
        if pd.notna(row['problems_mentioned']) and row['problems_mentioned']:
            problems = str(row['problems_mentioned']).split(';')
            for problem in problems:
                problem = problem.strip()
                if problem and problem != 'None':
                    problem_rows.append({
                        'month': row['date'].to_period('M'),
                        'problem': problem
                    })
    
    if not problem_rows:
        st.info("No problem categories found in the data.")
        return
    
    problems_df = pd.DataFrame(problem_rows)
    
    # Count problems by month
    monthly_problems = problems_df.groupby(['month', 'problem']).size().unstack(fill_value=0)
    
    # Calculate percentages (multiply by 100 for percentage values)
    monthly_problems_pct = monthly_problems.div(monthly_problems.sum(axis=1), axis=0) * 100
    
    # Reset index for plotting
    monthly_problems_pct = monthly_problems_pct.reset_index()
    monthly_problems_pct['month'] = monthly_problems_pct['month'].dt.to_timestamp()
    
    # Melt for stacked bar chart
    plot_data = monthly_problems_pct.melt(id_vars='month', var_name='Problem Category', value_name='Percentage')
    
    # Define specific colors for common problem categories
    color_map = {
        'Fit': '#1f77b4',        # Blue
        'Size': '#ff7f0e',       # Orange
        'Material': '#2ca02c',   # Green
        'Color': '#d62728',      # Red
        'Durability': '#9467bd', # Purple
        'Comfort': '#8c564b',    # Brown
        'Design': '#e377c2',     # Pink
        'Price': '#7f7f7f',      # Gray
        'Breathability': '#bcbd22', # Olive
        'Shipping': '#17becf',   # Cyan
        'Packaging': '#aec7e8',  # Light blue
        'Brand': '#ffbb78',      # Light orange
        'Other': '#98df8a'       # Light green
    }
    
    # Add any missing categories with default colors
    unique_problems = plot_data['Problem Category'].unique()
    default_colors = px.colors.qualitative.Set3
    for i, problem in enumerate(unique_problems):
        if problem not in color_map:
            color_map[problem] = default_colors[i % len(default_colors)]
    
    # Create stacked bar chart using go.Bar for proper stacking
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add a trace for each problem category
    for problem in monthly_problems_pct.columns[1:]:  # Skip 'month' column
        fig.add_trace(go.Bar(
            x=monthly_problems_pct['month'],
            y=monthly_problems_pct[problem],
            name=problem,
            marker_color=color_map.get(problem, '#888888'),
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',  # This is the key - ensures bars stack
        yaxis=dict(
            tickformat='.0f%%',
            range=[0, 100],
            title=''
        ),
        xaxis=dict(
            title='Month',
            tickformat='%b %Y'
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Problem Category"
        ),
        bargap=0.15
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_positive_categories_chart(df: pd.DataFrame):
    """Create stacked bar chart showing positive categories over time"""
    if 'date' not in df.columns or 'positive_mentions' not in df.columns:
        st.info("No date and positive mention data available for chart.")
        return
    
    st.markdown("#### Positive Categories (% of Monthly Total)")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_filtered = df.dropna(subset=['date'])
    
    if df_filtered.empty:
        st.info("No data available for positive categories.")
        return
    
    # Parse positive categories
    positive_rows = []
    for _, row in df_filtered.iterrows():
        if pd.notna(row['positive_mentions']) and row['positive_mentions']:
            positives = str(row['positive_mentions']).split(';')
            for positive in positives:
                positive = positive.strip()
                if positive and positive != 'None':
                    positive_rows.append({
                        'month': row['date'].to_period('M'),
                        'positive': positive
                    })
    
    if not positive_rows:
        st.info("No positive categories found in the data.")
        return
    
    positives_df = pd.DataFrame(positive_rows)
    
    # Count positives by month
    monthly_positives = positives_df.groupby(['month', 'positive']).size().unstack(fill_value=0)
    
    # Calculate percentages (multiply by 100 for percentage values)
    monthly_positives_pct = monthly_positives.div(monthly_positives.sum(axis=1), axis=0) * 100
    
    # Reset index for plotting
    monthly_positives_pct = monthly_positives_pct.reset_index()
    monthly_positives_pct['month'] = monthly_positives_pct['month'].dt.to_timestamp()
    
    # Melt for stacked bar chart
    plot_data = monthly_positives_pct.melt(id_vars='month', var_name='Positive Category', value_name='Percentage')
    
    # Define specific colors for common positive categories
    color_map = {
        'Comfort': '#2ca02c',     # Green
        'Material': '#1f77b4',    # Blue
        'Fit': '#ff7f0e',         # Orange
        'Design': '#9467bd',      # Purple
        'Durability': '#8c564b',  # Brown
        'Color': '#e377c2',       # Pink
        'Brand': '#7f7f7f',       # Gray
        'Price': '#bcbd22',       # Olive
        'Breathability': '#17becf', # Cyan
        'Packaging': '#aec7e8',   # Light blue
        'Shipping': '#ffbb78',    # Light orange
        'Other': '#98df8a',       # Light green
        'Style': '#d62728',       # Red
        'Quality': '#ff9896'      # Light red
    }
    
    # Add any missing categories with default colors
    unique_positives = plot_data['Positive Category'].unique()
    default_colors = px.colors.qualitative.Set3
    for i, positive in enumerate(unique_positives):
        if positive not in color_map:
            color_map[positive] = default_colors[i % len(default_colors)]
    
    # Create stacked bar chart using go.Bar for proper stacking
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add a trace for each positive category
    for positive in monthly_positives_pct.columns[1:]:  # Skip 'month' column
        fig.add_trace(go.Bar(
            x=monthly_positives_pct['month'],
            y=monthly_positives_pct[positive],
            name=positive,
            marker_color=color_map.get(positive, '#888888'),
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',  # This is the key - ensures bars stack
        yaxis=dict(
            tickformat='.0f%%',
            range=[0, 100],
            title=''
        ),
        xaxis=dict(
            title='Month',
            tickformat='%b %Y'
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Positive Category"
        ),
        bargap=0.15
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_results(df: pd.DataFrame):
    """Display detailed results table"""
    st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
    
    # Column selection - prefer product_description over product
    available_cols = df.columns.tolist()
    if 'product_description' in available_cols:
        default_cols = ['date', 'product_description', 'rating', 'sentiment', 'problems_mentioned', 'positive_mentions']
    else:
        default_cols = ['date', 'product', 'rating', 'sentiment', 'problems_mentioned', 'positive_mentions']
    display_cols = [col for col in default_cols if col in available_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display:",
        options=available_cols,
        default=display_cols
    )
    
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True, height=400)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv,
            f"filtered_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if 'original_text' in df.columns:
            text_export = "\n\n".join([
                f"Review {i+1}:\nSentiment: {row['sentiment']}\nRating: {row.get('rating', 'N/A')}\nText: {row['original_text']}"
                for i, row in df.iterrows()
            ])
            st.download_button(
                "📝 Export Review Texts",
                text_export,
                f"review_texts_{datetime.now():%Y%m%d_%H%M%S}.txt",
                "text/plain",
                use_container_width=True
            )

def create_review_samples(df: pd.DataFrame):
    """Display random sample reviews in a formatted layout"""
    st.markdown("#### Review Samples")
    
    # Check if we have the necessary columns
    if 'original_text' not in df.columns:
        st.info("No review text data available for display.")
        return
    
    # Filter to only reviews with actual text content
    # Convert to string first to handle any non-string values, then check for content
    df['original_text_str'] = df['original_text'].fillna('').astype(str)
    df_with_text = df[
        (df['original_text_str'].str.strip() != '') & 
        (df['original_text_str'] != 'nan') &
        (df['original_text_str'].str.len() >= 3)  # Only show reviews with at least 3 characters
    ].copy()
    
    if len(df_with_text) == 0:
        st.warning("No reviews with text content found in the filtered data.")
        return
    
    # Create controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Slider for number of reviews
        max_available = min(25, len(df_with_text))
        num_reviews = st.slider(
            "Reviews to display:",
            min_value=1,
            max_value=max_available,
            value=min(5, max_available),
            key="sample_size_slider"
        )
    
    with col2:
        # Random seed control for reproducibility
        if st.button("🔄 Refresh Sample", key="refresh_sample"):
            # This will trigger a rerun and get new random samples
            pass
    
    with col3:
        st.info(f"Showing {num_reviews} random reviews from {len(df_with_text):,} reviews with text (out of {len(df):,} filtered)")
    
    # Get random sample from reviews with text
    if len(df_with_text) > 0:
        sample_df = df_with_text.sample(n=min(num_reviews, len(df_with_text)))
        
        # Display each review
        for _, row in sample_df.iterrows():
            # Create a container for each review
            with st.container():
                # Header row with date, product, and rating
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    # Display date
                    date_val = row.get('date')
                    if date_val is not None and pd.notna(date_val):
                        try:
                            date_str = pd.to_datetime(date_val).strftime('%m/%d/%Y')
                        except:
                            date_str = str(date_val)
                        st.caption(date_str)
                    else:
                        st.caption("No date")
                
                with col2:
                    # Display product description if available, otherwise product SKU
                    product_name = row.get('product_description', row.get('product', 'Unknown Product'))
                    st.markdown(f"**{product_name}**")
                
                with col3:
                    # Display rating as stars
                    rating_val = row.get('rating')
                    if rating_val is not None and pd.notna(rating_val):
                        try:
                            rating = int(float(rating_val))
                            stars = '⭐' * rating + '☆' * (5 - rating)
                            st.markdown(f"**{stars}**")
                        except:
                            st.markdown("**Rating: N/A**")
                    else:
                        st.markdown("**Rating: N/A**")
                
                # Review text (we know it exists since we filtered for it)
                review_text = str(row.get('original_text', ''))
                # Truncate very long reviews
                if len(review_text) > 500:
                    review_text = review_text[:500] + "..."
                st.write(review_text)
                
                # Add sentiment and problem indicators if available
                indicators = []
                sentiment_val = row.get('sentiment')
                if sentiment_val is not None and pd.notna(sentiment_val):
                    if sentiment_val == 'Positive':
                        indicators.append("😊 Positive")
                    elif sentiment_val == 'Negative':
                        indicators.append("😔 Negative")
                    else:
                        indicators.append("😐 Neutral")
                
                problems_val = row.get('problems_mentioned')
                if problems_val is not None and pd.notna(problems_val) and problems_val != 'None':
                    indicators.append(f"⚠️ Issues: {problems_val}")
                
                positives_val = row.get('positive_mentions')
                if positives_val is not None and pd.notna(positives_val) and positives_val != 'None':
                    indicators.append(f"✅ Positives: {positives_val}")
                
                if indicators:
                    st.caption(" | ".join(indicators))
                
                # Add separator
                st.divider()
    else:
        st.warning("No reviews available to sample.")

def create_wordcloud_visualization(df: pd.DataFrame):
    """Create word cloud visualization from review texts"""
    if not WORDCLOUD_AVAILABLE:
        st.error("WordCloud library not available. Please install it first.")
        return
    
    st.markdown("#### Word Cloud Analysis")
    
    # Check if we have text data
    if 'original_text' not in df.columns:
        st.info("No review text data available for word cloud generation.")
        return
    
    # Combine all review texts
    all_text = ' '.join(df['original_text'].dropna().astype(str))
    
    if not all_text.strip():
        st.warning("No text content found in reviews.")
        return
    
    # Word cloud configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        colormap = st.selectbox(
            "Color Scheme:",
            ["viridis", "plasma", "coolwarm", "RdYlBu", "Spectral", "Set3"],
            index=0
        )
    
    with col2:
        max_words = st.slider(
            "Max Words:",
            min_value=50,
            max_value=200,
            value=100,
            step=10
        )
    
    with col3:
        background_color = st.selectbox(
            "Background:",
            ["white", "black", "lightgray"],
            index=0
        )
    
    # Common stopwords to filter out
    stopwords = set([
        'the', 'not', 'a', 'an', 'am', 'as', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'it', 'they',
        'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'my', 'your', 'his', 'her', 'its', 'our', 'we', 'us', 'me', 'him', 'she'
    ])
    
    try:
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10,
            stopwords=stopwords,
            contour_width=1,
            contour_color='steelblue'
        ).generate(all_text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Most Frequent Terms in {len(df):,} Reviews", fontsize=16, pad=20)
        
        # Display the word cloud
        st.pyplot(fig)
        
        # Show word frequency statistics
        with st.expander("📊 Word Frequency Statistics"):
            # Extract word frequencies
            word_freq = wordcloud.words_
            
            # Convert to DataFrame for display
            freq_df = pd.DataFrame(
                list(word_freq.items()),
                columns=['Word', 'Frequency']
            ).sort_values('Frequency', ascending=False).head(20)
            
            # Display as bar chart
            fig_bar = px.bar(
                freq_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title='Top 20 Most Frequent Words',
                color='Frequency',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")

def fetch_available_models(api_url=None):
    """Fetch available models from LM Studio - consistent with analyze app"""
    # Get timeout from environment variable
    timeout = int(os.getenv("LLM_MODEL_FETCH_TIMEOUT", "10"))
    
    # If an api_url is provided, extract the base URL from it
    if api_url:
        # Remove the /v1/chat/completions part if present
        if '/v1/chat/completions' in api_url:
            base_url = api_url.replace('/v1/chat/completions', '')
        elif '/v1' in api_url:
            base_url = api_url.rsplit('/v1', 1)[0]
        else:
            base_url = api_url
    else:
        # Fall back to environment variable
        base_url = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
    
    try:
        # Try to fetch models from the /v1/models endpoint
        models_url = f"{base_url}/v1/models"
        response = requests.get(models_url, timeout=timeout)
        if response.status_code == 200:
            models_data = response.json()
            if 'data' in models_data:
                return [model['id'] for model in models_data['data']]
    except requests.exceptions.RequestException as e:
        # Optionally log the error for debugging
        print(f"Error fetching models: {e}")
    return []

def compute_ai_metrics(df: pd.DataFrame, granularity: str = 'week', aggregation_level: str = 'product') -> Dict[str, Any]:
    """Compute aggregated metrics for AI analysis"""
    
    # Convert date to datetime and create period column
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if granularity == 'week':
        df['period'] = df['date'].dt.to_period('W').dt.start_time
    else:  # month
        df['period'] = df['date'].dt.to_period('M').dt.start_time
    
    # Determine the aggregation column
    if aggregation_level != 'product':
        # Use the specified aggregation level
        agg_column = aggregation_level
        # Check if column exists
        if agg_column not in df.columns:
            return {'error': f"Aggregation column '{agg_column}' not found in data"}
    else:
        # Use product column
        agg_column = 'product'
    
    # Filter out rows with invalid dates
    df = df.dropna(subset=['date', 'period'])
    
    if df.empty:
        return {'error': 'No valid data after date processing'}
    
    # Create sentiment binary columns
    df['is_positive'] = df['sentiment'] == 'Positive'
    df['is_negative'] = df['sentiment'] == 'Negative'
    df['is_neutral'] = df['sentiment'] == 'Neutral'
    
    # Group by period and aggregation level
    group_cols = ['period', agg_column]
    if aggregation_level == 'product' and 'product_description' in df.columns:
        group_cols.append('product_description')
    
    # Basic aggregations
    agg_dict = {
        'sentiment': 'size',  # count reviews
        'rating': 'mean',
        'is_positive': 'mean',
        'is_negative': 'mean',
        'is_neutral': 'mean'
    }
    
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    # Use business-friendly column names
    agg_df.columns = ['period', 'group_key'] + (['product_description'] if 'product_description' in group_cols else []) + \
                      ['n_reviews', 'avg_rating', 'positive_feedback_rate', 'negative_feedback_rate', 'neutral_feedback_rate']
    
    # Keep original names for backward compatibility in calculations
    agg_df['p_pos'] = agg_df['positive_feedback_rate']
    agg_df['p_neg'] = agg_df['negative_feedback_rate']
    agg_df['p_neu'] = agg_df['neutral_feedback_rate']
    
    # Process problem and positive mentions
    problem_counts = []
    positive_counts = []
    
    for (period, group_key), group in df.groupby(['period', agg_column]):
        # Count problems
        problems = {}
        for prob_str in group['problems_mentioned'].dropna():
            if prob_str and prob_str != 'None':
                for p in str(prob_str).split(';'):
                    p = p.strip()
                    if p and p != 'None':
                        problems[p] = problems.get(p, 0) + 1
        
        # Count positives
        positives = {}
        for pos_str in group['positive_mentions'].dropna():
            if pos_str and pos_str != 'None':
                for p in str(pos_str).split(';'):
                    p = p.strip()
                    if p and p != 'None':
                        positives[p] = positives.get(p, 0) + 1
        
        problem_counts.append({'period': period, 'group_key': group_key, 'problem_counts': problems})
        positive_counts.append({'period': period, 'group_key': group_key, 'positive_counts': positives})
    
    # Merge problem and positive counts
    problem_df = pd.DataFrame(problem_counts)
    positive_df = pd.DataFrame(positive_counts)
    
    agg_df = agg_df.merge(problem_df, on=['period', 'group_key'], how='left')
    agg_df = agg_df.merge(positive_df, on=['period', 'group_key'], how='left')
    
    # Fill missing counts with empty dicts
    agg_df['problem_counts'] = agg_df['problem_counts'].fillna({}).apply(lambda x: x if x else {})
    agg_df['positive_counts'] = agg_df['positive_counts'].fillna({}).apply(lambda x: x if x else {})
    
    # Calculate rolling baselines and z-scores
    agg_df = agg_df.sort_values(['group_key', 'period'])
    
    # Rolling statistics for negative sentiment
    agg_df['neg_roll_mean'] = agg_df.groupby('group_key')['p_neg'].transform(
        lambda s: s.rolling(8, min_periods=4, center=False).mean()
    )
    agg_df['neg_roll_std'] = agg_df.groupby('group_key')['p_neg'].transform(
        lambda s: s.rolling(8, min_periods=4, center=False).std()
    )
    
    # Calculate z-score for negative sentiment
    agg_df['z_neg'] = 0.0  # Initialize
    mask = (agg_df['neg_roll_std'] > 0) & agg_df['neg_roll_mean'].notna()
    agg_df.loc[mask, 'z_neg'] = (
        (agg_df.loc[mask, 'p_neg'] - agg_df.loc[mask, 'neg_roll_mean']) / 
        agg_df.loc[mask, 'neg_roll_std']
    )
    
    # Calculate deltas vs baseline
    agg_df['rating_delta'] = agg_df.groupby('group_key')['avg_rating'].transform(
        lambda s: s - s.rolling(8, min_periods=4, center=False).mean()
    )
    agg_df['neg_delta_pct'] = agg_df.groupby('group_key')['p_neg'].transform(
        lambda s: ((s / s.rolling(8, min_periods=4, center=False).mean()) - 1) * 100
    )
    
    # Identify anomalies (|z| >= 2 or significant change)
    agg_df['anomaly_flags'] = agg_df.apply(
        lambda row: ['p_neg'] if abs(row.get('z_neg', 0)) >= 2 else [],
        axis=1
    )
    
    return {
        'aggregated_data': agg_df,
        'granularity': granularity,
        'aggregation_level': aggregation_level,
        'total_reviews': len(df),
        'groups_analyzed': df[agg_column].nunique(),
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    }

def prepare_llm_payload(metrics: Dict[str, Any], top_n_groups: int = 20) -> Dict[str, Any]:
    """Prepare the compact JSON payload for LLM analysis"""
    
    agg_df = metrics['aggregated_data']
    aggregation_level = metrics.get('aggregation_level', 'product')
    
    # Get top groups by review volume
    top_groups = (
        agg_df.groupby('group_key')['n_reviews']
        .sum()
        .nlargest(top_n_groups)
        .index.tolist()
    )
    
    # Filter to top groups and recent periods
    recent_df = agg_df[agg_df['group_key'].isin(top_groups)].copy()
    recent_df = recent_df.sort_values('period').tail(200)  # Last 200 data points
    
    # Build series data for JSON
    series = []
    for _, row in recent_df.iterrows():
        series_item = {
            'period': row['period'].strftime('%Y-%m-%d'),
            'group': row['group_key'],
            'n_reviews': int(row['n_reviews']),
            'customer_rating': round(float(row['avg_rating']), 2) if pd.notna(row['avg_rating']) else None,
            'positive_feedback_pct': round(float(row['positive_feedback_rate']) * 100, 1) if pd.notna(row['positive_feedback_rate']) else 0,
            'neutral_feedback_pct': round(float(row['neutral_feedback_rate']) * 100, 1) if pd.notna(row['neutral_feedback_rate']) else 0,
            'negative_feedback_pct': round(float(row['negative_feedback_rate']) * 100, 1) if pd.notna(row['negative_feedback_rate']) else 0,
            'problem_counts': dict(row.get('problem_counts', {})),
            'positive_counts': dict(row.get('positive_counts', {})),
            'deltas': {
                'rating_delta': round(float(row.get('rating_delta', 0)), 2) if pd.notna(row.get('rating_delta')) else 0,
                'neg_delta_pct': round(float(row.get('neg_delta_pct', 0)), 1) if pd.notna(row.get('neg_delta_pct')) else 0
            },
            'statistical_significance': {
                'negative_feedback_unusual': abs(float(row.get('z_neg', 0))) >= 2.0 if pd.notna(row.get('z_neg')) else False,
                'significance_level': 'high' if abs(float(row.get('z_neg', 0))) >= 3.0 else 'moderate' if abs(float(row.get('z_neg', 0))) >= 2.0 else 'normal'
            },
            'anomaly_flags': row.get('anomaly_flags', [])
        }
        series.append(series_item)
    
    # Build complete payload
    payload = {
        'meta': {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'period': metrics.get('date_range', 'Unknown'),
            'granularity': metrics.get('granularity', 'week'),
            'aggregation_level': aggregation_level,
            'total_reviews': metrics.get('total_reviews', 0),
            'groups_analyzed': len(top_groups),
            'anomaly_rules': {
                'z_abs_threshold': 2.0,
                'min_count': 10  # Lowered threshold for aggregated data
            }
        },
        'series': series
    }
    
    return payload

def create_ai_analysis_prompt(payload: Dict[str, Any]) -> tuple[str, str]:
    """Create the structured prompt for LLM analysis"""
    
    system_prompt = """You are a business intelligence assistant helping retail executives understand customer feedback trends.
Write in clear, non-technical business language. Avoid statistical jargon like z-scores, p-values, or standard deviations.
Focus on what matters to business leaders: customer satisfaction changes, emerging issues, and actionable insights.
Output ONLY the requested JSON structure followed by a business executive summary."""
    
    aggregation_level = payload.get('meta', {}).get('aggregation_level', 'product')
    min_count = payload.get('meta', {}).get('anomaly_rules', {}).get('min_count', 10)
    
    user_prompt = f"""Analyze customer feedback trends and provide business insights.

Data: {json.dumps(payload, indent=2)}

Analysis Guidelines:
- Look for significant changes in customer satisfaction (threshold: {min_count}+ reviews with notable shifts)
- Identify patterns that persist for 2+ time periods
- Translate data into business language (avoid technical statistics)
- Focus on actionable insights for business leaders

Create a JSON analysis with these insights:

{{
  "brand_trends": [
    {{"theme":"<business trend like 'Customer satisfaction improving'>","direction":"improving/declining","evidence":[{{"period":"YYYY-MM-DD","metric":"<business metric>","value":"<value with context>"}}]}}
  ],
  "group_highlights": [
    {{"group":"<{aggregation_level}>", "issue":"<business-friendly description like 'Sharp drop in satisfaction'>", "change":"<e.g., '15% decrease'>", "periods":["YYYY-MM-DD"]}}
  ],
  "emerging_topics": [
    {{"label":"<business topic>","groups":["<{aggregation_level}>"],"trend":"<what's happening in business terms>"}}
  ],
  "risk_watchlist": [
    {{"group":"<{aggregation_level}>", "reason":"<business risk explanation>", "action":"<business recommendation>"}}
  ],
  "positive_drivers": [
    {{"theme":"<what's working well>","groups":["<{aggregation_level}>"],"evidence":"<business evidence>"}}
  ],
  "negative_patterns": [
    {{"theme":"<business problem>","groups":["<{aggregation_level}>"],"evidence":"<business impact>","severity":"high/medium/low"}}
  ]
}}

After the JSON, write an 8-line executive summary in business language that a CEO would understand."""
    
    return system_prompt, user_prompt

def call_llm_api(system_prompt: str, user_prompt: str, model_config: Dict[str, str]) -> str:
    """Call the LLM API (LM Studio or other providers) - using same pattern as analyze app"""
    
    # Get configuration from model_config
    provider = model_config.get('provider', 'LM Studio (Local)')
    api_url = model_config.get('api_url', '')
    api_key = model_config.get('api_key', 'not-needed')
    model_id = model_config.get('model_id', 'gemma-2-9b-it')
    temperature = float(model_config.get('temperature', 0.1))
    
    # Handle provider-specific defaults if API URL is empty
    if not api_url:
        if provider == "OpenAI":
            api_url = "https://api.openai.com/v1/chat/completions"
        else:
            # LM Studio or custom provider
            base_url = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
            api_url = f'{base_url}/v1/chat/completions'
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    # Only add Authorization header if API key is provided and not 'not-needed'
    if api_key and api_key != 'not-needed':
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Use different parameter name for OpenAI vs other providers
    max_tokens_param = 'max_completion_tokens' if provider == "OpenAI" else 'max_tokens'
    
    # Set a high token limit to avoid truncation
    max_tokens = 50000  # High limit to ensure complete response
    
    # Build the request data
    data = {
        'model': model_id,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        max_tokens_param: max_tokens
    }
    
    # For OpenAI, only add temperature if it's not 1.0 (some models only support default)
    # For other providers, always include temperature
    if provider == "OpenAI":
        if temperature != 1.0:
            # Only add for models that support it
            if not model_id.startswith('o1'):  # o1 models don't support temperature
                data['temperature'] = temperature
    else:
        data['temperature'] = temperature
    
    # Get timeout from environment variable
    timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        # Debug logging for empty responses
        if not result.get('choices'):
            return f"Error: No choices in response. Full response: {json.dumps(result)}"
        
        if not result['choices'][0].get('message'):
            return f"Error: No message in response. Full response: {json.dumps(result)}"
        
        content = result['choices'][0]['message'].get('content', '')
        if not content:
            # Check if there's a refusal or other issue
            if result['choices'][0].get('finish_reason') == 'content_filter':
                return "Error: Content was filtered by OpenAI's safety system"
            elif result['choices'][0].get('finish_reason') == 'length':
                return "Error: Response was truncated due to length"
            else:
                return f"Error: Empty response from model. Finish reason: {result['choices'][0].get('finish_reason', 'unknown')}"
        
        return content
    except requests.exceptions.HTTPError as e:
        # Enhanced error handling for common issues
        if e.response.status_code == 404:
            if provider == "OpenAI":
                try:
                    error_detail = e.response.json().get('error', {}).get('message', '')
                    if 'model' in error_detail.lower() or 'does not exist' in error_detail.lower():
                        return f"OpenAI Error: Model '{model_id}' not found. Try 'gpt-4', 'gpt-4-turbo-preview', or 'gpt-3.5-turbo'"
                    else:
                        return f"OpenAI Error (404): {error_detail or 'Invalid endpoint or model. Check your model name.'}"
                except:
                    return f"OpenAI Error: Model '{model_id}' not found. Valid models: gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo"
            else:
                return f"LLM API request failed (404): {str(e)}"
        elif e.response.status_code == 401:
            return f"Authentication failed: Invalid API key. Please check your API key."
        elif e.response.status_code == 429:
            try:
                error_detail = e.response.json().get('error', {})
                retry_after = e.response.headers.get('Retry-After', 'a few seconds')
                return f"OpenAI Rate Limit: {error_detail.get('message', 'Too many requests')}. Wait {retry_after} before retrying. Consider using a paid API key or local models."
            except:
                return f"Rate limit exceeded: Too many requests. Wait 20-60 seconds before retrying. Consider using a paid OpenAI API key with higher limits."
        elif e.response.status_code == 400:
            try:
                error_msg = e.response.json().get('error', {}).get('message', str(e))
            except:
                error_msg = str(e)
            return f"Bad request: {error_msg}"
        else:
            return f"LLM API request failed: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"Failed to connect to LLM provider: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {str(e)}"

def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Finds and extracts the first JSON object string from a text block.
    This is the same robust helper used in process_reviews.py
    """
    try:
        # Find the first opening curly brace
        start_index = text.find('{')
        if start_index == -1:
            return None
        
        # Find the last closing curly brace
        end_index = text.rfind('}')
        if end_index == -1 or end_index < start_index:
            return None
        
        # Return the substring that looks like a JSON object
        return text[start_index:end_index + 1]
    except Exception:
        return None

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response to extract JSON and narrative"""
    
    # Handle error responses
    if response.startswith("Error") or response.startswith("Failed") or response.startswith("LLM API"):
        return {
            'success': False,
            'error': True,
            'analysis': {},
            'narrative': response,
            'raw_response': response
        }
    
    # Check if response looks truncated (no closing brace at end)
    response_stripped = response.strip()
    if response_stripped and not (response_stripped.endswith('}') or response_stripped.endswith('.')):
        # Response may be truncated
        return {
            'success': False,
            'error': True,
            'message': 'Response appears truncated. Increase max_tokens or reduce data size.',
            'analysis': {},
            'narrative': 'LLM response was cut off before completing the analysis.',
            'raw_response': response
        }
    
    # Try multiple methods to extract JSON
    json_str = None
    narrative = ""
    
    # Method 1: Look for ```json code blocks
    if '```json' in response.lower():
        start = response.lower().find('```json') + 7
        end = response.find('```', start)
        if end > start:
            json_str = response[start:end].strip()
            narrative = response[end+3:].strip()
    
    # Method 2: Look for ``` code blocks (without json marker)
    if not json_str and '```' in response:
        parts = response.split('```')
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd indices are code blocks
                test_str = part.strip()
                if test_str.startswith('{'):
                    json_str = test_str
                    # Get narrative from remaining parts
                    narrative = ''.join(parts[i+1:]).strip()
                    break
    
    # Method 3: Use the robust helper function
    if not json_str:
        json_str = _extract_json_from_text(response)
        if json_str:
            # Find narrative after the JSON
            json_end = response.rfind('}')
            if json_end != -1:
                narrative = response[json_end + 1:].strip()
    
    # Try to parse the JSON
    if json_str:
        try:
            # Clean up common issues
            json_str = json_str.strip()
            # Remove any markdown formatting
            if json_str.startswith('`'):
                json_str = json_str.strip('`')
            if json_str.startswith('json'):
                json_str = json_str[4:].strip()
            
            # Parse the JSON
            analysis_json = json.loads(json_str)
            
            # Extract narrative if not already found
            if not narrative:
                # Look for narrative after JSON in original response
                json_in_response = response.find(json_str)
                if json_in_response != -1:
                    narrative = response[json_in_response + len(json_str):].strip()
            
            return {
                'success': True,
                'analysis': analysis_json,
                'narrative': narrative,
                'raw_response': response,
                'extracted_json': json_str  # For debugging
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': True,
                'parse_error': str(e),
                'analysis': {},
                'narrative': response,
                'raw_response': response,
                'attempted_json': json_str  # For debugging
            }
    
    # Fallback if no JSON found
    return {
        'success': False,
        'error': False,
        'no_json_found': True,
        'analysis': {},
        'narrative': response,
        'raw_response': response
    }

def build_chat_context(df: pd.DataFrame) -> str:
    """Build context about the current data for the chat"""
    context_parts = []
    
    # Basic statistics
    context_parts.append(f"Dataset Overview:")
    context_parts.append(f"- Total reviews: {len(df):,}")
    context_parts.append(f"- Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    context_parts.append(f"- Average rating: {df['rating'].mean():.2f}")
    
    # Product information
    unique_products = df['product'].nunique()
    context_parts.append(f"- Unique products: {unique_products}")
    
    # Top products by review count
    top_products = df['product'].value_counts().head(10)
    context_parts.append("\nTop 10 products by review count:")
    for product, count in top_products.items():
        avg_rating = df[df['product'] == product]['rating'].mean()
        context_parts.append(f"  - {product}: {count} reviews, {avg_rating:.2f} avg rating")
    
    # Style information if available
    if 'STYLE_CODE' in df.columns:
        unique_styles = df['STYLE_CODE'].nunique()
        context_parts.append(f"\n- Unique styles: {unique_styles}")
        
        # Top styles by review count
        top_styles = df['STYLE_CODE'].value_counts().head(10)
        context_parts.append("\nTop 10 styles by review count:")
        for style, count in top_styles.items():
            if pd.notna(style):
                avg_rating = df[df['STYLE_CODE'] == style]['rating'].mean()
                sentiment_dist = df[df['STYLE_CODE'] == style]['sentiment'].value_counts()
                context_parts.append(f"  - {style}: {count} reviews, {avg_rating:.2f} avg rating")
                if not sentiment_dist.empty:
                    context_parts.append(f"    Sentiment: {dict(sentiment_dist)}")
    
    # Problem categories summary
    if 'problems_mentioned' in df.columns:
        all_problems = []
        for problems in df['problems_mentioned'].dropna():
            if isinstance(problems, str):
                try:
                    problems_list = eval(problems) if problems.startswith('[') else [problems]
                    all_problems.extend(problems_list)
                except:
                    pass
        if all_problems:
            problem_counts = pd.Series(all_problems).value_counts().head(10)
            context_parts.append("\nTop 10 problems mentioned:")
            for problem, count in problem_counts.items():
                context_parts.append(f"  - {problem}: {count} mentions")
    
    # Positive mentions summary
    if 'positive_mentions' in df.columns:
        all_positives = []
        for positives in df['positive_mentions'].dropna():
            if isinstance(positives, str):
                try:
                    positives_list = eval(positives) if positives.startswith('[') else [positives]
                    all_positives.extend(positives_list)
                except:
                    pass
        if all_positives:
            positive_counts = pd.Series(all_positives).value_counts().head(10)
            context_parts.append("\nTop 10 positive aspects mentioned:")
            for positive, count in positive_counts.items():
                context_parts.append(f"  - {positive}: {count} mentions")
    
    return "\n".join(context_parts)

def handle_chat_query(query: str, df: pd.DataFrame, model_config: Dict[str, str]) -> str:
    """Handle a chat query about the data"""
    
    # Build context about the current data
    data_context = build_chat_context(df)
    
    # Create system prompt for chat
    system_prompt = f"""You are an AI assistant specialized in analyzing product review data. 
You have access to the following information about the current dataset:

{data_context}

Based on this data, provide insightful, data-driven answers to questions about:
- Product performance and ratings
- Style strengths and weaknesses
- Customer sentiment and feedback patterns
- Problem areas and positive aspects
- Trends and comparisons between products/styles

Always base your answers on the actual data provided. If asked about something not in the data, 
acknowledge the limitation. Provide specific examples and numbers when relevant."""

    # User's query
    user_prompt = query
    
    # Call the LLM
    try:
        response = call_llm_api(system_prompt, user_prompt, model_config)
        return response
    except Exception as e:
        return f"Error processing your query: {str(e)}"

def create_ai_analysis_tab(df: pd.DataFrame):
    """Create the AI-powered analysis tab with chat functionality"""
    
    st.markdown("#### 🤖 AI-Powered Analysis")
    
    # Identify available aggregation columns
    available_agg_columns = []
    column_display_names = {}
    
    # ONLY the 5 aggregation columns requested by user
    potential_columns = {
        'GENDER_TEXT': 'Gender',
        'PRODUCT_CLASS_TEXT': 'Product Class',
        'PRODUCT_SUB_CLASS_TEXT': 'Product Sub-Class',
        'END_USE_TEXT': 'End Use',
        'STYLE_CODE': 'Style Code'
    }
    
    # Check which columns actually exist in the dataframe
    for col_name, display_name in potential_columns.items():
        if col_name in df.columns:
            # Only add if column has valid data
            if df[col_name].notna().any():
                available_agg_columns.append(col_name)
                column_display_names[col_name] = display_name
    
    # Always include product-level as an option
    available_agg_columns.append('product')
    column_display_names['product'] = 'Individual Product'
    
    # Configuration section
    with st.expander("⚙️ AI Analysis Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Aggregation level selector
            aggregation_level = st.selectbox(
                "Aggregation Level:",
                available_agg_columns,
                format_func=lambda x: column_display_names.get(x, x),
                index=0 if available_agg_columns else None,
                help="Choose how to group products for analysis"
            )
            
            granularity = st.selectbox(
                "Time Granularity:",
                ["week", "month"],
                index=0,
                help="Choose the time period for aggregation"
            )
        
        with col2:
            top_n = st.slider(
                f"Top N {column_display_names.get(aggregation_level, 'Groups')}:",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help=f"Number of top {column_display_names.get(aggregation_level, 'groups').lower()} to analyze"
            )
            
            min_reviews = st.number_input(
                "Min Reviews per Period:",
                min_value=3,
                max_value=100,
                value=15,
                step=5,
                help="Minimum reviews required for statistical significance"
            )
    
    # LLM Configuration Status (Read-only from .env)
    st.markdown("##### 🔧 LLM Configuration Status")
    
    # Get configuration from environment variables (AI-specific, with fallback to generic)
    provider = os.getenv("AI_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "LM Studio (Local)"))
    api_url = os.getenv("AI_LLM_API_URL", os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions"))
    api_key = os.getenv("AI_LLM_API_KEY", os.getenv("LLM_API_KEY", "not-needed"))
    model_id = os.getenv("AI_LLM_MODEL_ID", os.getenv("LLM_MODEL_ID", "gemma-2-9b-it"))
    temperature = float(os.getenv("AI_LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.1")))
    
    # Check connection status
    connection_status = "🔴 Not Connected"
    available_models = []
    if provider == "LM Studio (Local)":
        available_models = fetch_available_models(api_url)
        if available_models:
            connection_status = f"✅ Connected ({len(available_models)} models available)"
        else:
            connection_status = "⚠️ Unable to connect to LLM server"
    
    # Display configuration in a clean info box
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Provider:** {provider}  
            **Model:** {model_id}  
            **Temperature:** {temperature}
            """)
        
        with col2:
            st.info(f"""
            **Status:** {connection_status}  
            **Server:** {api_url.split('/v1')[0] if '/v1' in api_url else api_url}  
            """)
        
        if not available_models and provider == "LM Studio (Local)":
            st.warning("""
            ⚠️ Cannot connect to LLM server. Please check:
            1. Your LLM server is running at the configured address
            2. The server URL in .env file is correct
            3. Your model is loaded in the server
            """)
        
        st.caption("💡 To change configuration, edit the `.env` file and restart the app")
    
    # Analysis button and results
    if st.button("🚀 Generate AI Analysis", type="primary"):
        # Store a flag to prevent duplicate API calls
        if 'ai_analysis_in_progress' not in st.session_state:
            st.session_state.ai_analysis_in_progress = False
        
        if st.session_state.ai_analysis_in_progress:
            st.warning("Analysis already in progress. Please wait...")
            return
        
        st.session_state.ai_analysis_in_progress = True
        
        with st.spinner("Computing metrics..."):
            # Compute metrics
            metrics = compute_ai_metrics(df, granularity, aggregation_level)
            
            if 'error' in metrics:
                st.error(metrics['error'])
                return
            
            st.success(f"✅ Computed metrics for {metrics.get('groups_analyzed', metrics.get('products_analyzed', 0))} groups across {metrics['total_reviews']} reviews")
        
        with st.spinner("Preparing LLM payload..."):
            # Prepare payload
            payload = prepare_llm_payload(metrics, top_n)
            
            # Show payload size
            payload_str = json.dumps(payload)
            token_estimate = len(payload_str) // 4  # Rough token estimate
            st.info(f"📦 Payload size: {len(payload_str):,} characters (~{token_estimate:,} tokens)")
        
        with st.spinner("Calling LLM for analysis..."):
            # Create prompt
            system_prompt, user_prompt = create_ai_analysis_prompt(payload)
            
            # Show debug info and check prompt size
            prompt_size = len(system_prompt) + len(user_prompt)
            
            # Warn if prompt is very large
            if prompt_size > 100000:  # ~25k tokens
                st.warning(f"⚠️ Large prompt detected ({prompt_size:,} chars). Consider reducing 'Top N Products' to avoid truncation.")
            
            if provider == "OpenAI":
                with st.expander("Debug: Request Info", expanded=False):
                    st.write(f"Model: {model_id}")
                    st.write(f"Temperature: {temperature}")
                    st.write(f"API URL: {api_url}")
                    st.write(f"Prompt size: {prompt_size:,} chars (~{prompt_size//4:,} tokens)")
            
            # Call LLM
            model_config = {
                'provider': provider,
                'api_url': api_url,
                'api_key': api_key,
                'model_id': model_id,
                'temperature': temperature
            }
            
            response = call_llm_api(system_prompt, user_prompt, model_config)
            
            # Parse response
            parsed = parse_llm_response(response)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 AI Analysis Results")
        
        if parsed['success']:
            # Display structured analysis
            analysis = parsed['analysis']
            
            # Create tabs for different insights
            tabs = st.tabs(["📈 Trends", "🎯 Highlights", "⚠️ Risks", "✨ Positives", "❌ Negatives", "📝 Narrative", "🔍 Raw Data"])
            
            with tabs[0]:  # Trends
                if 'brand_trends' in analysis and analysis['brand_trends']:
                    st.markdown("##### Key Market Trends")
                    for trend in analysis['brand_trends']:
                        # Get direction and theme
                        direction = trend.get('direction', 'Unknown')
                        theme = trend.get('theme', 'Unknown trend')
                        
                        # Use appropriate icon based on whether it's positive or negative
                        if 'improving' in direction.lower() or 'up' in direction.lower():
                            if 'satisfaction' in theme.lower() or 'rating' in theme.lower():
                                direction_icon = "📈"  # Good if satisfaction is up
                            else:
                                direction_icon = "⚠️"  # Bad if complaints/issues are up
                        else:
                            if 'satisfaction' in theme.lower() or 'rating' in theme.lower():
                                direction_icon = "📉"  # Bad if satisfaction is down
                            else:
                                direction_icon = "✅"  # Good if complaints/issues are down
                        
                        st.write(f"{direction_icon} **{theme}**")
                        
                        if 'evidence' in trend:
                            # Group evidence by metric type for cleaner display
                            for evidence in trend['evidence'][:3]:
                                period = evidence.get('period', '')
                                metric = evidence.get('metric', '')
                                value = evidence.get('value', '')
                                
                                # Format the evidence in business-friendly way
                                if period and metric and value:
                                    # Convert all technical metric names to business language
                                    if 'rating' in metric.lower() or 'customer_rating' in metric:
                                        display = f"Customer rating: {value}"
                                    elif 'negative_feedback' in metric or 'p_neg' in metric:
                                        # Convert to percentage if needed
                                        try:
                                            if float(value) <= 1:  # Decimal format
                                                pct = float(value) * 100
                                                display = f"Customers with complaints: {pct:.0f}%"
                                            else:  # Already percentage
                                                display = f"Customers with complaints: {value}%"
                                        except:
                                            display = f"Negative feedback rate: {value}"
                                    elif 'positive_feedback' in metric or 'p_pos' in metric:
                                        try:
                                            if float(value) <= 1:  # Decimal format
                                                pct = float(value) * 100
                                                display = f"Satisfied customers: {pct:.0f}%"
                                            else:  # Already percentage
                                                display = f"Satisfied customers: {value}%"
                                        except:
                                            display = f"Positive feedback rate: {value}"
                                    elif 'sentiment' in metric.lower():
                                        display = f"Customer sentiment: {value}"
                                    elif 'n_reviews' in metric:
                                        display = f"Number of reviews: {value}"
                                    else:
                                        # Clean up any remaining technical terms
                                        clean_metric = metric.replace('_', ' ').replace('pct', 'percent').replace('avg', 'average')
                                        display = f"{clean_metric.title()}: {value}"
                                    
                                    # Format date nicely
                                    try:
                                        from datetime import datetime
                                        date_obj = datetime.strptime(period, '%Y-%m-%d')
                                        formatted_date = date_obj.strftime('%B %d, %Y')
                                        st.caption(f"  • {formatted_date}: {display}")
                                    except:
                                        st.caption(f"  • {period}: {display}")
                else:
                    st.info("No significant brand trends detected")
                
                if 'emerging_topics' in analysis and analysis['emerging_topics']:
                    st.markdown("##### Emerging Topics")
                    for topic in analysis['emerging_topics']:
                        st.write(f"🆕 **{topic.get('label', 'Unknown')}**: {topic.get('trend', 'Unknown')}")
                        if 'products' in topic:
                            st.caption(f"  Products: {', '.join(topic['products'][:5])}")
            
            with tabs[1]:  # Highlights
                # Check for both 'group_highlights' (new) and 'product_highlights' (old)
                highlights = analysis.get('group_highlights', analysis.get('product_highlights', []))
                if highlights:
                    st.markdown("##### Key Insights")
                    for highlight in highlights:
                        # Handle both 'group' (new) and 'product' (old) fields
                        group_name = highlight.get('group', highlight.get('product', 'Unknown'))
                        issue = highlight.get('issue', 'Unknown')
                        change = highlight.get('change', highlight.get('delta_pct', 'N/A'))
                        
                        # Determine severity icon based on the description
                        if any(word in issue.lower() for word in ['sharp', 'significant', 'major', 'critical']):
                            severity = "🔴"
                        elif any(word in issue.lower() for word in ['moderate', 'notable', 'concerning']):
                            severity = "🟡"  
                        else:
                            severity = "🟢"
                        
                        st.write(f"{severity} **{group_name}**")
                        st.write(f"  {issue}")
                        if change and change != 'N/A':
                            st.caption(f"  Change: {change}")
                else:
                    st.info("No significant highlights detected")
            
            with tabs[2]:  # Risks
                if 'risk_watchlist' in analysis and analysis['risk_watchlist']:
                    st.markdown("##### Risk Watchlist")
                    for risk in analysis['risk_watchlist']:
                        # Handle both 'group' and 'product' fields
                        item_name = risk.get('group', risk.get('product', 'Unknown'))
                        st.write(f"⚠️ **{item_name}**")
                        st.write(f"  Reason: {risk.get('reason', 'No reason provided')}")
                        st.info(f"  💡 Action: {risk.get('action', 'Monitor closely')}")
                else:
                    st.success("No significant risks detected")
            
            with tabs[3]:  # Positives
                if 'positive_drivers' in analysis and analysis['positive_drivers']:
                    st.markdown("##### Positive Drivers")
                    for driver in analysis['positive_drivers']:
                        st.write(f"✅ **{driver.get('theme', 'Unknown')}**")
                        # Handle both 'groups' and 'products' fields
                        groups = driver.get('groups', driver.get('products', []))
                        if groups:
                            st.caption(f"  Groups: {', '.join(groups[:5])}")
                        st.caption(f"  Evidence: {driver.get('evidence', 'No evidence')}")
                else:
                    st.info("No significant positive drivers detected")
            
            with tabs[4]:  # Negatives
                if 'negative_patterns' in analysis and analysis['negative_patterns']:
                    st.markdown("##### Negative Patterns")
                    for pattern in analysis['negative_patterns']:
                        st.write(f"❌ **{pattern.get('theme', 'Unknown issue')}**")
                        # Handle both 'groups' and 'products' fields
                        groups = pattern.get('groups', pattern.get('products', []))
                        if groups:
                            st.caption(f"  Affected groups: {', '.join(groups[:5])}")
                        st.caption(f"  Evidence: {pattern.get('evidence', 'No evidence')}")
                        if 'severity' in pattern:
                            severity_color = "🔴" if pattern['severity'] == 'high' else "🟡" if pattern['severity'] == 'medium' else "🟢"
                            st.caption(f"  Severity: {severity_color} {pattern['severity']}")
                else:
                    st.info("No significant negative patterns detected")
            
            with tabs[5]:  # Narrative
                st.markdown("##### AI Narrative Summary")
                if parsed['narrative']:
                    st.write(parsed['narrative'])
                else:
                    st.info("No narrative provided")
            
            with tabs[6]:  # Raw Data
                st.markdown("##### Raw Analysis Data")
                
                # Show the metrics summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", f"{metrics['total_reviews']:,}")
                with col2:
                    st.metric("Groups Analyzed", metrics.get('groups_analyzed', metrics.get('products_analyzed', 0)))
                with col3:
                    st.metric("Date Range", metrics['date_range'])
                
                # Show the raw JSON analysis
                with st.expander("View Raw JSON Analysis"):
                    st.json(analysis)
                
                # Show aggregated data sample
                with st.expander("View Aggregated Metrics (First 50 rows)"):
                    st.dataframe(
                        metrics['aggregated_data'].head(50),
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ Could not parse structured analysis from LLM response")
            
            # Show debugging information
            if 'error' in parsed and parsed['error']:
                if 'message' in parsed:
                    st.error(f"**Issue**: {parsed['message']}")
                else:
                    st.error(f"**Error Type**: {parsed.get('narrative', 'Unknown error')}")
                
                if 'parse_error' in parsed:
                    st.error(f"**JSON Parse Error**: {parsed['parse_error']}")
                
                if 'attempted_json' in parsed:
                    st.markdown("##### Attempted to parse this JSON:")
                    st.code(parsed['attempted_json'], language='json')
            
            # Show response analysis
            raw_resp = parsed.get('raw_response', '')
            if raw_resp:
                st.caption(f"📊 Response size: {len(raw_resp)} characters")
                
                # Check for common issues
                if len(raw_resp) < 100:
                    st.warning("Response is very short. Check if the LLM is working correctly.")
                elif not ('{' in raw_resp and '}' in raw_resp):
                    st.warning("Response doesn't appear to contain JSON. The model may not understand the prompt.")
                elif raw_resp.count('{') != raw_resp.count('}'):
                    st.warning("JSON appears incomplete (mismatched braces). Try increasing max_tokens.")
            
            # Show raw response for debugging
            st.markdown("##### Raw LLM Response:")
            with st.expander("View Full Response", expanded=False):
                st.text_area("", value=parsed['raw_response'], height=400)
            
            # Helpful tips
            st.info("""
            **Troubleshooting Tips:**
            1. Check if LM Studio is running and the model is loaded
            2. Try increasing the temperature slightly (0.1-0.3)
            3. Ensure the model supports JSON output (Gemma, Llama, Mistral usually work well)
            4. Try a smaller data set (reduce "Top N Products")
            5. Check the console/terminal for any error messages
            """)
        
        # Reset the flag after analysis is complete
        st.session_state.ai_analysis_in_progress = False
    
    # Chat Section - Interactive Q&A about the data
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Data")
    st.info("Chat with AI about your review data. Ask about specific products, styles, trends, or comparisons.")
    
    # Initialize chat history in session state if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sample questions for quick selection
    sample_questions = {
        "📊 Style Analysis": "What are the strengths and weaknesses of the most reviewed styles?",
        "🔍 Product Comparison": "Compare the top 5 products by rating and sentiment",
        "📈 Trend Analysis": "What are the main trends in customer feedback over time?",
        "❌ Problem Areas": "What are the most common problems customers mention?",
        "✨ Top Performers": "Which products have the highest customer satisfaction and why?",
        "🎯 Recommendations": "What improvements would you recommend based on customer feedback?"
    }
    
    # Create container for sample questions
    with st.expander("💡 Sample Questions", expanded=False):
        st.markdown("Click any question below to use it:")
        for label, question in sample_questions.items():
            st.code(question, language=None)
    
    # Text input for user query
    user_query = st.text_area(
        "Ask a question about your data:",
        placeholder="e.g., 'What are the strengths and weaknesses of the W303 style?' or 'Compare the sentiment for different product classes'",
        height=100,
        key="chat_query_input"
    )
    
    # Submit button for the chat
    if st.button("🤖 Ask AI", type="primary", use_container_width=True):
        if user_query:
            # Show spinner while processing
            with st.spinner("Analyzing your question..."):
                # Prepare model configuration
                model_config = {
                    'provider': provider,
                    'api_url': api_url,
                    'api_key': api_key,
                    'model_id': model_id,
                    'temperature': temperature
                }
                
                # Get response from chat handler
                response = handle_chat_query(user_query, df, model_config)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_query,
                    'answer': response
                })
        else:
            st.warning("Please enter a question to ask about the data.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 📜 Chat History")
        
        # Add clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display conversation history in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:100]}...", expanded=(i==0)):
                st.markdown("**Question:**")
                st.write(chat['question'])
                st.markdown("**Answer:**")
                st.write(chat['answer'])
    
    # Help section
    with st.expander("ℹ️ About AI Analysis"):
        st.markdown("""
        This AI-powered analysis feature:
        
        1. **Aggregates your review data** into time periods (weekly/monthly)
        2. **Computes statistical metrics** including z-scores and anomaly detection
        3. **Sends a compact summary** to an LLM for intelligent analysis
        4. **Returns structured insights** about trends, risks, and opportunities
        
        The analysis is based on the filtered data currently displayed in your dashboard.
        
        **Tips for best results:**
        - Use at least 4-8 weeks of data for trend detection
        - Focus on top products by review volume
        - Adjust temperature based on your needs (0.0-0.2 for facts, 0.3-0.5 for insights)
        """)

def create_dashboard(df: pd.DataFrame):
    """Create the main dashboard with all visualizations"""
    st.markdown('<div class="section-header">📊 Interactive Dashboard</div>', unsafe_allow_html=True)
    
    # Data is already filtered when passed to this function
    if df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Create visualizations - Updated tabs with AI Analysis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Mentions Analysis", 
        "📝 Review Samples",
        "☁️ Word Cloud", 
        "😊 Sentiment Trend", 
        "⭐ Ratings", 
        "🚨 Problem Categories",
        "✨ Positive Categories",
        "🤖 AI Analysis"
    ])
    
    with tab1:
        create_mentions_chart(df)
    
    with tab2:
        create_review_samples(df)
    
    with tab3:
        create_wordcloud_visualization(df)
    
    with tab4:
        # Sentiment area chart over time
        create_sentiment_area_chart(df)
    
    with tab5:
        create_rating_distribution_chart(df)
    
    with tab6:
        create_problem_categories_chart(df)
    
    with tab7:
        create_positive_categories_chart(df)
    
    with tab8:
        create_ai_analysis_tab(df)

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">📊 Review Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    if not load_analysis_results():
        st.info("Please select or upload an analysis results file to begin.")
        return
    
    df = st.session_state.analysis_data
    
    # Try to merge with product metadata if available (only if not already merged)
    load_product_metadata()
    if st.session_state.products_data is not None and 'GENDER_CODE' not in df.columns:
        products_df = st.session_state.products_data
        products_df['PRODUCT_SKU_TEXT_UPPER'] = products_df['PRODUCT_SKU_TEXT'].str.upper()
        df['product_upper'] = df['product'].str.upper()
        
        # Merge with product data - including both CODE and TEXT columns for aggregation
        merge_columns = ['PRODUCT_SKU_TEXT_UPPER']
        
        # Add CODE columns for filtering
        code_columns = ['GENDER_CODE', 'PRODUCT_CLASS_CODE', 'PRODUCT_SUB_CLASS_CODE', 
                       'END_USE_CODE', 'SIZE_CODE']
        
        # Add the 5 TEXT columns requested for aggregation
        text_columns = ['GENDER_TEXT', 'PRODUCT_CLASS_TEXT', 'PRODUCT_SUB_CLASS_TEXT', 
                       'END_USE_TEXT', 'STYLE_CODE', 'STYLE_CODE_AND_TEXT']
        
        # Only include columns that exist in the products_df
        available_columns = merge_columns.copy()
        for col in code_columns + text_columns:
            if col in products_df.columns:
                available_columns.append(col)
        
        df = df.merge(
            products_df[available_columns],
            left_on='product_upper',
            right_on='PRODUCT_SKU_TEXT_UPPER',
            how='left'
        )
        # Use product_description from analysis file if available, otherwise use STYLE_CODE_AND_TEXT
        if 'product_description' not in df.columns:
            df['product_description'] = df['STYLE_CODE_AND_TEXT'].fillna(df['product'])
        st.session_state.analysis_data = df
    
    # Create filter section FIRST (always visible)
    create_filter_section(df)
    
    # Apply filters to get filtered dataframe
    filtered_df = apply_filters(df)
    
    # Show filter status if data is filtered
    if len(filtered_df) < len(df):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"🔍 Filters active: Showing {len(filtered_df):,} of {len(df):,} reviews")
        with col2:
            if st.button("Clear Filters"):
                # Clear the filter values
                st.session_state.filters = {
                    'gender': 'All',
                    'product_class': 'All',
                    'product_subclass': 'All',
                    'size': 'All',
                    'end_use': 'All',
                    'time_frame': 'All Time'
                }
                # Clear the widget states to reset the selectboxes
                if 'gender_filter' in st.session_state:
                    del st.session_state.gender_filter
                if 'class_filter' in st.session_state:
                    del st.session_state.class_filter
                if 'subclass_filter' in st.session_state:
                    del st.session_state.subclass_filter
                if 'size_filter' in st.session_state:
                    del st.session_state.size_filter
                if 'enduse_filter' in st.session_state:
                    del st.session_state.enduse_filter
                if 'time_frame_filter' in st.session_state:
                    del st.session_state.time_frame_filter
                if 'start_date_filter' in st.session_state:
                    del st.session_state.start_date_filter
                if 'end_date_filter' in st.session_state:
                    del st.session_state.end_date_filter
                st.rerun()
    
    # Display metrics based on FILTERED data
    display_summary_metrics(filtered_df)
    
    # Create dashboard with FILTERED data
    create_dashboard(filtered_df)
    
    # Display detailed results with FILTERED data
    display_detailed_results(filtered_df)
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Dashboard Info")
        st.info(f"📁 Current file: {st.session_state.selected_file}")
        st.metric("Total Reviews", f"{len(df):,}")
        
        if st.button("🔄 Reload Data"):
            st.session_state.analysis_data = None
            st.session_state.selected_file = None
            st.rerun()
        
        st.divider()
        
        st.caption("💡 Tips:")
        st.caption("• Use filters to focus on specific segments")
        st.caption("• Click legend items to show/hide data")
        st.caption("• Export filtered data for further analysis")

if __name__ == "__main__":
    main()