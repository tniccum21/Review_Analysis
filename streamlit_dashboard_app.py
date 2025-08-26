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

def compute_ai_metrics(df: pd.DataFrame, granularity: str = 'week') -> Dict[str, Any]:
    """Compute aggregated metrics for AI analysis"""
    
    # Convert date to datetime and create period column
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if granularity == 'week':
        df['period'] = df['date'].dt.to_period('W').dt.start_time
    else:  # month
        df['period'] = df['date'].dt.to_period('M').dt.start_time
    
    # Filter out rows with invalid dates
    df = df.dropna(subset=['date', 'period'])
    
    if df.empty:
        return {'error': 'No valid data after date processing'}
    
    # Create sentiment binary columns
    df['is_positive'] = df['sentiment'] == 'Positive'
    df['is_negative'] = df['sentiment'] == 'Negative'
    df['is_neutral'] = df['sentiment'] == 'Neutral'
    
    # Group by period and product
    group_cols = ['period', 'product']
    if 'product_description' in df.columns:
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
    agg_df.columns = ['period', 'product'] + (['product_description'] if 'product_description' in group_cols else []) + \
                      ['n_reviews', 'avg_rating', 'p_pos', 'p_neg', 'p_neu']
    
    # Process problem and positive mentions
    problem_counts = []
    positive_counts = []
    
    for (period, product), group in df.groupby(['period', 'product']):
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
        
        problem_counts.append({'period': period, 'product': product, 'problem_counts': problems})
        positive_counts.append({'period': period, 'product': product, 'positive_counts': positives})
    
    # Merge problem and positive counts
    problem_df = pd.DataFrame(problem_counts)
    positive_df = pd.DataFrame(positive_counts)
    
    agg_df = agg_df.merge(problem_df, on=['period', 'product'], how='left')
    agg_df = agg_df.merge(positive_df, on=['period', 'product'], how='left')
    
    # Fill missing counts with empty dicts
    agg_df['problem_counts'] = agg_df['problem_counts'].fillna({}).apply(lambda x: x if x else {})
    agg_df['positive_counts'] = agg_df['positive_counts'].fillna({}).apply(lambda x: x if x else {})
    
    # Calculate rolling baselines and z-scores
    agg_df = agg_df.sort_values(['product', 'period'])
    
    # Rolling statistics for negative sentiment
    agg_df['neg_roll_mean'] = agg_df.groupby('product')['p_neg'].transform(
        lambda s: s.rolling(8, min_periods=4, center=False).mean()
    )
    agg_df['neg_roll_std'] = agg_df.groupby('product')['p_neg'].transform(
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
    agg_df['rating_delta'] = agg_df.groupby('product')['avg_rating'].transform(
        lambda s: s - s.rolling(8, min_periods=4, center=False).mean()
    )
    agg_df['neg_delta_pct'] = agg_df.groupby('product')['p_neg'].transform(
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
        'total_reviews': len(df),
        'products_analyzed': df['product'].nunique(),
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    }

def prepare_llm_payload(metrics: Dict[str, Any], top_n_products: int = 20) -> Dict[str, Any]:
    """Prepare the compact JSON payload for LLM analysis"""
    
    agg_df = metrics['aggregated_data']
    
    # Get top products by review volume
    top_products = (
        agg_df.groupby('product')['n_reviews']
        .sum()
        .nlargest(top_n_products)
        .index.tolist()
    )
    
    # Filter to top products and recent periods
    recent_df = agg_df[agg_df['product'].isin(top_products)].copy()
    recent_df = recent_df.sort_values('period').tail(200)  # Last 200 data points
    
    # Build series data for JSON
    series = []
    for _, row in recent_df.iterrows():
        series_item = {
            'period': row['period'].strftime('%Y-%m-%d'),
            'product': row['product'],
            'n_reviews': int(row['n_reviews']),
            'avg_rating': round(float(row['avg_rating']), 2) if pd.notna(row['avg_rating']) else None,
            'p_pos': round(float(row['p_pos']), 3) if pd.notna(row['p_pos']) else 0,
            'p_neu': round(float(row['p_neu']), 3) if pd.notna(row['p_neu']) else 0,
            'p_neg': round(float(row['p_neg']), 3) if pd.notna(row['p_neg']) else 0,
            'problem_counts': dict(row.get('problem_counts', {})),
            'positive_counts': dict(row.get('positive_counts', {})),
            'deltas': {
                'rating_delta': round(float(row.get('rating_delta', 0)), 2) if pd.notna(row.get('rating_delta')) else 0,
                'neg_delta_pct': round(float(row.get('neg_delta_pct', 0)), 1) if pd.notna(row.get('neg_delta_pct')) else 0
            },
            'z_scores': {
                'p_neg': round(float(row.get('z_neg', 0)), 2) if pd.notna(row.get('z_neg')) else 0
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
            'total_reviews': metrics.get('total_reviews', 0),
            'products_analyzed': len(top_products),
            'anomaly_rules': {
                'z_abs_threshold': 2.0,
                'min_count': 15
            }
        },
        'series': series
    }
    
    return payload

def create_ai_analysis_prompt(payload: Dict[str, Any]) -> tuple[str, str]:
    """Create the structured prompt for LLM analysis"""
    
    system_prompt = """You are a senior retail analytics copilot. You ONLY use facts in the provided JSON.
Prioritize statistically meaningful change. Avoid vague claims.
When you cite a reason, name the metric(s) and the period(s) that moved."""
    
    user_prompt = f"""Goal: Find trends over time, surface anomalies, and explain likely drivers with supporting slices.

Data: {json.dumps(payload, indent=2)}

Rules:
- Treat each row as period×product summary.
- An anomaly requires min_count >= 15 AND |z| >= 2.0.
- Prefer changes that persist >=2 periods.
- Map problems/positives into themes if obvious.
- If data are insufficient, say so explicitly.

Output a structured JSON analysis followed by a brief narrative (8-12 lines):

{{
  "brand_trends": [
    {{"theme":"<theme>","direction":"up/down","evidence":[{{"period":"YYYY-MM-DD","metric":"<metric>","value":<value>}}]}}
  ],
  "product_highlights": [
    {{"product":"<product>", "issue":"<description>", "metric":"<metric>", "z":<z-score>, "delta_pct":"<change>", "periods":["YYYY-MM-DD"]}}
  ],
  "emerging_topics": [
    {{"label":"<topic>","products":["<product>"],"trend":"<description>","support_trend":[<counts>]}}
  ],
  "risk_watchlist": [
    {{"product":"<product>", "reason":"<explanation>", "action":"<recommendation>"}}
  ],
  "positive_drivers": [
    {{"theme":"<theme>","products":["<product>"],"evidence":"<description>"}}
  ]
}}

Then add a narrative that tells the story in plain English."""
    
    return system_prompt, user_prompt

def call_llm_api(system_prompt: str, user_prompt: str, model_config: Dict[str, str]) -> str:
    """Call the LLM API (LM Studio or other providers)"""
    
    # Default to LM Studio endpoint
    api_url = model_config.get('api_url', 'http://localhost:1234/v1/chat/completions')
    api_key = model_config.get('api_key', 'not-needed')  # LM Studio doesn't need a real key
    model_id = model_config.get('model_id', 'gemma-2-9b-it')
    temperature = float(model_config.get('temperature', 0.1))
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    data = {
        'model': model_id,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': temperature,
        'max_tokens': 2000
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error calling LLM API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {str(e)}"

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response to extract JSON and narrative"""
    
    # Try to find JSON block in the response
    json_start = response.find('{')
    json_end = response.rfind('}')
    
    if json_start != -1 and json_end != -1:
        json_str = response[json_start:json_end + 1]
        try:
            analysis_json = json.loads(json_str)
            # Extract narrative (everything after the JSON)
            narrative = response[json_end + 1:].strip()
            return {
                'success': True,
                'analysis': analysis_json,
                'narrative': narrative,
                'raw_response': response
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback if JSON parsing fails
    return {
        'success': False,
        'analysis': {},
        'narrative': response,
        'raw_response': response
    }

def create_ai_analysis_tab(df: pd.DataFrame):
    """Create the AI-powered analysis tab"""
    
    st.markdown("#### 🤖 AI-Powered Analysis")
    
    # Configuration section
    with st.expander("⚙️ AI Analysis Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            granularity = st.selectbox(
                "Time Granularity:",
                ["week", "month"],
                index=0,
                help="Choose the time period for aggregation"
            )
        
        with col2:
            top_n = st.slider(
                "Top N Products:",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Number of top products to analyze"
            )
        
        with col3:
            min_reviews = st.number_input(
                "Min Reviews per Period:",
                min_value=5,
                max_value=100,
                value=15,
                step=5,
                help="Minimum reviews required for statistical significance"
            )
    
    # LLM Configuration
    st.markdown("##### 🔧 LLM Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox(
            "LLM Provider:",
            ["LM Studio (Local)", "OpenAI", "Custom API"],
            index=0
        )
        
        if provider == "LM Studio (Local)":
            api_url = st.text_input(
                "API URL:",
                value="http://localhost:1234/v1/chat/completions",
                help="Local LM Studio endpoint"
            )
            api_key = "not-needed"
        elif provider == "OpenAI":
            api_url = "https://api.openai.com/v1/chat/completions"
            api_key = st.text_input(
                "API Key:",
                type="password",
                help="Your OpenAI API key"
            )
        else:
            api_url = st.text_input(
                "API URL:",
                help="Your custom API endpoint"
            )
            api_key = st.text_input(
                "API Key:",
                type="password",
                help="API key if required"
            )
    
    with col2:
        # Try to fetch available models
        available_models = []
        if provider == "LM Studio (Local)":
            try:
                models_response = requests.get(
                    api_url.replace('/v1/chat/completions', '/v1/models'),
                    timeout=3
                )
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    if 'data' in models_data:
                        available_models = [m['id'] for m in models_data['data']]
            except:
                pass
        
        if available_models:
            model_id = st.selectbox(
                "Model:",
                available_models,
                index=0
            )
        else:
            model_id = st.text_input(
                "Model ID:",
                value="gemma-2-9b-it" if provider == "LM Studio (Local)" else "gpt-4",
                help="Model identifier"
            )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Lower = more focused, Higher = more creative"
        )
    
    # Analysis button and results
    if st.button("🚀 Generate AI Analysis", type="primary"):
        with st.spinner("Computing metrics..."):
            # Compute metrics
            metrics = compute_ai_metrics(df, granularity)
            
            if 'error' in metrics:
                st.error(metrics['error'])
                return
            
            st.success(f"✅ Computed metrics for {metrics['products_analyzed']} products across {metrics['total_reviews']} reviews")
        
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
            
            # Call LLM
            model_config = {
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
            tabs = st.tabs(["📈 Trends", "🎯 Highlights", "⚠️ Risks", "✨ Positives", "📝 Narrative", "🔍 Raw Data"])
            
            with tabs[0]:  # Trends
                if 'brand_trends' in analysis and analysis['brand_trends']:
                    st.markdown("##### Brand Trends")
                    for trend in analysis['brand_trends']:
                        direction_icon = "📈" if trend.get('direction') == 'up' else "📉"
                        st.write(f"{direction_icon} **{trend.get('theme', 'Unknown')}**: {trend.get('direction', 'Unknown')}")
                        if 'evidence' in trend:
                            for evidence in trend['evidence'][:3]:  # Show top 3 evidence points
                                st.caption(f"  • {evidence.get('period', '')}: {evidence.get('metric', '')} = {evidence.get('value', '')}")
                else:
                    st.info("No significant brand trends detected")
                
                if 'emerging_topics' in analysis and analysis['emerging_topics']:
                    st.markdown("##### Emerging Topics")
                    for topic in analysis['emerging_topics']:
                        st.write(f"🆕 **{topic.get('label', 'Unknown')}**: {topic.get('trend', 'Unknown')}")
                        if 'products' in topic:
                            st.caption(f"  Products: {', '.join(topic['products'][:5])}")
            
            with tabs[1]:  # Highlights
                if 'product_highlights' in analysis and analysis['product_highlights']:
                    st.markdown("##### Product Highlights")
                    for highlight in analysis['product_highlights']:
                        z_score = highlight.get('z', 0)
                        severity = "🔴" if abs(z_score) > 3 else "🟡" if abs(z_score) > 2 else "🟢"
                        st.write(f"{severity} **{highlight.get('product', 'Unknown')}**")
                        st.write(f"  Issue: {highlight.get('issue', 'Unknown')}")
                        st.caption(f"  Z-score: {z_score:.2f} | Change: {highlight.get('delta_pct', 'N/A')}")
                else:
                    st.info("No significant product highlights detected")
            
            with tabs[2]:  # Risks
                if 'risk_watchlist' in analysis and analysis['risk_watchlist']:
                    st.markdown("##### Risk Watchlist")
                    for risk in analysis['risk_watchlist']:
                        st.write(f"⚠️ **{risk.get('product', 'Unknown')}**")
                        st.write(f"  Reason: {risk.get('reason', 'Unknown')}")
                        st.info(f"  💡 Action: {risk.get('action', 'No recommendation')}")
                else:
                    st.success("No significant risks detected")
            
            with tabs[3]:  # Positives
                if 'positive_drivers' in analysis and analysis['positive_drivers']:
                    st.markdown("##### Positive Drivers")
                    for driver in analysis['positive_drivers']:
                        st.write(f"✅ **{driver.get('theme', 'Unknown')}**")
                        if 'products' in driver:
                            st.caption(f"  Products: {', '.join(driver['products'][:5])}")
                        st.caption(f"  Evidence: {driver.get('evidence', 'No evidence')}")
                else:
                    st.info("No significant positive drivers detected")
            
            with tabs[4]:  # Narrative
                st.markdown("##### AI Narrative Summary")
                if parsed['narrative']:
                    st.write(parsed['narrative'])
                else:
                    st.info("No narrative provided")
            
            with tabs[5]:  # Raw Data
                st.markdown("##### Raw Analysis Data")
                
                # Show the metrics summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", f"{metrics['total_reviews']:,}")
                with col2:
                    st.metric("Products Analyzed", metrics['products_analyzed'])
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
            st.markdown("##### Raw Response:")
            st.text_area("", value=parsed['raw_response'], height=400)
    
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
        
        # Merge with product data - including SIZE_CODE and STYLE_CODE_AND_TEXT
        df = df.merge(
            products_df[['PRODUCT_SKU_TEXT_UPPER', 'GENDER_CODE', 'PRODUCT_CLASS_CODE', 
                        'PRODUCT_SUB_CLASS_CODE', 'END_USE_CODE', 'SIZE_CODE', 'STYLE_CODE_AND_TEXT']],
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