#!/usr/bin/env python3
"""
Streamlit Review Analysis App
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ReviewAnalyzer App",
    page_icon="🤖",
    layout="wide"
)

try:
    from process_reviews import (
        get_valid_problem_categories,
        create_extraction_config,
        analyze_review_with_llm
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    def get_valid_problem_categories():
        return ["Fit", "Comfort", "Price"]
    create_extraction_config = None
    analyze_review_with_llm = None

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
</style>
""", unsafe_allow_html=True)


# --- NEW: Helper function to robustly find JSON in a string ---
def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Finds and extracts the first JSON object string from a text block.
    This is useful for cleaning LLM responses that include conversational text.
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

def fetch_available_models():
    lm_studio_host = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
    try:
        response = requests.get(f"{lm_studio_host}/v1/models", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            if 'data' in models_data:
                return [model['id'] for model in models_data['data']]
    except requests.exceptions.RequestException:
        pass
    return []

def initialize_session_state():
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'problem_categories' not in st.session_state:
        st.session_state.problem_categories = get_valid_problem_categories()
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'processing_errors' not in st.session_state:
        st.session_state.processing_errors = []
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {
            'model_id': 'gemma-2-9b-it',
            'temperature': 0.0
        }
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

def process_reviews_batch(
    df: pd.DataFrame,
    column_mapping: dict,
    problem_categories: list,
    model_config: dict,
    max_rows: Optional[int] = None
):
    st.session_state.is_processing = True
    results, errors = [], []
    df_sample = (
        df.sample(n=max_rows)
        if max_rows and len(df) > max_rows
        else df.copy()
    )
    total_rows = len(df_sample)
    if not BACKEND_AVAILABLE:
        st.error("Backend not available. Cannot proceed.")
        st.session_state.is_processing = False
        return [], ["Backend missing"]
    try:
        prompt = create_extraction_config(
            valid_categories=problem_categories,
            model_id=model_config['model_id'],
            temperature=model_config['temperature']
        )
    except Exception as e:
        st.error(f"❌ Failed to create LLM extraction configuration: {e}.")
        st.session_state.is_processing = False
        return [], [f"Configuration error: {e}"]
    progress_bar = st.progress(0, text="Initializing...")
    status_placeholder = st.empty()
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        progress_bar.progress(
            (i + 1) / total_rows,
            text=f"Processing review {i+1}/{total_rows}..."
        )
        try:
            product_id = str(row.get(column_mapping.get('product'), 'N/A'))
            rating = row.get(column_mapping.get('rating'))
            date = row.get(column_mapping.get('date'))
            review_text = str(row.get(column_mapping.get('text'), ''))
            title_col = column_mapping.get('title')
            title = str(row.get(title_col, '')) if title_col else ""
            if pd.isna(review_text) or review_text.lower() == 'nan':
                review_text = ""
            if pd.isna(title) or title.lower() == 'nan':
                title = ""
            full_text = f"{title}. {review_text}".strip()
            if not full_text:
                continue
            try:
                sentiment, problem_list, positive_list = analyze_review_with_llm(
                    review_text=full_text,
                    rating=rating,
                    prompt=prompt,
                    model_config=model_config
                )
            except (ValueError, RuntimeError, ConnectionError) as e:
                error_msg = f"Error processing review (Index {idx}): {e}"
                errors.append(error_msg)
                status_placeholder.warning(
                    f"⚠️ Skipped review {i+1} due to LLM error. "
                    "See details at the end."
                )
                continue
            problems_str = (
                "; ".join(sorted(list(set(problem_list))))
                if problem_list
                else "None"
            )
            positives_str = (
                "; ".join(sorted(list(set(positive_list))))
                if positive_list
                else "None"
            )
            results.append({
                'date': date, 'product': product_id, 'rating': rating,
                'sentiment': sentiment, 'problems_mentioned': problems_str,
                'positive_mentions': positives_str,
                'original_text': full_text  # Save the full text that was analyzed, not just review_text
            })
        except Exception as e:
            errors.append(f"Unexpected error at Index {idx}: {e}")
            continue
    progress_bar.empty()
    status_placeholder.empty()
    st.session_state.is_processing = False
    return results, errors

def upload_csv_section():
    st.markdown(
        '<div class="section-header">📁 1. Upload CSV File</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Drag and drop a CSV file",
        type=['csv'],
        disabled=st.session_state.is_processing
    )
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            'last_uploaded_file_key' not in st.session_state or
            st.session_state.last_uploaded_file_key != file_key
        ):
            try:
                with st.spinner("Loading CSV..."):
                    df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                st.session_state.last_uploaded_file_key = file_key
                st.session_state.column_mapping = {}
                st.session_state.processing_results = None
                st.session_state.processing_errors = []
                st.success(f"✅ Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
                return False
        df = st.session_state.uploaded_data
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        return True
    return False

# --- MODIFIED: This function is now more robust ---
def match_csv_fields_to_analysis_fields(csv_columns: List[str], model_config: dict, sample_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    sample_data_str = ""
    if sample_df is not None and not sample_df.empty:
        sample_rows = min(5, len(sample_df))
        sample_data_str = f"\n\nSample Data Preview:\n{sample_df.head(sample_rows).to_markdown(index=False)}"
    
    prompt = f"""You are a precise CSV column mapping assistant. Your ONLY task is to return a JSON object mapping required analysis fields to the provided CSV column names.
CSV Columns Available: [{", ".join(csv_columns)}]
Analysis Fields Needed:
- product: Product SKU, ID, etc.
- rating: Numeric Rating
- date: Review Date
- title: Review Title (Optional)
- text: Review Text
{sample_data_str}
INSTRUCTIONS:
1. Review column names and sample data for the best match.
2. Use the EXACT CSV column name. For missing fields, use an empty string "".
3. Return ONLY the JSON object. NO explanations or markdown.
REQUIRED JSON FORMAT: {{"product": "col_name", "rating": "col_name", "date": "col_name", "title": "col_name", "text": "col_name"}}"""

    try:
        base_url = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            # Removed "response_format" for broader compatibility
            json={
                "model": model_config['model_id'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.0,
            },
            timeout=30
        )
        # Check for a successful response, not just 200
        if response.ok:
            response_text = response.json()['choices'][0]['message']['content']
            
            # Use the helper function to reliably find the JSON block
            json_string = _extract_json_from_text(response_text)

            if json_string:
                try:
                    mapping = json.loads(json_string)
                    validated_mapping = {field: '' for field in ['product', 'rating', 'date', 'title', 'text']}
                    for field, suggested_col in mapping.items():
                        if field in validated_mapping and suggested_col in csv_columns:
                            validated_mapping[field] = suggested_col
                    return validated_mapping
                except json.JSONDecodeError:
                    st.warning("AI found JSON, but it was malformed. Please map manually.")
            else:
                st.warning("AI response did not contain a valid JSON object. Please map manually.")

    except requests.exceptions.RequestException as e:
        st.error(f"AI field matching failed (Connection Error). Check LLM connection. {e}")
    except Exception as e:
        st.error(f"AI field matching failed: {e}")
    return {}

def column_selection_section():
    if st.session_state.uploaded_data is None:
        return False
    st.markdown('<div class="section-header">🎯 2. Column Selection</div>', unsafe_allow_html=True)
    df = st.session_state.uploaded_data
    columns = [''] + list(df.columns)
    if st.button("🤖 Auto-Match Fields with AI", disabled=st.session_state.is_processing or not BACKEND_AVAILABLE):
        with st.spinner("Analyzing CSV with AI..."):
            suggested_mapping = match_csv_fields_to_analysis_fields(csv_columns=list(df.columns), model_config=st.session_state.model_config, sample_df=df.head(5))
            if suggested_mapping:
                st.session_state.column_mapping = suggested_mapping
                st.success("✅ AI Auto-match complete. Please review.")
                st.rerun()
            else:
                st.warning("Could not auto-match. Please select manually.")
    if not st.session_state.column_mapping:
        st.session_state.column_mapping = {'product': '', 'rating': '', 'date': '', 'title': '', 'text': ''}
    st.write("Review and adjust column mappings:")
    col1, col2 = st.columns(2)
    def get_idx(field):
        val = st.session_state.column_mapping.get(field, '')
        return columns.index(val) if val in columns else 0
    with col1:
        st.session_state.column_mapping['product'] = st.selectbox("🏷️ Product Designator*", options=columns, index=get_idx('product'))
        st.session_state.column_mapping['rating'] = st.selectbox("⭐ Numeric Rating*", options=columns, index=get_idx('rating'))
        st.session_state.column_mapping['date'] = st.selectbox("📅 Review Date*", options=columns, index=get_idx('date'))
    with col2:
        st.session_state.column_mapping['text'] = st.selectbox("📄 Review Text*", options=columns, index=get_idx('text'))
        title_options = ['None (Skip)'] + list(df.columns)
        current_title = st.session_state.column_mapping.get('title', '')
        title_index = title_options.index(current_title) if current_title in title_options else 0
        selected_title = st.selectbox("📝 Review Title (Optional)", options=title_options, index=title_index)
        st.session_state.column_mapping['title'] = '' if selected_title == 'None (Skip)' else selected_title
    required = ['product', 'rating', 'date', 'text']
    missing = [field for field in required if not st.session_state.column_mapping.get(field)]
    if not missing:
        st.success("✅ All required columns selected.")
        return True
    st.warning(f"⚠️ Please select: {', '.join(missing)}")
    return False

def problem_categories_section():
    st.markdown('<div class="section-header">🏷️ 3. Problem Categories</div>', unsafe_allow_html=True)
    st.write("Define the categories for classification (one per line).")
    categories_text = "\n".join(st.session_state.problem_categories)
    updated_text = st.text_area("Problem Categories", value=categories_text, height=250)
    new_categories = [cat.strip() for cat in updated_text.split('\n') if cat.strip() and cat not in ['None', 'Other']]
    if new_categories != st.session_state.problem_categories:
        st.session_state.problem_categories = new_categories
    if st.button("Reset to Default Categories"):
        st.session_state.problem_categories = get_valid_problem_categories()
        st.rerun()

def processing_section():
    st.markdown('<div class="section-header">⚙️ 4. Execution</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_data is None or not BACKEND_AVAILABLE:
        return
    total_rows = len(st.session_state.uploaded_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Test Run")
        test_size = st.slider("Rows for test", 0, total_rows, min(20, total_rows), disabled=st.session_state.is_processing)
        if st.button("🧪 Run Test", type="secondary", disabled=st.session_state.is_processing):
            run_processing(test_size)
    with col2:
        st.subheader("🚀 Full Run")
        st.metric("Total Rows", f"{total_rows:,}")
        if st.button("🚀 Run All", type="primary", disabled=st.session_state.is_processing):
            run_processing(None)

def run_processing(max_rows: Optional[int]):
    start_time = time.time()
    st.session_state.processing_results, st.session_state.processing_errors = None, []
    results, errors = process_reviews_batch(st.session_state.uploaded_data, st.session_state.column_mapping, st.session_state.problem_categories, st.session_state.model_config, max_rows)
    st.session_state.processing_results, st.session_state.processing_errors = results, errors
    processing_time = time.time() - start_time
    if results:
        st.success(f"🎉 Analysis complete! Processed {len(results):,} reviews in {processing_time:.1f}s.")
    if errors:
        st.error(f"⚠️ Encountered {len(errors)} errors ({len(results)} successful).")
        with st.expander("🔍 View Error Details"):
            for error in errors:
                st.write(error)

def display_results_summary(df: pd.DataFrame):
    if df.empty:
        return
    st.markdown('<div class="section-header">📊 Results Summary</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Processed", f"{len(df):,}")
    with col2:
        pos_pct = (df['sentiment'].value_counts(normalize=True).get('Positive', 0)) * 100
        st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    with col3:
        prob_pct = (sum(1 for p in df['problems_mentioned'] if p != 'None') / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Reviews w/ Problems", f"{prob_pct:.1f}%")
    with col4:
        if 'positive_mentions' in df.columns:
            pos_mentions_pct = (sum(1 for p in df['positive_mentions'] if p != 'None') / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Reviews w/ Positives", f"{pos_mentions_pct:.1f}%")
        else:
            st.metric("Reviews w/ Positives", "N/A")
    with col5:
        avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
    display_cols = [c for c in ['date', 'product', 'rating', 'sentiment', 'problems_mentioned', 'positive_mentions', 'original_text'] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

def create_analytics_dashboard(df: pd.DataFrame):
    """
    Creates an analytics dashboard for review data with product metadata filters.

    This dashboard includes:
    - Product metadata filters (Gender, Product Class, Product Sub-Class)
    - What's Mentioned Most visualization showing positive vs negative mentions
    
    Args:
        df (pd.DataFrame): The DataFrame containing the processed review results merged with product data.
                           Must include 'product', 'rating', 'sentiment', 
                           'problems_mentioned', 'positive_mentions', and product metadata columns.
    """
    st.subheader("📈 Analytics Dashboard")

    # --- 1. Load Product Data ---
    try:
        # Read products.csv if it exists
        products_df = None
        if os.path.exists('products.csv'):
            products_df = pd.read_csv('products.csv')
            # Convert to uppercase for joining
            products_df['PRODUCT_SKU_TEXT_UPPER'] = products_df['PRODUCT_SKU_TEXT'].str.upper()
        
        df_processed = df.copy()
        
        # Convert product column to uppercase for joining
        df_processed['product_upper'] = df_processed['product'].str.upper()
        
        # Merge with product data if available
        if products_df is not None:
            df_processed = df_processed.merge(
                products_df[['PRODUCT_SKU_TEXT_UPPER', 'GENDER_CODE', 'PRODUCT_CLASS_CODE', 'PRODUCT_SUB_CLASS_CODE', 
                             'END_USE_CODE', 'GENDER_TEXT', 'PRODUCT_CLASS_TEXT', 'PRODUCT_SUB_CLASS_TEXT', 'END_USE_TEXT']],
                left_on='product_upper',
                right_on='PRODUCT_SKU_TEXT_UPPER',
                how='left'
            )
        
        # Convert date and rating columns
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed['rating'] = pd.to_numeric(df_processed['rating'], errors='coerce')

    except Exception as e:
        st.error(f"❌ Error preparing data for dashboard: {e}")
        return

    # --- 2. Product Metadata Filters ---
    st.markdown("#### Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Gender filter
    with col1:
        if 'GENDER_CODE' in df_processed.columns:
            gender_options = ['All'] + sorted(df_processed['GENDER_CODE'].dropna().unique().tolist())
            selected_gender = st.selectbox("Gender:", options=gender_options)
        else:
            selected_gender = 'All'
    
    # Product Class filter
    with col2:
        if 'PRODUCT_CLASS_CODE' in df_processed.columns:
            class_options = ['All'] + sorted(df_processed['PRODUCT_CLASS_CODE'].dropna().unique().tolist())
            selected_class = st.selectbox("Product Class:", options=class_options)
        else:
            selected_class = 'All'
    
    # Product Sub-Class filter
    with col3:
        if 'PRODUCT_SUB_CLASS_CODE' in df_processed.columns:
            subclass_options = ['All'] + sorted(df_processed['PRODUCT_SUB_CLASS_CODE'].dropna().unique().tolist())
            selected_subclass = st.selectbox("Product Sub-Class:", options=subclass_options)
        else:
            selected_subclass = 'All'
    
    # End Use filter
    with col4:
        if 'END_USE_CODE' in df_processed.columns:
            end_use_options = ['All'] + sorted(df_processed['END_USE_CODE'].dropna().unique().tolist())
            selected_end_use = st.selectbox("End Use:", options=end_use_options)
        else:
            selected_end_use = 'All'
    
    # Apply filters
    dff = df_processed.copy()
    
    if selected_gender != 'All' and 'GENDER_CODE' in dff.columns:
        dff = dff[dff['GENDER_CODE'] == selected_gender]
    
    if selected_class != 'All' and 'PRODUCT_CLASS_CODE' in dff.columns:
        dff = dff[dff['PRODUCT_CLASS_CODE'] == selected_class]
    
    if selected_subclass != 'All' and 'PRODUCT_SUB_CLASS_CODE' in dff.columns:
        dff = dff[dff['PRODUCT_SUB_CLASS_CODE'] == selected_subclass]
    
    if selected_end_use != 'All' and 'END_USE_CODE' in dff.columns:
        dff = dff[dff['END_USE_CODE'] == selected_end_use]

    if dff.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- 3. What's Mentioned Most Chart (NEW) ---
    st.markdown("#### What's Mentioned Most?")
    
    # Calculate counts for problems and positives
    problems_counts = {}
    positives_counts = {}
    
    for _, row in dff.iterrows():
        # Count problems
        if pd.notna(row['problems_mentioned']) and row['problems_mentioned'] != 'None':
            for problem in row['problems_mentioned'].split('; '):
                problems_counts[problem] = problems_counts.get(problem, 0) + 1
        
        # Count positives
        if 'positive_mentions' in row and pd.notna(row['positive_mentions']) and row['positive_mentions'] != 'None':
            for positive in row['positive_mentions'].split('; '):
                positives_counts[positive] = positives_counts.get(positive, 0) + 1
    
    # Get all unique categories mentioned
    all_categories = list(set(list(problems_counts.keys()) + list(positives_counts.keys())))
    
    # Calculate percentages and prepare data
    total_reviews = len(dff)
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
    if not mentions_df.empty:
        mentions_df = mentions_df.sort_values('Total %', ascending=False).head(12)
        
        # Create the diverging bar chart
        fig_mentions = go.Figure()
        
        # Add negative mentions (left side - orange)
        fig_mentions.add_trace(go.Bar(
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
        fig_mentions.add_trace(go.Bar(
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
        fig_mentions.update_layout(
            title='What\'s Mentioned Most?',
            xaxis=dict(
                title='Percentage of Reviews',
                tickformat='.0f',
                ticksuffix='%',
                range=[-max(mentions_df['Negative %'].max(), mentions_df['Positive %'].max()) - 5,
                       max(mentions_df['Negative %'].max(), mentions_df['Positive %'].max()) + 5]
            ),
            yaxis=dict(
                title='',
                autorange='reversed'  # Reverse to have highest at top
            ),
            barmode='overlay',
            template='plotly_white',
            height=400,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            )
        )
        
        # Add a vertical line at x=0
        fig_mentions.add_vline(x=0, line_width=1, line_color='gray')
        
        st.plotly_chart(fig_mentions, use_container_width=True)
    else:
        st.info("No mention data available to display.")

    # Chart rendering section removed - only keeping What's Mentioned Most

        
def download_results_section():
    if st.session_state.processing_results is None:
        return
    df = pd.DataFrame(st.session_state.processing_results)
    if df.empty:
        if not st.session_state.processing_errors:
            st.warning("No results generated.")
        return
    display_results_summary(df)
    create_analytics_dashboard(df)
    st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
    @st.cache_data
    def to_csv(df_to_convert):
        df_out = df_to_convert.rename(columns={'sentiment': 'Overall review sentiment'})
        cols = [c for c in ['date', 'product', 'rating', 'Overall review sentiment', 'problems_mentioned', 'positive_mentions', 'original_text'] if c in df_out.columns]
        return df_out[cols].to_csv(index=False).encode('utf-8')
    csv_data = to_csv(df)
    st.download_button("📥 Download Results as CSV", csv_data, f"review_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

def model_configuration_sidebar():
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        if not BACKEND_AVAILABLE:
            st.error("❌ LLM features disabled. 'process_reviews.py' missing.")
        st.info("Ensure your LLM host (e.g., LM Studio) is running.")
        available_models = fetch_available_models()
        if available_models:
            st.success("✅ Connected to LLM Host")
            current_model = st.session_state.model_config.get('model_id')
            if current_model not in available_models and available_models:
                current_model = available_models[0]
                st.session_state.model_config['model_id'] = current_model
            try:
                current_index = available_models.index(current_model) if current_model in available_models else 0
            except ValueError:
                current_index = 0
            st.session_state.model_config['model_id'] = st.selectbox("Select Model", options=available_models, index=current_index)
        else:
            st.error("❌ Could not connect. Using manual entry.")
            st.session_state.model_config['model_id'] = st.text_input("Model ID", value=st.session_state.model_config.get('model_id', 'gemma-2-9b-it'))
        if st.button("🔄 Refresh Connection"):
            st.rerun()
        st.session_state.model_config['temperature'] = st.slider("Temperature", 0.0, 1.0, st.session_state.model_config.get('temperature', 0.0), 0.01, help="0.0 for deterministic JSON")

def main():
    initialize_session_state()
    st.markdown('<div class="main-header">🤖 LLM Review Analyzer</div>', unsafe_allow_html=True)
    if not BACKEND_AVAILABLE:
        st.error("❌ Major Error: The backend logic file `process_reviews.py` is missing or has an error. The app cannot function without it.")
        return
    model_configuration_sidebar()
    if upload_csv_section():
        if column_selection_section():
            problem_categories_section()
            processing_section()
    else:
        st.info("Please upload a CSV file to begin analysis.")
    download_results_section()

if __name__ == "__main__":
    main()