#!/usr/bin/env python3
"""
Streamlit Review Analysis App - Analysis Module
This app handles the upload, configuration, and analysis of review data.
Results are saved to CSV for use by the dashboard app.
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import requests

st.set_page_config(
    page_title="Review Analyzer - Analysis",
    page_icon="🔬",
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
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to robustly find JSON in a string
def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Finds and extracts the first JSON object string from a text block.
    This is useful for cleaning LLM responses that include conversational text.
    """
    try:
        start_index = text.find('{')
        if start_index == -1:
            return None
        
        end_index = text.rfind('}')
        if end_index == -1 or end_index < start_index:
            return None
        
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
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'output_filename' not in st.session_state:
        st.session_state.output_filename = None

def process_reviews_batch(
    df: pd.DataFrame,
    column_mapping: dict,
    problem_categories: list,
    model_config: dict,
    max_rows: Optional[int] = None
):
    st.session_state.is_processing = True
    results, errors = [], []
    
    # Try to load and merge product metadata if available
    product_descriptions = {}
    if os.path.exists('products.csv'):
        try:
            products_df = pd.read_csv('products.csv', low_memory=False)
            # Create lookup dictionary for product descriptions
            products_df['PRODUCT_SKU_TEXT_UPPER'] = products_df['PRODUCT_SKU_TEXT'].str.upper()
            product_descriptions = dict(zip(
                products_df['PRODUCT_SKU_TEXT_UPPER'], 
                products_df['STYLE_CODE_AND_TEXT'].fillna(products_df['PRODUCT_SKU_TEXT'])
            ))
            st.info(f"Loaded product descriptions for {len(product_descriptions)} products")
        except Exception as e:
            st.warning(f"Could not load product metadata: {e}")
    
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
            
            # Get product description if available
            product_desc = product_descriptions.get(str(product_id).upper(), product_id)
            
            results.append({
                'date': date,
                'product': product_id,
                'product_description': product_desc,  # Add product description
                'rating': rating,
                'sentiment': sentiment,
                'problems_mentioned': problems_str,
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
        "Drag and drop a CSV file containing reviews",
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
                st.session_state.analysis_complete = False
                st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
                return False
        
        df = st.session_state.uploaded_data
        with st.expander("📊 Data Preview (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        return True
    return False

def match_csv_fields_to_analysis_fields(csv_columns: List[str], model_config: dict, sample_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Auto-match CSV columns to required analysis fields using LLM."""
    
    try:
        sample_data_str = ""
        if sample_df is not None and not sample_df.empty:
            # Create simplified sample preview
            sample_preview = []
            for col in csv_columns[:5]:  # Show first 5 columns only
                if col in sample_df.columns:
                    sample_values = sample_df[col].head(2).tolist()
                    sample_preview.append(f"{col}: {sample_values}")
            sample_data_str = f"\n\nSample Data:\n" + "\n".join(sample_preview)
        
        prompt = f"""Map these CSV columns to review analysis fields. Return ONLY a JSON object.

CSV Columns: {csv_columns}
{sample_data_str}

Required fields to map:
- product: Product identifier/SKU
- rating: Numeric rating (1-5)
- date: Review date
- title: Review title (optional, use empty string if not found)
- text: Review text/comment

Return JSON: {{"product": "column_name", "rating": "column_name", "date": "column_name", "title": "", "text": "column_name"}}"""

        base_url = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_config.get('model_id', 'local-model'),
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "stream": False
            },
            timeout=15
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                response_text = response_data['choices'][0]['message']['content']
                
                # Extract JSON from response
                json_string = _extract_json_from_text(response_text)
                if json_string:
                    try:
                        mapping = json.loads(json_string)
                        validated_mapping = {field: '' for field in ['product', 'rating', 'date', 'title', 'text']}
                        for field, suggested_col in mapping.items():
                            if field in validated_mapping and suggested_col in csv_columns:
                                validated_mapping[field] = suggested_col
                        
                        # If we got valid mappings, return them
                        if any(validated_mapping.values()):
                            return validated_mapping
                    except json.JSONDecodeError:
                        st.error("LLM returned invalid JSON. Please map fields manually.")
        else:
            st.error(f"LLM API error (status {response.status_code}). Please map fields manually.")
            
    except requests.exceptions.ConnectionError as e:
        st.error("Cannot connect to LM Studio. Please ensure it's running and try again.")
    except requests.exceptions.Timeout:
        st.error("LM Studio request timed out. Please try again.")
    except Exception as e:
        st.error(f"Error calling LLM: {str(e)}")
    
    return {}

def column_selection_section():
    if st.session_state.uploaded_data is None:
        return False
    
    st.markdown('<div class="section-header">🎯 2. Column Selection</div>', unsafe_allow_html=True)
    
    df = st.session_state.uploaded_data
    columns = [''] + list(df.columns)
    
    if st.button("🤖 Auto-Match Fields with AI", disabled=st.session_state.is_processing or not BACKEND_AVAILABLE):
        with st.spinner("Analyzing CSV structure with AI..."):
            suggested_mapping = match_csv_fields_to_analysis_fields(
                csv_columns=list(df.columns),
                model_config=st.session_state.model_config,
                sample_df=df.head(5)
            )
            if suggested_mapping:
                st.session_state.column_mapping = suggested_mapping
                st.success("✅ AI Auto-match complete. Please review and adjust if needed.")
                st.rerun()
            else:
                st.warning("Could not auto-match. Please select manually.")
    
    if not st.session_state.column_mapping:
        st.session_state.column_mapping = {'product': '', 'rating': '', 'date': '', 'title': '', 'text': ''}
    
    st.write("Map CSV columns to analysis fields:")
    
    col1, col2 = st.columns(2)
    
    def get_idx(field):
        val = st.session_state.column_mapping.get(field, '')
        return columns.index(val) if val in columns else 0
    
    with col1:
        st.session_state.column_mapping['product'] = st.selectbox(
            "🏷️ Product Identifier* (SKU/ID)", options=columns, index=get_idx('product')
        )
        st.session_state.column_mapping['rating'] = st.selectbox(
            "⭐ Numeric Rating*", options=columns, index=get_idx('rating')
        )
        st.session_state.column_mapping['date'] = st.selectbox(
            "📅 Review Date*", options=columns, index=get_idx('date')
        )
    
    with col2:
        st.session_state.column_mapping['text'] = st.selectbox(
            "📄 Review Text*", options=columns, index=get_idx('text')
        )
        title_options = ['None (Skip)'] + list(df.columns)
        current_title = st.session_state.column_mapping.get('title', '')
        title_index = title_options.index(current_title) if current_title in title_options else 0
        selected_title = st.selectbox(
            "📝 Review Title (Optional)", options=title_options, index=title_index
        )
        st.session_state.column_mapping['title'] = '' if selected_title == 'None (Skip)' else selected_title
    
    required = ['product', 'rating', 'date', 'text']
    missing = [field for field in required if not st.session_state.column_mapping.get(field)]
    
    if not missing:
        st.success("✅ All required columns selected.")
        return True
    
    st.warning(f"⚠️ Please select columns for: {', '.join(missing)}")
    return False

def problem_categories_section():
    st.markdown('<div class="section-header">🏷️ 3. Configure Problem Categories</div>', unsafe_allow_html=True)
    
    st.write("Define the categories for classification. The system will categorize both problems and positive mentions into these categories.")
    
    categories_text = "\n".join(st.session_state.problem_categories)
    updated_text = st.text_area(
        "Categories (one per line):",
        value=categories_text,
        height=250,
        help="Enter each category on a new line. 'Other' will be added automatically for items that don't fit."
    )
    
    new_categories = [cat.strip() for cat in updated_text.split('\n') 
                     if cat.strip() and cat not in ['None', 'Other']]
    
    if new_categories != st.session_state.problem_categories:
        st.session_state.problem_categories = new_categories
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset to Default Categories"):
            st.session_state.problem_categories = get_valid_problem_categories()
            st.rerun()
    
    with col2:
        st.info(f"📊 {len(st.session_state.problem_categories)} categories configured")

def processing_section():
    st.markdown('<div class="section-header">⚙️ 4. Run Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None or not BACKEND_AVAILABLE:
        if not BACKEND_AVAILABLE:
            st.error("❌ Backend not available. Please ensure process_reviews.py is present.")
        return
    
    # Add date range filter
    df = st.session_state.uploaded_data
    date_col = st.session_state.column_mapping.get('date')
    
    if date_col and date_col in df.columns:
        st.markdown("### 📅 Date Range Filter")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        date_df = df.dropna(subset=[date_col])
        
        if not date_df.empty:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                use_date_filter = st.checkbox("Apply Date Range Filter", key="use_date_filter")
            
            if use_date_filter:
                with col2:
                    start_date = st.date_input(
                        "Start Date",
                        value=date_df[date_col].min(),
                        min_value=date_df[date_col].min(),
                        max_value=date_df[date_col].max(),
                        key="analysis_start_date"
                    )
                
                with col3:
                    end_date = st.date_input(
                        "End Date",
                        value=date_df[date_col].max(),
                        min_value=date_df[date_col].min(),
                        max_value=date_df[date_col].max(),
                        key="analysis_end_date"
                    )
                
                # Apply date filter
                filtered_df = df[(df[date_col] >= pd.to_datetime(start_date)) & 
                                (df[date_col] <= pd.to_datetime(end_date))]
                st.session_state.filtered_data = filtered_df
                
                # Show filtered stats
                st.info(f"📊 Date filter applied: {len(filtered_df):,} of {len(df):,} reviews selected ({len(filtered_df)/len(df)*100:.1f}%)")
            else:
                st.session_state.filtered_data = df
        else:
            st.warning("No valid dates found in the date column.")
            st.session_state.filtered_data = df
    else:
        st.session_state.filtered_data = st.session_state.uploaded_data
    
    st.divider()
    
    # Use filtered data for metrics
    total_rows = len(st.session_state.filtered_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Reviews to Analyze", f"{total_rows:,}")
    
    with col2:
        test_size = st.number_input(
            "Test Sample Size",
            min_value=1,
            max_value=total_rows,
            value=min(20, total_rows),
            disabled=st.session_state.is_processing
        )
    
    with col3:
        st.metric("Selected Model", st.session_state.model_config['model_id'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            f"🧪 Run Test Analysis ({test_size} reviews)",
            type="secondary",
            disabled=st.session_state.is_processing,
            use_container_width=True
        ):
            run_processing(test_size)
    
    with col2:
        if st.button(
            f"🚀 Run Full Analysis ({total_rows:,} reviews)",
            type="primary",
            disabled=st.session_state.is_processing,
            use_container_width=True
        ):
            run_processing(None)

def run_processing(max_rows: Optional[int]):
    start_time = time.time()
    st.session_state.processing_results = None
    st.session_state.processing_errors = []
    st.session_state.analysis_complete = False
    
    # Use filtered data if available, otherwise use uploaded data
    data_to_process = st.session_state.get('filtered_data', st.session_state.uploaded_data)
    
    results, errors = process_reviews_batch(
        data_to_process,
        st.session_state.column_mapping,
        st.session_state.problem_categories,
        st.session_state.model_config,
        max_rows
    )
    
    st.session_state.processing_results = results
    st.session_state.processing_errors = errors
    processing_time = time.time() - start_time
    
    if results:
        st.session_state.analysis_complete = True
        
        # Auto-save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        st.session_state.output_filename = filename
        
        st.success(f"🎉 Analysis complete! Processed {len(results):,} reviews in {processing_time:.1f}s.")
        st.info(f"📁 Results saved to: {filename}")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pos_pct = (df_results['sentiment'].value_counts(normalize=True).get('Positive', 0)) * 100
            st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
        with col2:
            neg_pct = (df_results['sentiment'].value_counts(normalize=True).get('Negative', 0)) * 100
            st.metric("Negative Sentiment", f"{neg_pct:.1f}%")
        with col3:
            prob_pct = (sum(1 for p in df_results['problems_mentioned'] if p != 'None') / len(df_results)) * 100
            st.metric("Reviews w/ Problems", f"{prob_pct:.1f}%")
        with col4:
            pos_mentions_pct = (sum(1 for p in df_results['positive_mentions'] if p != 'None') / len(df_results)) * 100
            st.metric("Reviews w/ Positives", f"{pos_mentions_pct:.1f}%")
    
    if errors:
        st.error(f"⚠️ Encountered {len(errors)} errors during processing.")
        with st.expander("🔍 View Error Details"):
            for error in errors:
                st.write(f"• {error}")

def results_section():
    if not st.session_state.analysis_complete or st.session_state.processing_results is None:
        return
    
    st.markdown('<div class="section-header">📊 5. Results & Export</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state.processing_results)
    
    if df.empty:
        st.warning("No results to display.")
        return
    
    # Display results table
    with st.expander("📋 View Detailed Results", expanded=True):
        display_cols = ['date', 'product', 'rating', 'sentiment', 'problems_mentioned', 'positive_mentions']
        st.dataframe(df[display_cols], use_container_width=True, height=400)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Analysis Results (CSV)",
            csv_data,
            f"analysis_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("📊 Open Dashboard →", type="primary", use_container_width=True):
            st.info("Please run: streamlit run streamlit_dashboard_app.py")
            st.code(f"# Your results file: {st.session_state.output_filename}")

def model_configuration_sidebar():
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        
        if not BACKEND_AVAILABLE:
            st.error("❌ Backend unavailable. Check process_reviews.py")
            return
        
        st.info("Ensure your LLM host (e.g., LM Studio) is running on port 1234")
        
        available_models = fetch_available_models()
        
        if available_models:
            st.success(f"✅ Connected ({len(available_models)} models)")
            
            current_model = st.session_state.model_config.get('model_id')
            if current_model not in available_models and available_models:
                current_model = available_models[0]
                st.session_state.model_config['model_id'] = current_model
            
            try:
                current_index = available_models.index(current_model) if current_model in available_models else 0
            except ValueError:
                current_index = 0
            
            st.session_state.model_config['model_id'] = st.selectbox(
                "Select Model",
                options=available_models,
                index=current_index
            )
        else:
            st.warning("⚠️ No connection. Enter manually:")
            st.session_state.model_config['model_id'] = st.text_input(
                "Model ID",
                value=st.session_state.model_config.get('model_id', 'gemma-2-9b-it')
            )
        
        st.session_state.model_config['temperature'] = st.slider(
            "Temperature",
            0.0, 1.0,
            st.session_state.model_config.get('temperature', 0.0),
            0.01,
            help="0.0 for deterministic, consistent results"
        )
        
        if st.button("🔄 Refresh Connection"):
            st.rerun()
        
        st.divider()
        st.caption("💡 Tip: Use the dashboard app to visualize results after analysis.")

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🔬 Review Analysis Engine</div>', unsafe_allow_html=True)
    
    if not BACKEND_AVAILABLE:
        st.error(
            "❌ Critical Error: The backend file `process_reviews.py` is missing or has errors. "
            "Please ensure it's in the same directory."
        )
        return
    
    model_configuration_sidebar()
    
    # Main workflow
    if upload_csv_section():
        if column_selection_section():
            problem_categories_section()
            processing_section()
            results_section()
    else:
        st.info("👋 Welcome! Upload a CSV file containing reviews to begin analysis.")
        
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Upload CSV**: Upload your review data in CSV format
            2. **Map Columns**: Match your CSV columns to the required fields
            3. **Configure Categories**: Set up problem categories for classification
            4. **Run Analysis**: Process reviews with the LLM
            5. **Export Results**: Download results or open in the dashboard
            """)

if __name__ == "__main__":
    main()