# Code Structure

## Main Applications

### 1. **streamlit_analyze_app.py** - Review Analysis Application
Key functions:
- `initialize_session_state()`: Session state management
- `fetch_available_models()`: Get available LLM models
- `upload_csv_section()`: Handle CSV file uploads
- `match_csv_fields_to_analysis_fields()`: Auto-map CSV columns
- `column_selection_section()`: Manual column mapping UI
- `problem_categories_section()`: Configure problem categories
- `process_reviews_batch()`: Batch process reviews with LLM
- `run_processing()`: Main processing orchestrator
- `results_section()`: Display and download results
- `model_configuration_sidebar()`: LLM model configuration

### 2. **streamlit_dashboard_app.py** - Analytics Dashboard
Key functions:
- `initialize_session_state()`: Session state management
- `load_analysis_results()`: Load processed CSV results
- `load_product_metadata()`: Load product information
- `display_summary_metrics()`: Show key metrics
- `apply_filters()`: Filter data by various criteria
- `create_filter_section()`: Interactive filtering UI
- Charts:
  - `create_mentions_chart()`: Product mentions over time
  - `create_sentiment_distribution_chart()`: Sentiment breakdown
  - `create_rating_distribution_chart()`: Rating distribution
  - `create_sentiment_area_chart()`: Sentiment trends
  - `create_problem_categories_chart()`: Problem frequency
  - `create_positive_categories_chart()`: Positive aspects
- `create_wordcloud_visualization()`: Word cloud from reviews
- `display_detailed_results()`: Detailed review table
- `create_review_samples()`: Sample reviews display

### 3. **process_reviews.py** - Backend Processing Logic
- `create_extraction_config()`: Generate LLM prompts
- `analyze_review_with_llm()`: Analyze single review
- `get_valid_problem_categories()`: Default problem categories
- `_extract_json_from_text()`: JSON extraction utilities

### 4. **streamlit_review_app.py** - Alternative Review App
- Similar functionality to analyze app
- Different UI layout and features

## Data Files
- **all_reviews.csv**: Sample review data
- **products.csv**: Product metadata
- **analysis_results_*.csv**: Processed results from various runs

## Configuration
- **requirements.txt**: Python dependencies
- **.pylintrc**: Linting configuration (Google style with 2-space indentation)
- **.gitignore**: Standard Python project exclusions

## Documentation
- **README.md**: Main LangExtract library documentation
- **README_STREAMLIT_APP.md**: Streamlit app specific documentation
- **PROJECT_DOCUMENTATION.md**: Comprehensive project documentation
- **API_REFERENCE.md**: API documentation for review processing functions

## CI/CD
- **.github/workflows/**: GitHub Actions for CI, PR validation, and publishing
- Multiple workflow files for comprehensive PR checks and automation