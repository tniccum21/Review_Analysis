# Review Analysis App API Reference

## 📋 Table of Contents

1. [Review Processing API](#review-processing-api)
2. [Streamlit App API](#streamlit-app-api)
3. [Data Models](#data-models)
4. [Configuration](#configuration)
5. [Error Handling](#error-handling)

---

## 🔷 Review Processing API

### Core Functions

#### `create_extraction_config()`

Generate LLM prompt configuration for review analysis.

```python
def create_extraction_config(
    valid_categories: List[str],
    model_id: str,
    temperature: float
) -> str
```

**Parameters:**
- `valid_categories`: List of problem categories to identify
- `model_id`: LLM model identifier
- `temperature`: Model temperature (0.0-1.0)

**Returns:**
- `str`: Formatted prompt for LLM

**Example:**
```python
prompt = create_extraction_config(
    valid_categories=["Fit", "Comfort", "Size", "Material"],
    model_id="gpt-oss-20b",  # From .env or auto-selected
    temperature=0.0
)
```

---

#### `analyze_review_with_llm()`

Analyze single review for sentiment and problems.

```python
def analyze_review_with_llm(
    review_text: str,
    rating: Any,
    prompt: str,
    model_config: Dict[str, Any]
) -> Tuple[str, List[str], List[str]]
```

**Parameters:**
- `review_text`: Review content to analyze
- `rating`: Numeric rating (1-5)
- `prompt`: LLM prompt from `create_extraction_config()`
- `model_config`: Model configuration dictionary

**Returns:**
- `Tuple[str, List[str], List[str]]`: (sentiment, problems_list, positives_list)
  - `sentiment`: "Positive", "Negative", or "Neutral"
  - `problems_list`: List of identified problem categories
  - `positives_list`: List of identified positive features

**Example:**
```python
sentiment, problems, positives = analyze_review_with_llm(
    review_text="Product is comfortable but expensive",
    rating=3,
    prompt=prompt,
    model_config={
        "model_id": "gpt-oss-20b",  # Or auto-selected from LM Studio
        "temperature": 0.0
    }
)
# Returns: ("Neutral", ["Price"], ["Comfort"])
```

---

#### `get_valid_problem_categories()`

Get default problem categories.

```python
def get_valid_problem_categories() -> List[str]
```

**Returns:**
- `List[str]`: Default problem categories

**Default Categories:**
- Fit
- Comfort
- Size
- Material
- Color
- Durability
- Brand
- Design
- Price
- Breathability
- Shipping
- Packaging

---

### Helper Functions

#### `_extract_json_from_text()`

Extract JSON object from LLM response text.

```python
def _extract_json_from_text(text: str) -> Optional[str]
```

**Parameters:**
- `text`: LLM response text

**Returns:**
- `Optional[str]`: Extracted JSON string or None

---

## 🔷 Streamlit App API

### Session Management

#### `initialize_session_state()`

Initialize Streamlit session state variables.

```python
def initialize_session_state() -> None
```

**Session Variables:**
- `uploaded_data`: DataFrame of uploaded CSV
- `column_mapping`: Column name mappings
- `problem_categories`: List of categories
- `processing_results`: Analysis results
- `processing_errors`: Error messages
- `model_config`: LLM configuration
- `is_processing`: Processing flag

---

### Data Processing

#### `process_reviews_batch()`

Process multiple reviews in batch.

```python
def process_reviews_batch(
    df: pd.DataFrame,
    column_mapping: dict,
    problem_categories: list,
    model_config: dict,
    max_rows: Optional[int] = None
) -> Tuple[List[Dict], List[str]]
```

**Parameters:**
- `df`: Input DataFrame with reviews
- `column_mapping`: Maps DataFrame columns to required fields
- `problem_categories`: List of problem categories
- `model_config`: LLM configuration
- `max_rows`: Maximum rows to process (None for all)

**Returns:**
- `Tuple[List[Dict], List[str]]`: (results, errors)
  - `results`: List of processed review dictionaries
  - `errors`: List of error messages

**Result Dictionary Structure:**
```python
{
    'date': '2024-01-15',
    'product': 'SKU123',
    'rating': 4,
    'sentiment': 'Positive',
    'problems_mentioned': 'Fit; Comfort',
    'original_text': 'Review text...'
}
```

---

### Column Mapping

#### `match_csv_fields_to_analysis_fields()`

AI-powered CSV column mapping.

```python
def match_csv_fields_to_analysis_fields(
    csv_columns: List[str],
    model_config: dict,
    sample_df: pd.DataFrame = None
) -> Dict[str, str]
```

**Parameters:**
- `csv_columns`: List of CSV column names
- `model_config`: LLM configuration
- `sample_df`: Sample data for context

**Returns:**
- `Dict[str, str]`: Column mappings

**Mapping Structure:**
```python
{
    'product': 'product_id',
    'rating': 'stars',
    'date': 'review_date',
    'title': 'review_title',
    'text': 'review_content'
}
```

---

### Visualization

#### `create_analytics_dashboard()`

Create interactive analytics dashboard.

```python
def create_analytics_dashboard(df: pd.DataFrame) -> None
```

**Parameters:**
- `df`: DataFrame with processed results

**Dashboard Components:**
- Average rating over time with trendline
- Sentiment trend charts (count and percentage)
- Problem category distribution (count and percentage)
- Product filtering
- Monthly aggregations

---

### UI Section Functions

#### `upload_csv_section()`

Handle CSV file upload and preview.

```python
def upload_csv_section() -> bool
```

**Returns:**
- `bool`: True if file successfully uploaded

---

#### `column_selection_section()`

Handle column mapping interface.

```python
def column_selection_section() -> bool
```

**Returns:**
- `bool`: True if all required columns mapped

---

#### `problem_categories_section()`

Manage problem category configuration.

```python
def problem_categories_section() -> None
```

---

#### `processing_section()`

Handle review processing execution.

```python
def processing_section() -> None
```

---

#### `download_results_section()`

Display results and download options.

```python
def download_results_section() -> None
```

---

### Model Configuration

#### `fetch_available_models()`

Fetch available models from LM Studio.

```python
def fetch_available_models() -> List[str]
```

**Returns:**
- `List[str]`: Available model IDs

---

#### `model_configuration_sidebar()`

Manage LLM configuration in sidebar.

```python
def model_configuration_sidebar() -> None
```

---

## 🔷 Data Models

### Review Data Structure

#### Input CSV Format

Required columns (names can vary):
- Product identifier
- Numeric rating (1-5)
- Review date
- Review text
- Review title (optional)

#### Output CSV Format

Standard columns:
- `date`: Review date
- `product`: Product identifier
- `rating`: Original rating
- `sentiment`: Sentiment classification (Positive/Negative/Neutral)
- `problems_mentioned`: Semicolon-separated categories or "None"
- `positive_mentions`: Semicolon-separated positive features or "None"
- `original_text`: Original review text

---

## 🔷 Configuration

### Environment Variables

```bash
# LM Studio host URL
LM_STUDIO_HOST=http://169.254.11.42:1234

# Request timeouts
LLM_REQUEST_TIMEOUT=3000
LLM_MODEL_FETCH_TIMEOUT=10

# Review Analysis App configuration
ANALYZE_LLM_MODEL_ID=gpt-oss-20b  # Or leave empty for auto-select
ANALYZE_LLM_TEMPERATURE=0.0

# AI Dashboard App configuration  
AI_LLM_MODEL_ID=gpt-oss-120b  # Or leave empty for auto-select
AI_LLM_TEMPERATURE=0.1
```

### Model Configuration Object

```python
model_config = {
    'model_id': 'gpt-oss-20b',  # From .env or auto-selected
    'temperature': 0.0,
    'max_tokens': 1500,  # Increased for better completions
    'timeout': 300  # From LLM_REQUEST_TIMEOUT env var
}
```

### LM Studio API Format

Request format for LM Studio:
```python
{
    "model": "gpt-oss-20b",  # Or any loaded model
    "messages": [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "content"}
    ],
    "max_tokens": 1500,  # Higher for complete responses
    "temperature": 0.0
}
```

**Model Auto-Selection:**
If no model is specified in .env, the system queries `/v1/models` endpoint and auto-selects the first available model.

---

## 🔷 Error Handling

### Exception Types

#### `ConnectionError`

Raised when LLM connection fails.

```python
try:
    result = analyze_review_with_llm(...)
except ConnectionError as e:
    print(f"LLM connection failed: {e}")
```

#### `ValueError`

Raised for invalid data or parsing errors.

```python
try:
    sentiment, problems = analyze_review_with_llm(...)
except ValueError as e:
    print(f"Data parsing error: {e}")
```

#### Error Recovery

**IMPORTANT:** The `analyze_review_with_llm()` function **never raises exceptions**. It always returns safe defaults to ensure 100% review processing.

```python
# This function always succeeds
sentiment, problems, positives = analyze_review_with_llm(
    review_text="Any review text",
    rating=3,
    prompt=prompt,
    model_config=config
)
# On any error, returns: ("Neutral", [], [])
```

### Error Recovery Strategies

1. **Retry Logic**
   ```python
   max_retries = 3
   for attempt in range(max_retries):
       try:
           result = analyze_review_with_llm(...)
           break
       except ConnectionError:
           if attempt == max_retries - 1:
               raise
           time.sleep(2 ** attempt)
   ```

2. **Built-in Error Recovery**
   ```python
   # No try/except needed - function handles all errors internally
   sentiment, problems, positives = analyze_review_with_llm(...)
   # Always returns valid data, never skips reviews
   ```

3. **Batch Error Handling**
   ```python
   results, errors = [], []
   for review in reviews:
       try:
           result = process_review(review)
           results.append(result)
       except Exception as e:
           errors.append(f"Review {review['id']}: {e}")
           continue
   ```

---

## 📚 Usage Examples

### Complete Review Processing Pipeline

```python
import pandas as pd
from process_reviews import (
    create_extraction_config,
    analyze_review_with_llm,
    get_valid_problem_categories
)

# Load data
df = pd.read_csv('reviews.csv')

# Configure extraction
categories = get_valid_problem_categories()
# Model ID from .env or auto-selected from LM Studio
model_id = os.getenv("ANALYZE_LLM_MODEL_ID", "")
if not model_id:
    # Auto-select from available models
    models = fetch_available_models()
    model_id = models[0] if models else "gpt-oss-20b"

prompt = create_extraction_config(
    valid_categories=categories,
    model_id=model_id,
    temperature=0.0
)

# Process reviews
results = []
for _, row in df.iterrows():
    try:
        sentiment, problems, positives = analyze_review_with_llm(
            review_text=row['review_text'],
            rating=row['rating'],
            prompt=prompt,
            model_config={'model_id': 'gpt-oss-20b', 'temperature': 0.0}
        )
        results.append({
            'product': row['product_id'],
            'sentiment': sentiment,
            'problems': '; '.join(problems) if problems else 'None',
            'positives': '; '.join(positives) if positives else 'None'
        })
    except Exception as e:
        print(f"Error: {e}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('analysis_results.csv', index=False)
```

### Streamlit App Usage

```python
# Start the app
streamlit run streamlit_review_app.py

# The app provides:
# 1. CSV upload interface
# 2. Auto column mapping
# 3. Category configuration  
# 4. Batch processing
# 5. Results visualization
# 6. CSV export
```

### Custom Problem Categories

```python
# Define custom categories for specific domain
custom_categories = [
    "Battery Life",
    "Screen Quality", 
    "Performance",
    "Build Quality",
    "Software",
    "Customer Service"
]

prompt = create_extraction_config(
    valid_categories=custom_categories,
    model_id=os.getenv("ANALYZE_LLM_MODEL_ID", "gpt-oss-20b"),
    temperature=0.0
)
```

---

## 🔗 Additional Resources

- [Streamlit API Documentation](https://docs.streamlit.io/library/api-reference)
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)

---

**API Version:** 1.0.0  
**Last Updated:** 2024