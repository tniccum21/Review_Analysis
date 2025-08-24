# 🤖 Review Analysis Application

A powerful AI-powered application for analyzing product reviews, extracting sentiment, and identifying problem categories using local LLMs through LM Studio.

## ✨ Features

- **📊 Interactive Web Interface**: Beautiful Streamlit app with real-time analytics
- **🤖 AI-Powered Analysis**: Sentiment classification and problem categorization
- **🎯 Smart Column Mapping**: Automatic CSV column detection with AI
- **📈 Analytics Dashboard**: Interactive visualizations with time-series analysis
- **🏷️ Customizable Categories**: Define your own problem classification system
- **🧪 Test Mode**: Validate on samples before full processing
- **💾 Export Results**: Download analyzed data as CSV
- **🔒 Privacy-First**: Uses local LLMs - your data never leaves your machine

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LM Studio (for local LLM inference)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd langextract
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup LM Studio**
   - Download from [lmstudio.ai](https://lmstudio.ai/)
   - Install and download a model (e.g., `gemma2:2b`)
   - Start the API server on port 1234

5. **Run the app**
   ```bash
   streamlit run streamlit_review_app.py
   ```

## 📱 Application Overview

### Main Components

```
├── streamlit_review_app.py    # Interactive web interface
├── process_reviews.py         # Core analysis logic
├── review_app.py             # CLI version
└── requirements.txt          # Python dependencies
```

### Key Features

#### 1. CSV Upload & Preview
- Drag-and-drop file upload
- Automatic encoding detection
- Data preview with statistics

#### 2. Smart Column Mapping
- AI-powered automatic field detection
- Manual override options
- Validation checks

#### 3. Problem Categories
- Default categories (Fit, Comfort, Size, etc.)
- Fully customizable
- Category management interface

#### 4. Processing Options
- Test mode for sampling
- Full dataset processing
- Real-time progress tracking

#### 5. Analytics Dashboard
- Average rating trends
- Sentiment distribution
- Problem frequency analysis
- Time-series visualizations
- Product filtering

## 📊 Usage Guide

### Step 1: Prepare Your Data

Your CSV should have columns for:
- Product identifier (SKU, ID, etc.)
- Numeric rating (1-5)
- Review date
- Review text
- Review title (optional)

Example:
```csv
product_id,rating,date,title,review_text
SKU001,4,2024-01-15,"Great product","Love the quality and comfort"
SKU002,2,2024-01-16,"Disappointed","Size runs small and material feels cheap"
```

### Step 2: Configure LM Studio

1. Open LM Studio
2. Download a model (recommended: `gemma2:2b` or `llama-3.1-8b`)
3. Start the API server
4. Verify connection at `http://localhost:1234`

### Step 3: Run Analysis

1. Upload your CSV file
2. Map columns (or use AI auto-mapping)
3. Configure problem categories
4. Run test on sample (recommended)
5. Process full dataset
6. Export results

### Step 4: Analyze Results

The dashboard provides:
- **Sentiment Analysis**: Positive/Negative/Neutral distribution
- **Problem Identification**: Categorized issues from reviews
- **Trend Analysis**: Time-based patterns
- **Rating Correlation**: Sentiment vs rating analysis

## 🛠️ API Reference

### Core Functions

#### `analyze_review_with_llm()`
Analyzes a single review for sentiment and problems.

```python
sentiment, problems = analyze_review_with_llm(
    review_text="Product is great but shipping was slow",
    rating=4,
    prompt=prompt_config,
    model_config={'model_id': 'gemma2:2b', 'temperature': 0.0}
)
# Returns: ("Positive", ["Shipping"])
```

#### `process_reviews_batch()`
Processes multiple reviews efficiently.

```python
results, errors = process_reviews_batch(
    df=dataframe,
    column_mapping={'product': 'sku', 'rating': 'stars', ...},
    problem_categories=['Fit', 'Quality', 'Shipping'],
    model_config={'model_id': 'gemma2:2b', 'temperature': 0.0}
)
```

## 📈 Output Format

Results CSV includes:
- `date`: Review date
- `product`: Product identifier
- `rating`: Original rating
- `sentiment`: Classified sentiment
- `problems_mentioned`: Identified issues
- `original_text`: Original review

## 🔧 Configuration

### Environment Variables

```bash
# LM Studio connection
LM_STUDIO_HOST=http://localhost:1234

# Model settings
MODEL_ID=gemma2:2b
TEMPERATURE=0.0
```

### Problem Categories

Default categories:
- Fit, Comfort, Size
- Material, Color, Durability
- Brand, Design, Price
- Breathability, Shipping, Packaging

Customize in the app or via code:
```python
custom_categories = ["Battery", "Screen", "Performance", "Support"]
```

## 🐛 Troubleshooting

### Common Issues

**LM Studio Connection Error**
- Ensure LM Studio is running
- Check API server is started
- Verify model is loaded
- Test: `curl http://localhost:1234/v1/models`

**Memory Issues**
- Use smaller models (2B-7B parameters)
- Process in smaller batches
- Reduce max_tokens setting

**Slow Processing**
- Lower temperature to 0.0
- Use faster models
- Enable GPU acceleration in LM Studio

## 📊 Example Results

### Sentiment Distribution
```
Positive: 65%
Neutral: 20%
Negative: 15%
```

### Top Problem Categories
```
1. Fit (23%)
2. Material (18%)
3. Shipping (15%)
4. Price (12%)
5. Comfort (10%)
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

Apache License 2.0 - See LICENSE file for details.

## 🔗 Resources

- [LM Studio](https://lmstudio.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Project Repository](https://github.com/your-repo)

---

**Built with ❤️ for efficient review analysis**