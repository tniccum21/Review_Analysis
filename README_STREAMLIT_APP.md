# 📊 Streamlit Review Analysis App

A powerful, user-friendly interface for analyzing product reviews using AI-powered problem classification and sentiment analysis. Built on top of the `langextract` library and `process_reviews.py`.

## 🚀 Features

- **📁 Easy CSV Upload**: Drag & drop or browse to upload review data
- **🎯 Flexible Column Mapping**: Select which columns contain product IDs, ratings, dates, titles, and review text
- **🏷️ Customizable Problem Categories**: Edit and customize problem classification categories
- **🧪 Test Mode**: Run analysis on a random sample of reviews before processing the full dataset
- **📊 Progress Tracking**: Real-time progress bars during processing
- **💾 Export Results**: Download processed results as CSV with sentiment and problem classifications
- **🤖 LM Studio Integration**: Uses local LLM via LM Studio for privacy and control

## 📋 Prerequisites

1. **Python 3.8+**
2. **LM Studio** (for local LLM inference)
3. **Virtual Environment** (recommended)

## 🛠️ Installation & Setup

### Step 1: Clone and Setup Environment

```bash
# Navigate to your langextract directory
cd /Users/thomasniccum/Documents/langextract

# Create and activate virtual environment (if not already done)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: LM Studio Setup

#### Download and Install LM Studio

1. **Download LM Studio**
   - Visit [https://lmstudio.ai/](https://lmstudio.ai/)
   - Download the installer for your platform (Windows, macOS, or Linux)
   - Follow the installation instructions

2. **Download a Compatible Model**
   - Open LM Studio
   - Go to the **"Models"** tab
   - Search for and download a supported model. Recommended models:
     - `llama-3.1-8b-instruct` (default, good balance of speed/quality)
     - `mistral-7b-instruct-v0.2` (faster, smaller)
     - `llama-3.1-70b-instruct` (highest quality, requires more RAM)

#### Configure LM Studio Server

3. **Start the API Server**
   - In LM Studio, navigate to the **"Server"** tab
   - Click **"Start Server"** or toggle **"Enable API server"**
   - Note the server URL (default: `http://localhost:1234`)
   - Ensure the model is loaded and ready

4. **Verify Server Connection**
   ```bash
   # Test the API endpoint
   curl -X POST http://localhost:1234/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "llama-3.1-8b-instruct",
       "prompt": "Hello, world!",
       "max_tokens": 50
     }'
   ```

#### Environment Configuration (Optional)

5. **Set Environment Variables**
   ```bash
   # Set model ID (optional, can be configured in the app)
   export MODEL_ID="llama-3.1-8b-instruct"
   
   # Set temperature (optional, can be configured in the app)
   export TEMPERATURE="0.1"
   ```

   Or create a `.env` file:
   ```bash
   MODEL_ID=llama-3.1-8b-instruct
   TEMPERATURE=0.1
   ```

## 🚀 Running the App

### Start the Streamlit App

```bash
# Make sure you're in the langextract directory and virtual environment is active
cd /Users/thomasniccum/Documents/langextract
source .venv/bin/activate

# Start the Streamlit app
streamlit run streamlit_review_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the App

1. **📁 Upload CSV File**
   - Drag and drop or browse to select your CSV file containing reviews
   - The app will display a preview of your data

2. **🎯 Select Columns**
   - Map your CSV columns to the required fields:
     - Product Designator (SKU, Product ID, etc.)
     - Numeric Rating Column
     - Review Date Column
     - Review Title Column
     - Review Text Column

3. **🏷️ Configure Problem Categories**
   - Review and customize the problem categories
   - Add, remove, or modify categories as needed
   - Default categories include: Fit, Comfort, Size, Material, Color, Durability, etc.

4. **⚙️ Configure Model Settings** (in sidebar)
   - Set the Model ID to match your LM Studio model
   - Adjust temperature for response randomness
   - Test the connection to LM Studio

5. **🧪 Run Analysis**
   - Choose "Test Run" with a sample size for quick testing
   - Or "Run All" to process the entire dataset
   - Monitor progress with the real-time progress bar

6. **💾 Download Results**
   - Download the processed results as a CSV file
   - Results include: date, product, rating, sentiment, problems mentioned

## 📊 Output Format

The app generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `date` | Review date from your original data |
| `product` | Product identifier from your original data |
| `rating` | Numeric rating from your original data |
| `sentiment` | Overall review sentiment: positive, neutral, negative |
| `problems_mentioned` | List of problems identified and categorized |

## 🔧 Troubleshooting

### Common Issues

1. **"Connection Error" when running analysis**
   - ✅ Ensure LM Studio is running and the server is started
   - ✅ Check that the correct model is loaded in LM Studio
   - ✅ Verify the Model ID in the app matches your LM Studio model
   - ✅ Test the connection using the "Test LM Studio Connection" button

2. **"ModuleNotFoundError" for langextract**
   - ✅ Ensure you're in the correct virtual environment
   - ✅ Install langextract in editable mode: `pip install -e .`
   - ✅ Check that you're in the langextract project directory

3. **CSV Upload Issues**
   - ✅ Ensure your CSV file is properly formatted
   - ✅ Check for special characters or encoding issues
   - ✅ Verify column headers are present

4. **Memory Issues with Large Datasets**
   - ✅ Use "Test Run" mode first with a smaller sample
   - ✅ Consider processing data in batches
   - ✅ Ensure sufficient RAM for your chosen LM Studio model

### Performance Tips

- **Model Selection**: Smaller models (7B parameters) are faster but less accurate than larger models (70B parameters)
- **Temperature Setting**: Lower values (0.1-0.3) give more consistent results
- **Batch Processing**: For very large datasets, consider processing in smaller batches
- **Hardware**: More RAM and CPU cores will improve processing speed

## 📝 Example CSV Format

Your input CSV should have columns like:

```csv
product_id,rating,review_date,review_title,review_text
SKU123,4,2024-01-15,"Great product!","I love this product. It fits perfectly and is very comfortable."
SKU124,2,2024-01-16,"Poor quality","The material started falling apart after just one week of use."
```

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your LM Studio setup and model configuration
3. Ensure all dependencies are properly installed
4. Check the Streamlit app logs for detailed error messages

## 📄 License

This project follows the same license as the langextract library (Apache License 2.0).

---

**Happy Analyzing! 🎉**
