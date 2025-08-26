# Suggested Commands

## Running the Applications

### Main Applications
```bash
# Run the review analysis app (for processing new reviews)
streamlit run streamlit_analyze_app.py

# Run the analytics dashboard (for visualizing results)
streamlit run streamlit_dashboard_app.py

# Alternative review app interface
streamlit run streamlit_review_app.py

# Command-line review processor
python review_app.py
```

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run linting (Google style with 2-space indentation)
pylint *.py

# Format code (if pyink is installed)
pyink *.py

# Check Python version
python --version
python3 --version
```

## Git Commands
```bash
# Check status
git status

# View recent commits
git log --oneline -10

# Create feature branch
git checkout -b feature/your-feature-name

# Stage and commit changes
git add .
git commit -m "Your commit message"

# Push to remote
git push origin feature/your-feature-name
```

## Testing
```bash
# No formal test suite currently exists
# Manual testing via Streamlit UI recommended

# Test the analysis workflow:
streamlit run streamlit_analyze_app.py
# Upload all_reviews.csv and process

# Test the dashboard:
streamlit run streamlit_dashboard_app.py  
# Load any analysis_results_*.csv file
```

## Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## System Utilities (macOS/Darwin)
```bash
# List files
ls -la

# Find Python files
find . -name "*.py"

# Search in files (macOS has BSD grep)
grep -r "pattern" .

# File permissions
chmod +x script.py

# Process monitoring
ps aux | grep streamlit

# Kill streamlit processes if needed
pkill -f streamlit
```

## Data Management
```bash
# List analysis results
ls -la analysis_results_*.csv

# Count reviews in CSV
wc -l all_reviews.csv

# Preview CSV file
head -n 5 all_reviews.csv
```