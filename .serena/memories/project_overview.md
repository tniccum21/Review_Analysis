# LangExtract Project Overview

## Purpose
LangExtract is a Python library that leverages Large Language Models (LLMs) to extract structured information from unstructured text documents. This repository has been transformed into a Review Analysis Dashboard application built on top of the LangExtract library.

## Key Applications
1. **streamlit_analyze_app.py** - Main review analysis application
   - Upload and process review CSV files
   - LLM-powered sentiment and problem detection
   - Batch processing with progress tracking
   - Export analyzed results

2. **streamlit_dashboard_app.py** - Analytics dashboard application
   - Load and visualize analysis results
   - Interactive filtering and metrics
   - Multiple chart types (sentiment, ratings, problems)
   - Word cloud visualization
   - Detailed review samples

## Key Features
- **Review Analysis**: Analyzes customer reviews for sentiment and problem categorization
- **Interactive Dashboard**: Streamlit-based web interface for review processing and visualization
- **LLM Integration**: Supports both local (LM Studio/Ollama) and cloud-based models
- **Batch Processing**: Process multiple reviews with progress tracking
- **Analytics & Visualization**: Comprehensive charts and insights from processed reviews
- **CSV Export**: Download analyzed results for further analysis
- **Product Metadata Integration**: Links reviews to product information

## Tech Stack
- **Language**: Python 3.x
- **Web Framework**: Streamlit (>=1.28.0)
- **Data Processing**: pandas (>=2.0.0), numpy (>=1.24.0)
- **LLM Library**: langextract (core library)
- **Visualization**: matplotlib (>=3.5.0), wordcloud (>=1.9.0)
- **HTTP Client**: requests (>=2.31.0)
- **Date Handling**: python-dateutil (>=2.8.0)

## Project Type
This is a customized fork of the Google LangExtract library, transformed into a specialized review analysis application with two main Streamlit UIs:
- Analysis app for processing reviews
- Dashboard app for visualizing results