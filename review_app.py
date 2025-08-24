# streamlit_review_app.py

import streamlit as st
import pandas as pd
from typing import List

# Import the backend logic from your other file
# This assumes 'process_reviews.py' is in the same directory.
try:
    from process_reviews import (
        get_valid_problem_categories,
        create_extraction_config,
        process_single_review
    )
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False


def main():
    """
    Defines the main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Review Analyzer", layout="wide")
    st.title("📝 Product Review Problem Analyzer")

    # Check if the required library is installed
    if not LANGEXTRACT_AVAILABLE:
        st.error(
            "The `langextract` library is not installed. "
            "Please install it to use the app: `pip install langextract`"
        )
        st.stop() # Stop execution if library is missing

    # --- Sidebar for Configuration ---
    st.sidebar.header("⚙️ LLM Configuration")
    model_id = st.sidebar.text_input(
        "Model ID",
        value="meta-llama/Llama-3.1-8b-instruct-community",
        help="The model identifier for the OpenAI-compatible API."
    )
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls randomness. 0.0 is deterministic, 1.0 is creative."
    )

    # --- Main App Interface ---
    st.markdown("""
    Enter a product review below to automatically classify any problems mentioned.
    The app uses a local LLM (via an OpenAI-compatible API like LM Studio) to extract and categorize issues.
    """)

    # --- Review Input ---
    default_review = "Loved these shorts at first, they look great. But after four weeks the stitching came apart all along the side seam. Also, the color faded significantly after just one wash."
    review_text = st.text_area("Enter Review Text Here:", value=default_review, height=150)

    if st.button("Analyze Review", type="primary"):
        if not review_text.strip():
            st.warning("Please enter some review text to analyze.")
        else:
            with st.spinner("🧠 Analyzing review... Please wait."):
                try:
                    # 1. Get the configuration for the LLM call
                    categories = get_valid_problem_categories()
                    prompt, examples, config = create_extraction_config(
                        valid_categories=categories,
                        model_id=model_id,
                        temperature=temperature
                    )

                    # 2. Process the review text
                    problems = process_single_review(review_text, prompt, examples, config)

                    # 3. Display the results
                    st.success("✅ Analysis Complete!")

                    # Filter out the 'None' category for cleaner display if other problems exist
                    filtered_problems = [p for p in problems if p.get("category") != "None"]

                    if not filtered_problems and problems[0].get("category") == "None":
                         st.info("No specific problems were identified in this review. Looks like a positive one!")
                    else:
                        st.subheader("Detected Problems:")
                        df = pd.DataFrame(filtered_problems)
                        # Reorder columns for better readability
                        df = df[['category', 'description', 'extraction_text']]
                        st.dataframe(df, use_container_width=True)

                except (ConnectionError, RuntimeError, ValueError) as e:
                    st.error(f"An error occurred: {e}")
                    st.warning(
                        "Please ensure your local LLM server (e.g., LM Studio, Ollama) is running and accessible."
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()