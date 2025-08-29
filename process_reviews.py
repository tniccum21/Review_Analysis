"""
Process reviews and classify problems into categories using a direct LLM API call.
Backend logic for the Streamlit Review Analysis App.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- NEW: Helper function to robustly find JSON in a string ---
def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Finds and extracts the first JSON object string from a text block.
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

def get_valid_problem_categories():
    """Return the default list of valid problem categories."""
    return [
        "Fit", "Comfort", "Size", "Material", "Color", "Durability", "Brand",
        "Design", "Price", "Breathability", "Shipping", "Packaging"
    ]

def create_extraction_config(valid_categories: List[str], model_id: str, temperature: float) -> str:
    """Creates a new, unified prompt that asks for sentiment, problems, and positive mentions."""
    
    categories_list = valid_categories.copy()
    if "Other" not in categories_list:
        categories_list.append("Other")

    categories_str = ", ".join(f'"{cat}"' for cat in categories_list)

    prompt = f"""You are a review analysis expert. Your task is to analyze a customer review and return a single, valid JSON object containing the review's sentiment, any identified problems, and positive mentions.

TASK INSTRUCTIONS:
1.  **Determine Sentiment:** First, analyze the review text and star rating to determine the overall sentiment. The sentiment MUST be one of: "Positive", "Negative", or "Neutral".
2.  **Extract Problems:** Identify all topics mentioned as problems or complaints in the review.
3.  **Extract Positives:** Identify all topics mentioned positively or praised in the review.
4.  **Categorize Both:** Match each problem AND positive mention to a category from the VALID CATEGORIES list. Use "Other" for items that don't fit. If no problems/positives are found, the respective arrays must be empty.

VALID CATEGORIES: [{categories_str}]

REQUIRED JSON OUTPUT STRUCTURE:
{{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "problems": ["Category1", "Category2", ...],
  "positives": ["Category1", "Category2", ...]
}}

You MUST return ONLY the valid JSON object. Do not include any other text, explanations, or markdown.
"""
    return prompt

def analyze_review_with_llm(review_text: str, rating: Any, prompt: str, model_config: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Calls the LLM with both review text and rating to get sentiment, problems, and positives in a single call.
    """
    base_url = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
    user_content = f"Star Rating: {rating}/5\nReview Text: \"{review_text}\""
    
    # Get timeout from environment variable
    timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_config['model_id'],
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 500,
                "temperature": model_config['temperature'],
                # --- MODIFIED: Removed "response_format" for wider compatibility ---
            },
            timeout=timeout
        )
        response.raise_for_status() # This will raise an HTTPError for non-2xx responses

        response_text = response.json()['choices'][0]['message']['content']
        
        # --- MODIFIED: Use the robust helper function to find and parse JSON ---
        json_string = _extract_json_from_text(response_text)
        
        if not json_string:
            raise ValueError("LLM response did not contain a detectable JSON object.")

        parsed_json = json.loads(json_string)
        sentiment = parsed_json.get("sentiment", "Neutral")
        problems = parsed_json.get("problems", [])
        positives = parsed_json.get("positives", [])

        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"
        if not isinstance(problems, list):
            problems = []
        if not isinstance(positives, list):
            positives = []

        return sentiment, problems, positives

    except requests.exceptions.HTTPError as e:
        # Re-raise with a more user-friendly message, including the error details
        raise ConnectionError(f"LLM API request failed: {e}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to LLM provider: {e}")
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM JSON response: {e}")