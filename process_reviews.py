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

    prompt = f"""You are a review analysis expert. Analyze the review and return ONLY a valid JSON object.

VALID CATEGORIES: [{categories_str}]

REQUIRED JSON FORMAT:
{{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "problems": ["Category1", "Category2", ...],
  "positives": ["Category1", "Category2", ...]
}}

RULES:
1. sentiment must be exactly one of: "Positive", "Negative", or "Neutral"
2. problems and positives must be arrays (can be empty [])
3. Each category in arrays must be from VALID CATEGORIES
4. Return ONLY the JSON object, no markdown, no explanations
"""
    return prompt

def analyze_review_with_llm(review_text: str, rating: Any, prompt: str, model_config: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Calls the LLM with both review text and rating to get sentiment, problems, and positives in a single call.
    More robust error handling to avoid skipping reviews due to JSON parsing issues.
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
                "max_tokens": 1500,  # Increased for better completion
                "temperature": model_config['temperature'],
            },
            timeout=timeout
        )
        response.raise_for_status()

        response_data = response.json()
        
        # Get the content from the response
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            # Return safe defaults if no choices
            print(f"Warning: No choices in LLM response for review")
            return "Neutral", [], []
            
        message = response_data['choices'][0]['message']
        response_text = message.get('content', '')
        
        # If content is empty, check for reasoning field (some models use this)
        if not response_text and 'reasoning' in message:
            response_text = message.get('reasoning', '')
        
        if not response_text:
            # Return safe defaults if no content
            print(f"Warning: Empty LLM response for review")
            return "Neutral", [], []
        
        # Try to extract JSON from the response text
        json_string = _extract_json_from_text(response_text)
        
        if not json_string:
            # If no JSON found, try to parse the entire response as JSON
            # (in case it's pure JSON without any wrapper text)
            try:
                parsed_json = json.loads(response_text.strip())
            except:
                # Try to extract sentiment from text as fallback
                response_lower = response_text.lower()
                if "positive" in response_lower:
                    return "Positive", [], []
                elif "negative" in response_lower:
                    return "Negative", [], []
                else:
                    return "Neutral", [], []
        else:
            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error: {e}")
                # Return safe defaults
                return "Neutral", [], []

        # Extract values with safe defaults
        sentiment = parsed_json.get("sentiment", "Neutral")
        problems = parsed_json.get("problems", [])
        positives = parsed_json.get("positives", [])

        # Validate sentiment
        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"
        
        # Ensure problems and positives are lists
        if not isinstance(problems, list):
            problems = []
        if not isinstance(positives, list):
            positives = []

        return sentiment, problems, positives

    except requests.exceptions.HTTPError as e:
        # Log the error but return safe defaults instead of raising
        print(f"Warning: LLM API HTTP error: {e}")
        return "Neutral", [], []
    except requests.exceptions.RequestException as e:
        # Connection errors - return safe defaults
        print(f"Warning: LLM connection error: {e}")
        return "Neutral", [], []
    except Exception as e:
        # Any other error - return safe defaults
        print(f"Warning: Unexpected error in LLM analysis: {e}")
        return "Neutral", [], []