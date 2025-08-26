# Code Style and Conventions

## Python Style
- **Indentation**: 2 spaces (Google style, enforced by pyink formatter)
- **Line Length**: 80 characters maximum
- **Naming Conventions**:
  - Functions: snake_case (e.g., `analyze_review_with_llm`)
  - Classes: PascalCase
  - Constants: UPPER_CASE (e.g., `BACKEND_AVAILABLE`)
  - Private functions: Leading underscore (e.g., `_extract_json_from_text`)

## Import Style
- Standard library imports first
- Third-party imports second
- Local imports last
- Each group separated by blank line

## Documentation
- Docstrings for public functions and classes
- Type hints where applicable
- Comments for complex logic
- Markdown documentation for user-facing features

## Error Handling
- Try-except blocks for external API calls
- Graceful degradation when dependencies unavailable
- User-friendly error messages in Streamlit UI

## File Organization
- Main application logic in root directory
- Data files alongside code
- Documentation in markdown files
- GitHub workflows in .github directory

## Linting Configuration
- Uses pylint with Google-style configuration
- Disabled checks for line length (handled by formatter)
- Allows TODO/FIXME comments
- Ignores certain complexity metrics for practical reasons