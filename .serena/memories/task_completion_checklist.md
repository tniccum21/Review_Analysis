# Task Completion Checklist

When completing a development task in this project, ensure you:

## Code Quality
1. **Linting**: Run pylint to check code quality
   ```bash
   pylint *.py
   ```
   - Fix any critical errors
   - Review warnings for potential issues

2. **Code Style**: Ensure 2-space indentation is maintained
   - Functions use snake_case
   - Classes use PascalCase
   - Constants use UPPER_CASE

## Testing
1. **Manual Testing**: Since no formal test suite exists
   - Run the Streamlit app: `streamlit run streamlit_review_app.py`
   - Test with sample CSV files (all_reviews.csv)
   - Verify all features work:
     - File upload
     - Column mapping
     - Review processing
     - Results download

2. **Error Handling**: Test edge cases
   - Empty CSV files
   - Invalid data formats
   - Missing required columns
   - API connection failures

## Documentation
1. **Update Documentation** if needed:
   - README.md for library-level changes
   - README_STREAMLIT_APP.md for UI changes
   - API_REFERENCE.md for function signature changes

2. **Code Comments**: Add/update as needed
   - Document complex logic
   - Explain non-obvious decisions

## Version Control
1. **Git Commit**:
   - Stage changes: `git add .`
   - Meaningful commit message
   - Reference any issues if applicable

2. **Branch Management**:
   - Work on feature branches
   - Keep commits atomic and focused
   - Ensure main branch remains stable

## Verification
1. **Dependencies**: Ensure requirements.txt is updated if new packages added
2. **File Cleanup**: Remove any temporary files or debug outputs
3. **Console Output**: Remove or comment out debug print statements
4. **Data Files**: Don't commit large analysis result files unless necessary