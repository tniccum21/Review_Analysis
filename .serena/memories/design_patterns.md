# Design Patterns and Architecture

## Application Architecture
The project follows a modular architecture with clear separation of concerns:

1. **UI Layer** (Streamlit apps)
   - User interaction and visualization
   - Session state management
   - File upload/download handling

2. **Processing Layer** (process_reviews.py)
   - Business logic for review analysis
   - LLM integration and prompt engineering
   - Data transformation

3. **Data Layer** (CSV files)
   - Input data (reviews, products)
   - Output data (analysis results)

## Key Design Patterns

### 1. Session State Pattern (Streamlit)
- Uses `st.session_state` for maintaining state across reruns
- Initialization function: `initialize_session_state()`
- Prevents data loss during UI interactions

### 2. Batch Processing Pattern
- Process reviews in configurable batch sizes
- Progress tracking with visual feedback
- Error recovery for individual review failures

### 3. Configuration Pattern
- Centralized model configuration
- Valid problem categories management
- Prompt template configuration

### 4. Graceful Degradation
- Check for backend availability
- Fallback to local processing if API unavailable
- Clear error messages for missing dependencies

## Integration Points

### LLM Integration
- Supports multiple providers (LM Studio, Ollama, cloud APIs)
- Configurable model selection
- Temperature and parameter tuning

### Data Processing Pipeline
1. CSV upload and validation
2. Column mapping (automatic or manual)
3. Batch processing with LLM
4. Results aggregation
5. Analytics generation
6. Export functionality

## Error Handling Strategy
- Try-catch blocks around external calls
- User-friendly error messages
- Logging for debugging (when enabled)
- Partial result preservation on failure

## Performance Considerations
- Batch processing to optimize API calls
- Session state caching to avoid recomputation
- Lazy loading of heavy dependencies
- Progress indicators for long operations