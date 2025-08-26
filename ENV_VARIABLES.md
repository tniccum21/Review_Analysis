# Environment Variables Documentation

## Overview
Both Streamlit applications support environment variables to set default LLM configuration values. Each app can use **app-specific variables** for different models and settings, with fallback to generic variables for convenience.

## Supported Environment Variables

### App-Specific Configuration (Recommended)

#### Review Analysis App (`streamlit_analyze_app.py`)
For sentiment extraction and problem categorization - typically needs smaller, faster models:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `ANALYZE_LLM_MODEL_ID` | Model for review analysis | `"gemma-2-9b-it"` | `"llama-3-8b"`, `"mistral-7b"` |
| `ANALYZE_LLM_TEMPERATURE` | Temperature for analysis | `0.0` (deterministic) | `0.0` - `0.2` |

#### AI Dashboard App (`streamlit_dashboard_app.py`)
For comprehensive AI insights and analysis - typically needs larger, more capable models:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `AI_LLM_PROVIDER` | LLM provider for AI analysis | `"LM Studio (Local)"` | `"OpenAI"`, `"Custom API"` |
| `AI_LLM_API_URL` | API endpoint for AI analysis | Empty (uses provider defaults) | `"http://localhost:1234/v1/chat/completions"` |
| `AI_LLM_API_KEY` | API key for AI analysis | Empty | `"sk-your-api-key-here"` |
| `AI_LLM_MODEL_ID` | Model for AI insights | `"gemma-2-9b-it"` | `"gpt-4"`, `"claude-3"`, `"llama-3-70b"` |
| `AI_LLM_TEMPERATURE` | Temperature for AI analysis | `0.1` | `0.1` - `0.3` |

### Generic Configuration (Fallback)

These variables are used as fallback if app-specific variables are not set:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `LLM_PROVIDER` | Default LLM provider | `"LM Studio (Local)"` | `"OpenAI"`, `"Custom API"` |
| `LLM_API_URL` | Default API endpoint | Empty | `"http://localhost:1234/v1/chat/completions"` |
| `LLM_API_KEY` | Default API key | Empty | `"sk-your-api-key-here"` |
| `LLM_MODEL_ID` | Default model identifier | `"gemma-2-9b-it"` | `"gpt-4"` |
| `LLM_TEMPERATURE` | Default temperature | `0.0` or `0.1` | `0.0` - `1.0` |

### Common Configuration

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `LM_STUDIO_HOST` | LM Studio base URL (used by both apps) | `"http://localhost:1234"` | `"http://192.168.1.100:1234"` |

## Usage Examples

### Setting Environment Variables

#### macOS/Linux (bash/zsh)
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export LLM_PROVIDER="LM Studio (Local)"
export LM_STUDIO_HOST="http://localhost:1234"
export LLM_MODEL_ID="gemma-2-9b-it"
export LLM_TEMPERATURE="0.1"
```

#### Windows (Command Prompt)
```cmd
set LLM_PROVIDER=LM Studio (Local)
set LM_STUDIO_HOST=http://localhost:1234
set LLM_MODEL_ID=gemma-2-9b-it
set LLM_TEMPERATURE=0.1
```

#### Windows (PowerShell)
```powershell
$env:LLM_PROVIDER = "LM Studio (Local)"
$env:LM_STUDIO_HOST = "http://localhost:1234"
$env:LLM_MODEL_ID = "gemma-2-9b-it"
$env:LLM_TEMPERATURE = "0.1"
```

### Using .env File (Recommended)

Create a `.env` file in the project root:

```env
# Common Configuration
LM_STUDIO_HOST=http://localhost:1234

# Review Analysis App - Smaller, faster model for extraction
ANALYZE_LLM_MODEL_ID=llama-3-8b
ANALYZE_LLM_TEMPERATURE=0.0

# AI Dashboard App - Larger model for comprehensive insights
AI_LLM_PROVIDER=LM Studio (Local)
AI_LLM_MODEL_ID=llama-3-70b
AI_LLM_TEMPERATURE=0.2

# Optional: Generic fallback for both apps
LLM_MODEL_ID=gemma-2-9b-it
LLM_TEMPERATURE=0.1
```

Then install python-dotenv and load it in your scripts:
```bash
pip install python-dotenv
```

The applications will automatically use these defaults when available.

## Configuration Priority

1. **User Interface Input**: Values entered in the UI take precedence
2. **Environment Variables**: Used as defaults when UI fields are empty
3. **Application Defaults**: Fallback values if no environment variables are set

## Provider-Specific Configuration Examples

### Example 1: Different Models for Each App (LM Studio)
```env
# Common
LM_STUDIO_HOST=http://localhost:1234

# Fast extraction model for review analysis
ANALYZE_LLM_MODEL_ID=mistral-7b-instruct
ANALYZE_LLM_TEMPERATURE=0.0

# Powerful model for AI insights
AI_LLM_PROVIDER=LM Studio (Local)
AI_LLM_MODEL_ID=mixtral-8x7b-instruct
AI_LLM_TEMPERATURE=0.2
```

### Example 2: OpenAI for AI Analysis, Local for Review Processing
```env
# Review Analysis - Local model (fast, no API costs)
LM_STUDIO_HOST=http://localhost:1234
ANALYZE_LLM_MODEL_ID=llama-3-8b
ANALYZE_LLM_TEMPERATURE=0.0

# AI Dashboard - OpenAI GPT-4 (powerful insights)
AI_LLM_PROVIDER=OpenAI
AI_LLM_API_URL=https://api.openai.com/v1/chat/completions
AI_LLM_API_KEY=sk-your-api-key-here
AI_LLM_MODEL_ID=gpt-4
AI_LLM_TEMPERATURE=0.3
```

### Example 3: Same Model for Both Apps
```env
# Use generic variables for both apps
LM_STUDIO_HOST=http://localhost:1234
LLM_MODEL_ID=gemma-2-9b-it
LLM_TEMPERATURE=0.1
```

## Why Different Models?

The two applications have different requirements:

- **Review Analysis App**: Needs to process potentially thousands of reviews quickly and consistently. A smaller, faster model (7B-13B parameters) is ideal for structured extraction tasks.
  
- **AI Dashboard App**: Generates comprehensive insights, identifies patterns, and provides strategic recommendations. A larger model (30B+ parameters) provides better reasoning and more nuanced analysis.

## Tips

- **Security**: Never commit `.env` files containing API keys to version control. Add `.env` to your `.gitignore` file.
- **Model Selection**: Use smaller models (7B-13B) for extraction tasks, larger models (30B+) for analysis and insights.
- **Testing**: Use different `.env` files for development and production environments.
- **Debugging**: Check current environment variables with `env | grep LLM` (macOS/Linux) or `set | findstr LLM` (Windows).
- **Temperature**: Lower values (0.0-0.3) for more deterministic results, higher values (0.5-1.0) for more creative responses.

## Troubleshooting

### Environment Variables Not Loading

1. **Verify variables are set**: Run `echo $LLM_MODEL_ID` (macOS/Linux) or `echo %LLM_MODEL_ID%` (Windows)
2. **Restart terminal**: Environment variables may not be loaded in current session
3. **Check spelling**: Variable names are case-sensitive on macOS/Linux
4. **Use absolute paths**: For LM_STUDIO_HOST, ensure the URL includes protocol (http://)

### Connection Issues

If you see "Cannot connect to LM Studio":
1. Verify LM Studio is running on the specified port
2. Check `LM_STUDIO_HOST` or `LLM_API_URL` is correctly set
3. Test connection: `curl http://localhost:1234/v1/models`