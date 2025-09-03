# Environment Variables Documentation

## Overview
Both Streamlit applications support environment variables to set default LLM configuration values. Each app can use **app-specific variables** for different models and settings, with fallback to generic variables for convenience.

**Important**: No hard-coded model defaults are used. Models must be specified in the .env file or will be auto-selected from available LM Studio models.

## Supported Environment Variables

### App-Specific Configuration (Recommended)

#### Review Analysis App (`streamlit_analyze_app.py`)
For sentiment extraction and problem categorization - typically needs smaller, faster models:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `ANALYZE_LLM_MODEL_ID` | Model for review analysis | None (must be set or auto-selected) | `"gpt-oss-20b"`, `"llama-3-8b"`, `"mistral-7b"` |
| `ANALYZE_LLM_TEMPERATURE` | Temperature for analysis | `0.0` (deterministic) | `0.0` - `0.2` |

#### AI Dashboard App (`streamlit_dashboard_app.py`)
For comprehensive AI insights and analysis - typically needs larger, more capable models:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `AI_LLM_PROVIDER` | LLM provider for AI analysis | `"LM Studio (Local)"` | `"OpenAI"`, `"Custom API"` |
| `AI_LLM_API_URL` | API endpoint for AI analysis | Empty (uses provider defaults) | `"http://localhost:1234/v1/chat/completions"` |
| `AI_LLM_API_KEY` | API key for AI analysis | Empty | `"sk-your-api-key-here"` |
| `AI_LLM_MODEL_ID` | Model for AI insights | None (must be set or auto-selected) | `"gpt-oss-120b"`, `"gpt-4"`, `"claude-3"`, `"llama-3-70b"` |
| `AI_LLM_TEMPERATURE` | Temperature for AI analysis | `0.1` | `0.1` - `0.3` |

### Generic Configuration (Fallback)

These variables are used as fallback if app-specific variables are not set:

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `LLM_PROVIDER` | Default LLM provider | `"LM Studio (Local)"` | `"OpenAI"`, `"Custom API"` |
| `LLM_API_URL` | Default API endpoint | Empty | `"http://localhost:1234/v1/chat/completions"` |
| `LLM_API_KEY` | Default API key | Empty | `"sk-your-api-key-here"` |
| `LLM_MODEL_ID` | Default model identifier | None (must be set) | `"gpt-4"`, `"gemma-3-12b"` |
| `LLM_TEMPERATURE` | Default temperature | `0.0` or `0.1` | `0.0` - `1.0` |

### Common Configuration

| Variable | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `LM_STUDIO_HOST` | LM Studio base URL (used by both apps) | `"http://localhost:1234"` | `"http://192.168.1.100:1234"` |
| `LLM_REQUEST_TIMEOUT` | Request timeout in seconds | `300` | `60` - `600` |
| `LLM_MODEL_FETCH_TIMEOUT` | Timeout for fetching model list | `10` | `5` - `30` |

## Model Auto-Selection

If no model is specified in the environment variables:
1. The app will query LM Studio for available models
2. The first available model will be automatically selected
3. A notification will inform you which model is being used
4. If no models are available, an error message will guide you to configure the `.env` file

## Usage Examples

### Setting Environment Variables

#### macOS/Linux (bash/zsh)
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export LM_STUDIO_HOST="http://localhost:1234"
export ANALYZE_LLM_MODEL_ID="gpt-oss-20b"
export ANALYZE_LLM_TEMPERATURE="0.0"
export AI_LLM_MODEL_ID="gpt-oss-120b"
export AI_LLM_TEMPERATURE="0.1"
```

#### Windows (Command Prompt)
```cmd
set LM_STUDIO_HOST=http://localhost:1234
set ANALYZE_LLM_MODEL_ID=gpt-oss-20b
set ANALYZE_LLM_TEMPERATURE=0.0
set AI_LLM_MODEL_ID=gpt-oss-120b
set AI_LLM_TEMPERATURE=0.1
```

#### Windows (PowerShell)
```powershell
$env:LM_STUDIO_HOST = "http://localhost:1234"
$env:ANALYZE_LLM_MODEL_ID = "gpt-oss-20b"
$env:ANALYZE_LLM_TEMPERATURE = "0.0"
$env:AI_LLM_MODEL_ID = "gpt-oss-120b"
$env:AI_LLM_TEMPERATURE = "0.1"
```

### Using .env File (Recommended)

Create a `.env` file in the project root:

```env
# Common Configuration
LM_STUDIO_HOST=http://localhost:1234

# Review Analysis App - Smaller, faster model for extraction
ANALYZE_LLM_MODEL_ID=gpt-oss-20b
ANALYZE_LLM_TEMPERATURE=0.0

# AI Dashboard App - Larger model for comprehensive insights
AI_LLM_PROVIDER=LM Studio (Local)
AI_LLM_MODEL_ID=gpt-oss-120b
AI_LLM_TEMPERATURE=0.1

# Timeout Settings
LLM_REQUEST_TIMEOUT=300
LLM_MODEL_FETCH_TIMEOUT=10
```

Then install python-dotenv and it will be loaded automatically:
```bash
pip install python-dotenv
```

The applications will automatically use these defaults when available.

## Configuration Priority

1. **User Interface Input**: Values entered in the UI take precedence (if UI allows input)
2. **App-Specific Environment Variables**: `ANALYZE_*` or `AI_*` variables
3. **Generic Environment Variables**: `LLM_*` variables as fallback
4. **Auto-Selection**: If no model specified, first available model from LM Studio
5. **Error State**: If no models available and none specified

## Provider-Specific Configuration Examples

### Example 1: Different Models for Each App (LM Studio)
```env
# Common
LM_STUDIO_HOST=http://localhost:1234

# Fast extraction model for review analysis
ANALYZE_LLM_MODEL_ID=gpt-oss-20b
ANALYZE_LLM_TEMPERATURE=0.0

# Powerful model for AI insights
AI_LLM_PROVIDER=LM Studio (Local)
AI_LLM_MODEL_ID=gpt-oss-120b
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

### Example 3: Let System Auto-Select Models
```env
# Only specify the LM Studio host
# Models will be auto-selected from available models
LM_STUDIO_HOST=http://localhost:1234
ANALYZE_LLM_TEMPERATURE=0.0
AI_LLM_TEMPERATURE=0.1
```

## Why Different Models?

The two applications have different requirements:

- **Review Analysis App**: Needs to process potentially thousands of reviews quickly and consistently. A smaller, faster model (7B-20B parameters) is ideal for structured extraction tasks.
  
- **AI Dashboard App**: Generates comprehensive insights, identifies patterns, and provides strategic recommendations. A larger model (30B+ parameters) provides better reasoning and more nuanced analysis.

## Tips

- **Security**: Never commit `.env` files containing API keys to version control. Add `.env` to your `.gitignore` file.
- **Model Selection**: Use smaller models (7B-20B) for extraction tasks, larger models (30B+) for analysis and insights.
- **Testing**: Use different `.env` files for development and production environments.
- **Debugging**: Check current environment variables with `env | grep LLM` (macOS/Linux) or `set | findstr LLM` (Windows).
- **Temperature**: Lower values (0.0-0.3) for more deterministic results, higher values (0.5-1.0) for more creative responses.
- **Auto-Selection**: If you don't specify models, the system will use the first available model from LM Studio.

## Troubleshooting

### Environment Variables Not Loading

1. **Verify variables are set**: Run `echo $ANALYZE_LLM_MODEL_ID` (macOS/Linux) or `echo %ANALYZE_LLM_MODEL_ID%` (Windows)
2. **Restart terminal**: Environment variables may not be loaded in current session
3. **Check spelling**: Variable names are case-sensitive on macOS/Linux
4. **Use absolute paths**: For LM_STUDIO_HOST, ensure the URL includes protocol (http://)

### Connection Issues

If you see "Cannot connect to LM Studio":
1. Verify LM Studio is running on the specified port
2. Check `LM_STUDIO_HOST` is correctly set
3. Test connection: `curl http://localhost:1234/v1/models`
4. Ensure at least one model is loaded in LM Studio

### Model Not Found

If the specified model is not found:
1. The app will automatically use the first available model
2. You'll see a notification showing which model is being used
3. To use a specific model, ensure it's loaded in LM Studio
4. Check available models: `curl http://localhost:1234/v1/models`