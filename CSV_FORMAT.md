# CSV File Format Documentation

## Input CSV File Requirements

The Review Analysis system accepts CSV files containing customer reviews. The system includes AI-powered column matching to automatically map your CSV columns to the required fields.

### Required Fields

Your CSV must contain columns that can be mapped to these required fields:

| Field | Description | Common Column Names | Example Values |
|-------|-------------|-------------------|----------------|
| **product** | Product identifier or SKU | `SKU_TEXT`, `PRODUCT_SKU_TEXT`, `PRODUCT_ID`, `ITEM_CODE`, `PRODUCT_NAME_TEXT` | `"ABC123"`, `"SHOE-001"`, `"12345"` |
| **rating** | Numeric rating (typically 1-5) | `RATING_AMOUNT`, `STAR_RATING`, `SCORE`, `RATING`, `STARS` | `5`, `4.5`, `1` |
| **date** | Review submission date | `REVIEW_DATE`, `DATE_CREATED`, `TIMESTAMP`, `CREATED_AT`, `DATE` | `"2024-01-15"`, `"01/15/2024"`, `"2024-01-15 10:30:00"` |
| **text** | The actual review content | `COMMENT_TEXT`, `REVIEW_TEXT`, `FEEDBACK`, `REVIEW_CONTENT`, `MESSAGE` | `"Great product! Fits perfectly..."` |

### Optional Fields

| Field | Description | Common Column Names | Example Values |
|-------|-------------|-------------------|----------------|
| **title** | Review title or subject | `REVIEW_TITLE_TEXT`, `SUBJECT`, `HEADLINE`, `TITLE`, `SUMMARY` | `"Excellent Quality"`, `"Disappointed"` |

### Additional Columns (Preserved but Not Required)

Your CSV may contain additional columns that will be preserved in the output:
- `REVIEW_ID_TEXT` - Unique review identifier
- `VERIFIED_FLAG` - Whether purchase was verified
- `SOURCE_TEXT` - Source of the review (website, app, etc.)
- `STORE_NAME_TEXT` - Store or location
- `AUTHOR_ATTRIBUTES_VARIANT` - Reviewer information
- `HELPFUL_COUNT_QUANTITY` - Number of helpful votes
- `REVIEW_TAGS_VARIANT` - Tags or labels

## Sample Input CSV

### Minimal Required Format
```csv
SKU_TEXT,RATING_AMOUNT,REVIEW_DATE,COMMENT_TEXT
ABC123,5,2024-01-15,"This product exceeded my expectations! Great quality."
XYZ789,2,2024-01-16,"The size runs small and material feels cheap."
DEF456,4,2024-01-17,"Good value for money. Would recommend."
```

### Full Format Example
```csv
REVIEW_ID_TEXT,REVIEW_TITLE_TEXT,REVIEW_TYPE_TEXT,VERIFIED_FLAG,SOURCE_TEXT,STORE_NAME_TEXT,PRODUCT_NAME_TEXT,SKU_TEXT,PRODUCT_OPTIONS_VARIANT,COMMENT_TEXT,AUTHOR_ATTRIBUTES_VARIANT,RATING_AMOUNT,REVIEW_DATE,HELPFUL_COUNT_QUANTITY,REVIEW_TAGS_VARIANT
R001,"Love it!",CUSTOMER,TRUE,Website,Online,Blue Running Shoes,SHOE-001,Size:10/Color:Blue,"These shoes are amazing! Perfect fit and very comfortable for long runs.",John D.,5,2024-01-15,23,comfort;fit;quality
R002,"Not happy",CUSTOMER,TRUE,Mobile App,Store #5,Red T-Shirt,SHIRT-002,Size:M/Color:Red,"Material started pilling after first wash. Size runs small.",Jane S.,2,2024-01-16,5,material;size
```

## Product Metadata File (Optional)

If you want to enhance product information in the analysis, include a `products.csv` file:

### Required Columns
| Column | Description | Example |
|--------|-------------|---------|
| `PRODUCT_SKU_TEXT` | Product SKU (must match review SKU) | `"SHOE-001"` |
| `STYLE_CODE_AND_TEXT` | Product description | `"Men's Running Shoe - Blue Lightning"` |

### Sample products.csv
```csv
PRODUCT_SKU_TEXT,STYLE_CODE_AND_TEXT
SHOE-001,Men's Running Shoe - Blue Lightning
SHIRT-002,Women's Athletic T-Shirt - Red
PANTS-003,Yoga Pants - Black Stretch
```

## Output CSV Format

After analysis, the system generates a results CSV with the following structure:

### Output Columns
| Column | Description | Example |
|--------|-------------|---------|
| `date` | Review date from input | `"2024-01-15"` |
| `product` | Product SKU from input | `"SHOE-001"` |
| `product_description` | Enhanced description (if products.csv provided) | `"Men's Running Shoe - Blue Lightning"` |
| `rating` | Numeric rating from input | `5` |
| `sentiment` | AI-determined sentiment | `"Positive"`, `"Negative"`, `"Neutral"` |
| `problems_mentioned` | Categorized problems (semicolon-separated) | `"Size; Material"` or `"None"` |
| `positive_mentions` | Categorized positives (semicolon-separated) | `"Comfort; Fit; Quality"` or `"None"` |
| `original_text` | Complete review text analyzed | `"These shoes are amazing..."` |

### Sample Output
```csv
date,product,product_description,rating,sentiment,problems_mentioned,positive_mentions,original_text
2024-01-15,SHOE-001,Men's Running Shoe - Blue Lightning,5,Positive,None,"Comfort; Fit; Quality","Love it! These shoes are amazing! Perfect fit and very comfortable for long runs."
2024-01-16,SHIRT-002,Women's Athletic T-Shirt - Red,2,Negative,"Material; Size",None,"Not happy. Material started pilling after first wash. Size runs small."
2024-01-17,PANTS-003,Yoga Pants - Black Stretch,4,Positive,None,"Price; Quality","Good value for money. Would recommend."
```

## Date Format Support

The system supports various date formats and will automatically parse:
- ISO format: `2024-01-15`
- US format: `01/15/2024`
- European format: `15/01/2024` (if unambiguous)
- With time: `2024-01-15 10:30:00`
- Text format: `Jan 15, 2024`

## Column Mapping Features

### AI-Powered Auto-Matching
The system uses AI to automatically match your CSV columns to required fields based on:
- Column names
- Sample data values
- Common patterns

### Manual Mapping
If auto-matching fails or needs adjustment, you can manually select columns using dropdown menus in the interface.

## Data Validation

The system performs these validations:
- **Rating Range**: Ensures ratings are numeric and typically between 1-5
- **Date Parsing**: Validates and standardizes date formats
- **Text Content**: Handles empty reviews gracefully (defaults to "Neutral" sentiment)
- **Product IDs**: Converts to uppercase for matching with products.csv

## Tips for Best Results

1. **Column Names**: Use descriptive column names for better auto-matching
2. **Date Consistency**: Use consistent date format throughout your CSV
3. **Rating Values**: Ensure ratings are numeric (not text like "5 stars")
4. **Text Encoding**: Save CSV as UTF-8 to handle special characters
5. **File Size**: For files >10,000 rows, consider using test mode first

## Error Handling

The system handles these issues gracefully:
- **Missing Values**: Empty cells are handled with appropriate defaults
- **Malformed Data**: Invalid entries are logged but don't stop processing
- **Large Files**: Progress tracking shows status for large datasets
- **LLM Errors**: Reviews that fail analysis default to "Neutral" sentiment

## File Size Considerations

- **Small** (<1,000 reviews): Process in full mode
- **Medium** (1,000-10,000 reviews): Test with sample first
- **Large** (>10,000 reviews): Consider batch processing or increased timeouts
- **Note**: GitHub has a 100MB file size limit (use Git LFS for larger files)