# Scrubb and Guard - German PII Scrubber

A lightweight, on-device PII (Personally Identifiable Information) scrubbing tool for German text. Designed for mobile deployment with regex patterns and deny lists (no heavy AI models required).

## Features

- **Regex-based detection** for structured PII:
  - German KV-Nummer (Medical ID)
  - IBAN numbers
  - Phone numbers
  - Email addresses
  - Dates/years

- **Deny list matching** for:
  - German first names (male & female)
  - German surnames
  - German cities/locations

- **Mobile-ready**: Exports configuration and deny lists for mobile app integration

## Project Structure

```
scrubb-and-guard/
├── scrubb_guard/            # Main package
│   ├── __init__.py          # Package exports
│   └── pii_scrubber.py      # Core scrubbing module
├── apps/
│   └── app-pii-scrubber.py  # Streamlit demo app
├── data/
│   ├── namen.txt            # German surnames
│   └── orte.txt             # German cities
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the package in development mode:
```bash
pip install -e .
```

## Usage

### As a Library

```python
from scrubb_guard import GermanPIIScrubber

scrubber = GermanPIIScrubber()
text = "Ich heiße Thomas Müller, wohne in Berlin und meine KV-Nummer ist A123456789."
cleaned = scrubber.scrub(text)
print(cleaned)
# Output: Ich heiße [PERSON] [PERSON], wohne in [LOCATION] und meine KV-Nummer ist [MEDICAL_ID].
```

### Command Line

You can also run the module directly:
```bash
python -m scrubb_guard.pii_scrubber
```

### Streamlit App

Run the demo app:
```bash
streamlit run apps/app-pii-scrubber.py
```

### Export for Mobile

```python
scrubber = GermanPIIScrubber()
scrubber.export_mobile_config("mobile_assets")
```

This creates:
- `mobile_assets/pii_config.json` - Regex patterns configuration
- `mobile_assets/deny_names.txt` - Names deny list
- `mobile_assets/deny_cities.txt` - Cities deny list

## How It Works

1. **Stage 1 - Regex Patterns**: Matches structured identifiers (IBAN, phone, email, etc.)
2. **Stage 2 - Deny Lists**: Exact word matching against known names and cities
3. **Stage 3 - Number Scrubbing**: Removes sequences of 3+ digits

## Data Sources

The tool uses local data files in the `data/` directory:
- `data/namen.txt` - German surnames
- `data/orte.txt` - German cities

These files are loaded at initialization time.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

# Streamlit Cloud deployment
