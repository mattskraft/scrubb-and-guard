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
├── pii_scrubber.py          # Core scrubbing module
├── apps/
│   └── app_pii-scrubber.py  # Streamlit demo app
├── data/
│   ├── namen.txt            # German surnames
│   └── orte.txt             # German cities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```python
from pii_scrubber import GermanPIIScrubber

scrubber = GermanPIIScrubber()
text = "Ich heiße Thomas Müller, wohne in Berlin und meine KV-Nummer ist A123456789."
cleaned = scrubber.scrub(text)
print(cleaned)
# Output: Ich heiße [PERSON] [PERSON], wohne in [LOCATION] und meine KV-Nummer ist [MEDICAL_ID].
```

### Streamlit App

Run the demo app:
```bash
streamlit run apps/app_pii-scrubber.py
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

The tool automatically downloads German name and location data from:
- GitHub repositories with German name datasets
- German postal code databases

Data is cached locally in `pii_data/` directory after first download.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

