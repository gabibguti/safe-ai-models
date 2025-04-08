# Safe AI Models

Learning how to create and deploy safe AI models.

## Development

## Use Virtual Environment
```bash
# Create venv
python -m venv .venv
# Activate venv
source .venv/bin/activate
```

## Install dependencies
```bash
# Install new dependency
pip install <package>
# Update dependencies file
pip freeze > requirements.txt
# Install dependencies
pip install -r requirements.txt
```

## Test

### PyTorch Hub
```bash
python -m pytorch_model.generate_model
python -m fetch_flower_prediction_model
```
To test different save and loads, you must generate the new binary, point the load to your binary, and upload all configurations to GitHub. Then run the prediction.

### HuggingFace
First, uncomment save and fetch from HuggingFace, then run:
```bash
python -m pytorch_model.generate_model
python -m fetch_flower_prediction_model
```

## Troubleshooting

### SSL: CERTIFICATE_VERIFY_FAILED
OS: macOS
Error:
```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)
```
Fix:
```
/Applications/Python\ 3.11/Install\ Certificates.command 
 -- pip install --upgrade certifi
```

### Torch Hub Load GitHub Repository Not Found
The repository must be public for it to be visible for Torch Hub Load.