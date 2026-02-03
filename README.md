# URLGuard-CLS

URLGuard-CLS is a lightweight URL content risk classification system based on Transformer models.
It classifies URLs into high-level risk categories by combining URL structure and a short excerpt of fetched web content, enabling automated risk analysis without reliance on proprietary threat intelligence services at inference time.

The model is trained using GPU acceleration but is designed for CPU-only deployment, making it suitable for research environments, internal services, and lightweight production systems.

⸻

## Features
-	Transformer-based URL content classification (DistilBERT)
-	GPU-accelerated training, CPU-only inference
-	High-level, abstract risk labels (non-proprietary taxonomy)
-	No dependency on CUDA, NVIDIA drivers, or GPU at runtime
-	REST API service implemented with FastAPI
-	Designed for containerized and on-premise deployment

⸻

## Risk Labels
The model outputs one of the following public, abstract labels:
-	BENIGN – General low-risk or informational content
-	NSFW – Adult or sexually explicit content
-	ILLEGAL – Content associated with illegal or highly regulated activities
-	MALICIOUS – Malware, ransomware, or active exploitation
-	DECEPTIVE – Phishing, scams, or misleading behavior
-	HIGH_RISK – Gray-area or potentially harmful content
-	INFRASTRUCTURE – Internet platforms, services, or technical infrastructure

⸻

## System Overview
1.	Input: URL
2.	Content Fetching:
-	Fetches publicly accessible web content
-	Extracts and truncates visible text (first 300 characters)
3.	Inference:
-	Combines URL and extracted text
-	Runs classification using a fine-tuned DistilBERT model
4.	Output:
-	Single risk label from the public label set

## Installation
1. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Running the API Server
```bash
export API_KEY=your_api_key
uvicorn main:app --host 0.0.0.0 --port 8000
```
## API Usage
### Authentication
```bash
X-API-Key: <your_api_key>
```
### POST /predict
Request
```json
{
  "url": "https://example.com"
}
```
Response
```json
{
  "label": "BENIGN"
}
```

## Model Training
-	Base model: distilbert-base-uncased
-	Training performed with GPU acceleration