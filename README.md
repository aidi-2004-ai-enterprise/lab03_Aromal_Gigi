Lab 3: Penguins Classification with XGBoost and FastAPI
This project builds a robust ML pipeline to classify penguin species. It includes data preprocessing, XGBoost model training, and deployment with FastAPI. All inputs are validated and the API is fully documented.

Directory Structure
pgsql
Copy
Edit
├── train.py
├── app/
│   ├── main.py
│   └── data/
│       ├── model.json
│       └── metadata.json
├── pyproject.toml
├── README.md
Setup Instructions
1. Clone and Set Up Environment
bash
Copy
Edit
git clone https://github.com/aidi-2004-ai-enterprise/lab03_Aromal_Gigi.git
cd lab3_Aromal_Gigi
uv venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# or: source .venv/bin/activate   # Mac/Linux
uv pip install -r requirements.txt
2. Train the Model
bash
Copy
Edit
python train.py
This saves the model and metadata to app/data/.

3. Launch the API
bash
Copy
Edit
uv run uvicorn app.main:app --reload
API Docs: http://127.0.0.1:8000/docs

Health Check: http://127.0.0.1:8000/health

Root Greeting: http://127.0.0.1:8000/

API Usage Examples
Valid Request
With cURL (Windows Command Prompt/Terminal)
bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Torgersen\"}"
Or with PowerShell:
powershell
Copy
Edit
$body = @{
  bill_length_mm    = 39.1
  bill_depth_mm     = 18.7
  flipper_length_mm = 181
  body_mass_g       = 3750
  year              = 2007
  sex               = "male"
  island            = "Torgersen"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
Expected response:

json
Copy
Edit
{"species":"Adelie"}
Invalid Input: Wrong Sex
cURL:
bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"shemale\",\"island\":\"Torgersen\"}"
Expected response:

json
Copy
Edit
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "sex"],
      "msg": "Input should be 'male' or 'female'",
      "input": "shemale",
      "ctx": {"expected": "'male' or 'female'"}
    }
  ]
}
Invalid Input: Wrong Island
cURL:
bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"year\":2007,\"sex\":\"male\",\"island\":\"Australia\"}"
Expected response:

json
Copy
Edit
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "island"],
      "msg": "Input should be 'Torgersen', 'Biscoe' or 'Dream'",
      "input": "Australia",
      "ctx": {"expected": "'Torgersen', 'Biscoe' or 'Dream'"}
    }
  ]
}
Demo Video
A demo video (demo.mp4) is included showing:

Running training & launching the server

Sending valid and invalid requests (for both sex & island)

Browsing to /docs and /health

Root greeting message

Acknowledgements & Dependencies
This project relies on the following open-source libraries and tools:

FastAPI — for building the REST API

Uvicorn — lightning-fast ASGI server for FastAPI

pydantic — for data validation

XGBoost — for gradient boosting classification

scikit-learn — for preprocessing, train/test split, and evaluation

pandas — for data wrangling and one-hot encoding

seaborn — for loading the Palmer Penguins dataset

uv — for fast, modern Python dependency management

curl — for API testing (example requests)

Special thanks to the Palmer Penguins dataset and to the FastAPI, scikit-learn, and XGBoost communities for excellent documentation and open-source code.