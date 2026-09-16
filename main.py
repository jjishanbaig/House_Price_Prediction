
import io
import joblib
import pandas as pd 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model  = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


#input schema 
class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description="Media Income of Neighbourhood")
    HouseAge : float = Field(ge=0, description="Media Income of Neighbourhood")
    AveRooms : float = Field(gt=0, description="Media Income of Neighbourhood")
    AveBedrms : float = Field(gt=0, description="Media Income of Neighbourhood")
    Population : float = Field(gt=0, description="Media Income of Neighbourhood")
    AveOccup : float = Field(gt=0, description="Media Income of Neighbourhood")
    Latitude : float = Field(ge=32, le=42, description="Media Income of Neighbourhood")
    Longitude : float = Field(ge=-125, le=-114, description="Media Income of Neighbourhood")

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>California House Price Predictor</title>
        <style>
            :root {
                --bg: #07111f;
                --panel: rgba(15, 23, 42, 0.82);
                --panel-2: rgba(30, 41, 59, 0.9);
                --primary: #7dd3fc;
                --secondary: #38bdf8;
                --accent: #fbbf24;
                --text: #e2e8f0;
                --muted: #94a3b8;
                --success: #34d399;
                --shadow: rgba(14, 165, 233, 0.35);
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "Segoe UI", Tahoma, sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.28), transparent 25%),
                    radial-gradient(circle at bottom right, rgba(251, 191, 36, 0.18), transparent 25%),
                    linear-gradient(135deg, #020817, #0f172a 45%, #111827);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 30px;
            }
            .container {
                width: min(1200px, 100%);
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 28px;
                box-shadow: 0 30px 80px rgba(2, 6, 23, 0.7);
                backdrop-filter: blur(12px);
                overflow: hidden;
            }
            .header {
                padding: 28px 34px 20px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.14);
                background: linear-gradient(90deg, rgba(12, 18, 28, 0.96), rgba(15, 23, 42, 0.75));
            }
            .brand {
                display: flex;
                align-items: center;
                gap: 14px;
                font-weight: 700;
                letter-spacing: 0.04em;
                font-size: 1.1rem;
            }
            .brand-badge {
                width: 42px;
                height: 42px;
                border-radius: 12px;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: #082f49;
                font-weight: 900;
                box-shadow: 0 10px 25px var(--shadow);
            }
            .content {
                display: grid;
                grid-template-columns: 1.2fr 0.8fr;
                gap: 0;
            }
            .form-panel {
                padding: 32px 34px 40px;
                background: rgba(15, 23, 42, 0.72);
            }
            .result-panel {
                padding: 32px 26px;
                background: linear-gradient(180deg, rgba(8, 47, 73, 0.4), rgba(13, 30, 48, 0.8));
                border-left: 1px solid rgba(148, 163, 184, 0.14);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            h1 {
                margin: 0 0 8px;
                font-size: clamp(2rem, 2.6vw, 3.1rem);
                line-height: 1.1;
            }
            .subtitle {
                margin: 0 0 24px;
                color: var(--muted);
                line-height: 1.6;
                max-width: 620px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(180px, 1fr));
                gap: 18px;
            }
            .field {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .field label {
                color: var(--muted);
                font-size: 0.83rem;
                font-weight: 600;
                letter-spacing: 0.02em;
            }
            input {
                width: 100%;
                background: rgba(15, 23, 42, 0.9);
                color: var(--text);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                padding: 14px 14px;
                font-size: 1rem;
                outline: none;
                transition: border 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
            }
            input:focus {
                border-color: var(--secondary);
                box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.18);
                transform: translateY(-1px);
            }
            .action-row {
                margin-top: 28px;
                display: flex;
                align-items: center;
                gap: 16px;
                flex-wrap: wrap;
            }
            .file-upload {
                margin-top: 30px;
                padding-top: 24px;
                border-top: 1px solid rgba(148, 163, 184, 0.14);
            }
            .file-upload h2 { margin: 0 0 8px; font-size: 1.15rem; }
            .file-help { margin: 0 0 14px; color: var(--muted); font-size: 0.9rem; }
            .file-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
            .file-actions input { max-width: 300px; padding: 11px; }
            .download-link { display: none; color: var(--primary); font-weight: 700; }
            .table-wrap {
                display: none;
                margin-top: 18px;
                max-height: 220px;
                overflow: auto;
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 12px;
            }
            table { width: 100%; border-collapse: collapse; font-size: 0.8rem; white-space: nowrap; }
            th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.1); }
            th { color: var(--primary); }
            td { color: var(--muted); }
            button {
                border: none;
                border-radius: 14px;
                padding: 15px 24px;
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                background: linear-gradient(135deg, var(--secondary), var(--primary));
                color: #082f49;
                box-shadow: 0 15px 25px rgba(56, 189, 248, 0.28);
            }
            button:hover {
                transform: translateY(-2px);
            }
            .status {
                color: var(--muted);
                font-size: 0.92rem;
            }
            .result-box {
                width: 100%;
                max-width: 350px;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 22px;
                padding: 28px 22px;
                text-align: center;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.45);
            }
            .label {
                color: var(--muted);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }
            .price {
                margin-top: 18px;
                font-weight: 800;
                font-size: clamp(2.3rem, 3vw, 3.5rem);
                line-height: 1.1;
                color: var(--success);
                text-shadow: 0 0 18px rgba(52, 211, 153, 0.25);
            }
            .range {
                margin-top: 14px;
                color: var(--muted);
                font-size: 0.95rem;
            }
            @media (max-width: 860px) {
                .content {
                    grid-template-columns: 1fr;
                }
                .result-panel {
                    border-left: none;
                    border-top: 1px solid rgba(148, 163, 184, 0.14);
                    padding-top: 20px;
                }
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="brand">
                    <div class="brand-badge">$</div>
                    <span>California House Price Predictor</span>
                </div>
            </div>

            <div class="content">
                <div class="form-panel">
                    <h1>Estimate your home value</h1>
                    <p class="subtitle">
                        Enter key housing features to get a quick market estimate generated from a trained California housing model.
                    </p>

                    <form id="prediction-form">
                        <div class="grid">
                            <div class="field"><label>MedInc</label><input name="MedInc" type="number" value="3.5" step="any" required /></div>
                            <div class="field"><label>HouseAge</label><input name="HouseAge" type="number" value="25" step="any" required /></div>
                            <div class="field"><label>AveRooms</label><input name="AveRooms" type="number" value="5.0" step="any" required /></div>
                            <div class="field"><label>AveBedrms</label><input name="AveBedrms" type="number" value="1.0" step="any" required /></div>
                            <div class="field"><label>Population</label><input name="Population" type="number" value="700" step="any" required /></div>
                            <div class="field"><label>AveOccup</label><input name="AveOccup" type="number" value="2.5" step="any" required /></div>
                            <div class="field"><label>Latitude</label><input name="Latitude" type="number" value="36.0" step="any" required /></div>
                            <div class="field"><label>Longitude</label><input name="Longitude" type="number" value="-121.5" step="any" required /></div>
                        </div>

                        <div class="action-row">
                            <button type="submit">Predict Price</button>
                            <span id="status" class="status">Ready to estimate</span>
                        </div>
                    </form>

                    <section class="file-upload">
                        <h2>Predict from CSV</h2>
                        <p class="file-help">Upload all your housing rows and download the complete result file.</p>
                        <form id="file-form" class="file-actions">
                            <input id="csv-file" type="file" accept=".csv,text/csv" required />
                            <button type="submit">Upload CSV</button>
                            <a id="download-link" class="download-link" download="predictions.csv">Download results</a>
                            <span id="file-status" class="status">No file selected</span>
                        </form>
                        <div id="table-wrap" class="table-wrap"></div>
                    </section>
                </div>

                <div class="result-panel">
                    <div class="result-box">
                        <div class="label">Estimated Price</div>
                        <div id="price" class="price">$0</div>
                        <div id="range" class="range">Range: $0 to $0</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('prediction-form');
            const statusEl = document.getElementById('status');
            const priceEl = document.getElementById('price');
            const rangeEl = document.getElementById('range');
            const fileForm = document.getElementById('file-form');
            const csvFile = document.getElementById('csv-file');
            const fileStatusEl = document.getElementById('file-status');
            const downloadLink = document.getElementById('download-link');
            const tableWrap = document.getElementById('table-wrap');

            function parseCSV(csv) {
                const rows = [];
                let row = [];
                let cell = '';
                let quoted = false;

                for (let index = 0; index < csv.length; index += 1) {
                    const character = csv[index];
                    const nextCharacter = csv[index + 1];

                    if (character === '"' && quoted && nextCharacter === '"') {
                        cell += '"';
                        index += 1;
                    } else if (character === '"') {
                        quoted = !quoted;
                    } else if (character === ',' && !quoted) {
                        row.push(cell);
                        cell = '';
                    } else if ((character === '\\n' || character === '\\r') && !quoted) {
                        if (character === '\\r' && nextCharacter === '\\n') index += 1;
                        row.push(cell);
                        rows.push(row);
                        row = [];
                        cell = '';
                    } else {
                        cell += character;
                    }
                }

                if (cell || row.length) {
                    row.push(cell);
                    rows.push(row);
                }
                return rows;
            }

            function csvToTable(csv) {
                const rows = parseCSV(csv.trim());
                if (!rows.length) return '';
                const header = rows[0].map(cell => `<th>${cell}</th>`).join('');
                const body = rows.slice(1).map(row =>
                    `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`
                ).join('');
                return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
            }

            form.addEventListener('submit', async function (event) {
                event.preventDefault();
                statusEl.textContent = 'Predicting...';

                const payload = {};
                new FormData(form).forEach((value, key) => {
                    payload[key] = Number(value);
                });

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const result = await response.json();
                    if (!response.ok) {
                        throw new Error(result.detail || 'Prediction failed');
                    }

                    priceEl.textContent = result.predicted_price;
                    rangeEl.textContent = 'Range: ' + result.confidence_range;
                    statusEl.textContent = 'Success';
                } catch (error) {
                    statusEl.textContent = error.message;
                    priceEl.textContent = '$0';
                    rangeEl.textContent = 'Range: $0 to $0';
                }
            });

            fileForm.addEventListener('submit', async function (event) {
                event.preventDefault();
                const file = csvFile.files[0];
                if (!file) return;

                fileStatusEl.textContent = 'Uploading and predicting...';
                downloadLink.style.display = 'none';
                tableWrap.style.display = 'none';

                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    const response = await fetch('/predict-file', { method: 'POST', body: formData });
                    const result = await response.text();
                    if (!response.ok) {
                        let detail = result;
                        try { detail = JSON.parse(result).detail || result; } catch (_) {}
                        throw new Error(detail);
                    }

                    const downloadUrl = URL.createObjectURL(new Blob([result], { type: 'text/csv' }));
                    downloadLink.href = downloadUrl;
                    downloadLink.style.display = 'inline-block';
                    tableWrap.innerHTML = csvToTable(result);
                    tableWrap.style.display = 'block';
                    fileStatusEl.textContent = 'Success: all rows predicted';
                } catch (error) {
                    fileStatusEl.textContent = error.message;
                }
            });
        </script>
    </body>
    </html>
    """)

@app.get("/health")
def health():
    return {
        "status":"running",
        "model" : "RandomForestRegressor",
        "features" : features,
        "avg_error" : "$39000"
    }


#prediction
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted:.2f} hundred thousands",
            "confidence_range" : f"${price_usd - 39000:,.0f}  to ${price_usd + 39000:,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"prediction failed: {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV file only"
        )

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_columns}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="the uploaded file has no data rows"
        )

    try:
        predictions = model.predict(df[required_columns])
        df["predicted_price_usd"] = [f"${price * 100000:,.0f}" for price in predictions]

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


    





