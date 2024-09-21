from fastapi import FastAPI, HTTPException, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI instance
app = FastAPI()

# Serve static files from the 'templates' directory
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Load the machine learning model
model = joblib.load('Water_Quality_Model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define request body model
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_Carbon: float
    Trihalomethanes: float
    Turbidity: float

# Endpoint to serve the HTML file
@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

# Define POST endpoint for prediction
@app.post("/predict")
async def predict_water_quality(data: WaterQuality):
    try:
        # Extract features
        features = np.array([[data.ph, data.Hardness, data.Solids, data.Chloramines, data.Sulfate, data.Conductivity, data.Organic_Carbon, data.Trihalomethanes, data.Turbidity]])

        # Apply feature scaling
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Convert numpy int64 to native Python int
        prediction = int(prediction)

        # Return prediction
        return {"prediction": prediction}

    except Exception as e:
        # Return error if prediction fails
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI application on the specified host and port
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8884)
