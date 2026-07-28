import sys
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the model predictor once at application startup
model_predictor = VehicleDataClassifier()


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """Renders the main HTML form page for vehicle data input."""
    return templates.TemplateResponse(
        "vehicledata.html", {"request": request, "context": None}
    )


@app.get("/train")
async def train_route_client():
    """Endpoint to initiate the model training pipeline."""
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"status": True, "message": "Training successful!"}
    except Exception as e:
        return {"status": False, "error": str(e)}


@app.post("/")
async def predict_route_client(
    request: Request,
    Gender: Annotated[int, Form(...)],
    Age: Annotated[int, Form(...)],
    Driving_License: Annotated[int, Form(...)],
    Region_Code: Annotated[float, Form(...)],
    Previously_Insured: Annotated[int, Form(...)],
    Annual_Premium: Annotated[float, Form(...)],
    Policy_Sales_Channel: Annotated[float, Form(...)],
    Vintage: Annotated[int, Form(...)],
    Vehicle_Age_lt_1_Year: Annotated[int, Form(...)],
    Vehicle_Age_gt_2_Years: Annotated[int, Form(...)],
    Vehicle_Damage_Yes: Annotated[int, Form(...)],
):
    """Endpoint to receive form data, process it, and render predictions."""
    try:
        vehicle_data = VehicleData(
            Gender=Gender,
            Age=Age,
            Driving_License=Driving_License,
            Region_Code=Region_Code,
            Previously_Insured=Previously_Insured,
            Annual_Premium=Annual_Premium,
            Policy_Sales_Channel=Policy_Sales_Channel,
            Vintage=Vintage,
            Vehicle_Age_lt_1_Year=Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=Vehicle_Damage_Yes,
        )

        vehicle_df = vehicle_data.get_vehicle_input_data_frame()
        
        # Run prediction on the global model instance
        prediction = model_predictor.predict(dataframe=vehicle_df)[0]
        status = "Response-Yes" if prediction == 1 else "Response-No"

        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": f"Error: {e}"},
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)