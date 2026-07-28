# Vehicle Insurance Response Prediction

An end-to-end Machine Learning web application that predicts whether a customer will be interested in purchasing Vehicle Insurance based on demographics, vehicle details, and policy history.

The project features a **FastAPI** web interface, automated MLOps pipelines for training and inference, **AWS S3** model artifact registry integration, and containerization via **Docker**.

---

## Project Architecture & Tech Stack

- **Web Framework:** FastAPI with Jinja2 Templating
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **MLOps & Pipeline Management:** Custom OOP Modular Pipeline (Data Ingestion, Transformation, Model Trainer, Estimator)
- **Cloud Storage:** AWS S3 (Stores model artifacts and preprocessors)
- **Containerization:** Docker & Uvicorn

---

## Project Structure

```text
├── artifacts/                  # Local artifact storage (models, preprocessors)
├── static/                     # CSS, JavaScript, and asset files
├── templates/                  # HTML templates (vehicledata.html)
├── src/
│   ├── components/             # Ingestion, Data Transformation, Model Trainer
│   ├── constants/              # Global constants (DB names, S3 bucket config, app ports)
│   ├── entity/                 # Config & Artifact entity definitions
│   ├── exception/              # Custom Exception handling
│   ├── logger/                 # Logging setup
│   ├── pipline/                # Training and Prediction pipeline logic
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   └── utils/                  # Helper utilities (S3 helpers, model loader)
├── app.py                      # FastAPI Web Application Entrypoint
├── Dockerfile                  # Container build instructions
├── requirements.txt            # Python dependencies
└── README.md
```
