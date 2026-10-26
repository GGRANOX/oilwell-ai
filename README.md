# OilWell AI

Predicts the optimal pump power for an oil well from five sensor readings.
Hackathon project, 2025. I was the only developer on the team.

## How it works

Input features (one row per hour):

| Feature | Meaning |
|---|---|
| `oil_level` | Oil level in the well |
| `pressure` | Pressure |
| `current` | Pump motor current |
| `vibration` | Vibration level |
| `temperature` | Temperature |

Target: `optimal_power`.

The model is a `VotingRegressor` ensemble of three scikit-learn pipelines:

| Model | Settings | Scaler | Weight |
|---|---|---|---|
| Random Forest | 200 trees, max depth 12 | `RobustScaler` | 0.4 |
| Gradient Boosting | 150 estimators, learning rate 0.1, depth 5 | `StandardScaler` | 0.3 |
| MLP neural network | layers 100 → 50 → 25, ReLU, Adam, early stopping | `StandardScaler` | 0.3 |

Around the model:

- input validation: required columns, missing values, numeric types;
- emergency thresholds for oil level, pressure, current, vibration and temperature in `config.yaml`;
- time-series split for evaluation (MAE and R² are written to the log);
- rotating log file `oil_well_ai.log`;
- the trained model is saved to `oil_well_model.joblib` and loaded on the next start;
- unit tests for training, prediction and save/load.

## Data

We had no access to real well data during the hackathon, so the project generates a synthetic
`data.csv` (1000 hourly rows with random values). The target in this data is random, so the model
cannot learn a real relationship and R² stays around zero. The point of the project is the pipeline:
it can be retrained on real sensor data with the same columns without code changes.

## Run

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cd Hakaton
python Hakaton.py
```

Running the file starts the unit tests. They delete the old `data.csv` and model, generate new
synthetic data, train the ensemble and save it.

## Project structure

```
Hakaton/
├── Hakaton.py     # model, data checks, logging, unit tests
├── config.yaml    # thresholds, training parameters, column names
└── data.csv       # sample synthetic data
```

## Known issues and next steps

- `predict()` checks input with the same validation as training, so it expects the `optimal_power`
  column. Predicting from the five features alone currently fails, and `test_prediction` fails with it.
- `predict()` also applies a separate `RobustScaler` that is never fitted. The pipelines already scale
  their input, so this step should be removed.
- Training uses only the last time-series split instead of averaging metrics over all splits.
- The locale is hard-coded to `ru_RU.UTF-8`, which raises an error on systems without that locale.
- Train on real well data and report metrics against a simple baseline.
