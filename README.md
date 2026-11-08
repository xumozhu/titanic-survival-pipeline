# Titanic Survival Prediction Pipeline

This project implements Titanic survival prediction using logistic regression in both Python and R. Each implementation runs in a separate Docker container to provide reproducible execution environments.

## Project Structure

- `src/python/titanic_model.py` — Python logistic regression pipeline
- `src/python/Dockerfile` — Python Docker configuration
- `src/r/titanic_model.R` — R logistic regression pipeline
- `src/r/install_packages.R` — R dependency setup and verification
- `src/r/Dockerfile` — R Docker configuration
- `src/data/` — Local Titanic datasets and generated predictions
- `requirements.txt` — Python package dependencies

## Dataset Setup

1. Download the Titanic dataset from Kaggle:
   https://www.kaggle.com/competitions/titanic/data
2. Place `train.csv` and `test.csv` inside `src/data/`.

Dataset files are excluded from Git using `.gitignore`.

## Python Pipeline

Build the Docker image:

```bash
docker build -f src/python/Dockerfile -t titanic-python .
```

Run the Python pipeline:

```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data" \
  titanic-python
```

The Python pipeline generates `src/data/titanic_submission.csv`.

## R Pipeline

Build the Docker image:

```bash
docker build -f src/r/Dockerfile -t titanic-r .
```

Run the R pipeline:

```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data" \
  titanic-r
```

The R pipeline generates `src/data/titanic_submission_r.csv`.

## Model Results

Both implementations use seven predictors: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

| Metric | Python | R |
|---|---|---|
| Training observations | 712 | 712 |
| Validation observations | 179 | 179 |
| Validation accuracy | 80.45% | 80.45% |
| Correct validation predictions | 144 | 144 |
| Kaggle test predictions | 418 | 418 |

The implementations use different random sampling procedures and model defaults, so matching accuracy does not necessarily imply identical predictions.

## Reproducibility

- Python runs using Python 3.11 and dependencies pinned in `requirements.txt`.
- R runs using the `rocker/r-ver:4.5.2` Docker image.
- R dependency checks are performed during the Docker build.
- CSV files are mounted at runtime instead of being included in the Docker images.
- The model reports validation accuracy and generates predictions for the Kaggle test dataset.
