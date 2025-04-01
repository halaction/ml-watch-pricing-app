# Watch Pricing App

This app provides a user interface for predicting watch prices using machine learning models. It interacts with a backend API for model management, data handling, and evaluation.

## Launch

```{shell}
git clone https://github.com/halaction/ml-watch-pricing-app.git
cd ml-watch-pricing-app

docker compose up -d --build
```

## Usage

1. Open the Streamlit app at `http://localhost:8501`.
2. Wait for the API to become available.
3. Select a model from the sidebar.
4. Upload a CSV file containing watch data.
5. View the predicted prices, evaluation metrics, and interpretation plots.
