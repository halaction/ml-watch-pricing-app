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

## Functionality

The app offers the following functionalities:

### Server Status Check

- The app continuously checks the status of the backend API using the `check_server` function.
- If the API is unavailable, a spinner is displayed while the app waits for the API to become available.

### Model Selection

- Users can choose a model from a list of supported models via a radio button in the sidebar.
- The selected model is loaded for inference using a POST request to the backend API.
- If no model is selected, the previously loaded model (if any) is unloaded.

### Data Upload

- Users can upload a CSV file containing watch data for inference using a file uploader in the sidebar.
- The uploaded data is sent to the backend API for prediction using a POST request.
- Error handling is implemented to display error messages to the user in case of issues during data upload or inference.

### Inference Status

- The app displays the status of model and data readiness.
- If both model and data are ready, a success message is shown, indicating that the app is ready for inference.
- If either model or data is missing, an informative message prompts the user to complete the necessary steps.

### Evaluation

- If both model and data are ready, the app allows for model evaluation.
- It fetches pre-calculated metrics and performs inference on the uploaded data to calculate additional metrics.
- Users can select a metric from a list of supported metrics to visualize using a bar chart.
- A custom metric can be computed by providing the metric name.

### Interpretation

- The app provides tools for model interpretation.
- Permutation importance values are displayed in a bar chart, allowing users to understand feature importance.
- Partial dependence plots can be generated for selected features to visualize the relationship between the feature and the target variable.
- Users can choose to view all features, the top 10 features, or the bottom 10 features in the permutation importance plot.

## Backend Integration

The app interacts with a backend API using the `requests` library. The API is assumed to be running at `http://fastapi:8000` and provides the following endpoints:

- `/get_status`: Returns the status of the API.
- `/get_supported_models`: Returns a list of supported models.
- `/load_model`: Loads a specified model for inference.
- `/unload_model`: Unloads the currently loaded model.
- `/upload_data`: Uploads data for inference.
- `/load_data`: Loads data for inference.
- `/unload_data`: Unloads data for inference.
- `/evaluate`: Performs price prediction and calculates metrics on the uploaded data.
- `/get_metrics`: Returns pre-calculated metrics.
- `/get_supported_metrics`: Returns a list of supported metrics.
- `/compute_metric`: Computes a custom metric.
- `/get_dtypes`: Returns data types of features.
- `/get_importance_values`: Returns permutation importance values.
- `/get_dependence_values`: Returns partial dependence values.

## Note

- The app assumes the backend API is running and accessible at the specified URL.
- The input data should be in a CSV format with the required columns for the selected model.