import time

import streamlit as st
import requests
import pandas as pd
import plotly.express as px


BASE_URL = "http://fastapi:8000"
# BASE_URL = "http://0.0.0.0:8000"


def check_server():

    try:
        url = f"{BASE_URL}/get_status"
        response = requests.get(url)
        response.raise_for_status()
    except Exception:
        return False

    return True


def main():

    st.title("Watch Pricing")

    while True:
        if check_server():
            break
        else:
            with st.spinner("Waiting for API..."):
                time.sleep(5)

    with st.sidebar:

        with st.container():
            st.header("Model")

            url = f"{BASE_URL}/get_supported_models"
            response = requests.get(url)
            response.raise_for_status()
            supported_models = response.json()["supported_models"]

            model_type = st.pills(
                "Choose model",
                supported_models,
                selection_mode="single",
            )

            if model_type is not None:
                with st.spinner(f"Loading {model_type} for inference..."):
                    url = f"{BASE_URL}/load_model"
                    response = requests.post(url, json={"model_type": model_type})
                    response.raise_for_status()

                    st.session_state["load_model"] = response.json()
            else:
                if "load_model" in st.session_state:
                    url = f"{BASE_URL}/unload_model"
                    response = requests.post(url)
                    response.raise_for_status()

                    del st.session_state["load_model"]

        with st.container():
            st.header("Data")

            uploaded_file = st.file_uploader("Upload data", type=["csv"])
            if uploaded_file is not None:
                with st.spinner("Uploading data for inference..."):

                    url = f"{BASE_URL}/upload_data"
                    files = {"file": ("data-inference.csv", uploaded_file, "text/csv")}
                    try:
                        response = requests.post(url, files=files)
                        response.raise_for_status()
                    except requests.exceptions.HTTPError:
                        st.error(response.json()["detail"])
                    else:
                        st.session_state["upload_data"] = response.json()

                        url = f"{BASE_URL}/load_data"
                        response = requests.post(url, json={"data_split": "inference"})
                        response.raise_for_status()

                        st.session_state["load_data"] = response.json()
            else:
                if "load_data" in st.session_state:
                    url = f"{BASE_URL}/unload_data"
                    response = requests.post(url)
                    response.raise_for_status()

                    del st.session_state["load_data"]

        with st.container():
            st.header("Status")

            model_ready = "load_model" in st.session_state
            data_ready = "load_data" in st.session_state

            st.session_state["ready"] = model_ready and data_ready

            if model_ready and data_ready:
                st.success("Ready for inference!", icon="✅")
            elif not model_ready and data_ready:
                st.info("Almost there... Choose model for inference.", icon="ℹ️")
            elif model_ready and not data_ready:
                st.info("Almost there... Upload data for inference.", icon="ℹ️")
            else:
                st.info("Not ready yet... Choose model and upload data.", icon="ℹ️")

    with st.container():
        st.header("Description")

        st.markdown(
            """
            Streamlit app for the Watch Pricing project.
            """
        )

    if st.session_state["ready"]:

        with st.container():
            st.header("Evaluation")

            url = f"{BASE_URL}/get_metrics"
            response = requests.get(url)
            response.raise_for_status()
            metrics = response.json()

            url = f"{BASE_URL}/evaluate"
            response = requests.post(url)
            response.raise_for_status()
            metrics_inference = response.json()

            metrics.append(metrics_inference)
            metrics = pd.DataFrame(metrics)

            st.subheader("Metrics")

            # st.dataframe(metrics)

            url = f"{BASE_URL}/get_supported_metrics"
            response = requests.get(url)
            response.raise_for_status()
            supported_metrics = response.json()["supported_metrics"]

            metric_type = st.pills(
                "Choose metric",
                supported_metrics,
                selection_mode="single",
                default="r2_score",
            )

            if metric_type is not None:
                fig = px.bar(
                    metrics,
                    x=metric_type,
                    y="data_split",
                    category_orders={"data_split": metrics["data_split"]},
                    # color="data_split",
                    title="Metrics by Data Split",
                )
                st.plotly_chart(fig)

            st.subheader("Custom metric")

            metric_name = st.text_input("Metric name", "explained_variance")
            if metric_name is not None:
                url = f"{BASE_URL}/compute_metric"
                response = requests.post(url, json={"metric_name": metric_name})
                response.raise_for_status()
                metric_value = response.json()["metric_value"]

                st.metric("Metric value", round(metric_value, 4))

        with st.container():
            st.header("Interpretation")

            url = f"{BASE_URL}/get_dtypes"
            response = requests.get(url)
            response.raise_for_status()
            dtypes = response.json()

            url = f"{BASE_URL}/get_importance_values"
            response = requests.get(url)
            response.raise_for_status()
            importance_values = response.json()
            importance_values = pd.DataFrame(importance_values)

            url = f"{BASE_URL}/get_dependence_values"
            response = requests.get(url)
            response.raise_for_status()
            dependence_values = response.json()

            st.subheader("Permutation Importance")

            # st.dataframe(importance_values)

            view_options = ["All", "Top 10", "Bottom 10"]
            view = st.pills(
                "Choose features",
                view_options,
                selection_mode="single",
                default="All",
            )

            if view is not None:

                if view == "All":
                    _importance_values = importance_values
                elif view == "Top 10":
                    _importance_values = importance_values.head(10)
                elif view == "Bottom 10":
                    _importance_values = importance_values.tail(10)
                else:
                    _importance_values = None

                fig = px.bar(
                    _importance_values,
                    x="importance_mean",
                    y="feature",
                    category_orders={"feature": importance_values["feature"]},
                    # color="importance_mean",
                    title="Permutation Importances per Feature",
                )
                st.plotly_chart(fig)

            st.subheader("Partial Dependence")

            features = list(dependence_values.keys())
            feature = st.selectbox("Choose feature", features)

            if feature is not None:
                _dependence_values = pd.DataFrame(dependence_values[feature])
                dtype = dtypes[feature]

                if dtype in ["category", "bool"]:
                    fig = px.bar(
                        _dependence_values,
                        x="grid_values",
                        y="average",
                        labels={
                            "average": "average_price_pred",
                            "grid_values": feature,
                        },
                        category_orders={
                            "grid_values": _dependence_values["grid_values"],
                        },
                        # color="data_split",
                        title="Partial Dependence per Feature",
                    )
                else:
                    fig = px.line(
                        _dependence_values,
                        x="grid_values",
                        y="average",
                        labels={
                            "average": "average_price_pred",
                            "grid_values": feature,
                        },
                        # color="data_split",
                        title="Partial Dependence per Feature",
                    )

                st.plotly_chart(fig)


if __name__ == "__main__":
    main()
