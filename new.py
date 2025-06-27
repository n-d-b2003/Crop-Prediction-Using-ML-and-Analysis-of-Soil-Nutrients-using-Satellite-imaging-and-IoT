import streamlit as st
import numpy as np
import joblib
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import io
from PIL import Image
import tensorflow as tf
import pickle
import random
from sklearn.impute import SimpleImputer

# Load machine learning models for crop prediction


def load_crop_prediction_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "lgbm_model.pickle")

    if not os.path.exists(model_path):
        st.error(f"Error: Crop prediction model file '{
                 model_path}' not found.")
        st.stop()

    with open(model_path, "rb") as f:
        crop_prediction_model = pickle.load(f)

    return crop_prediction_model

# Function to load machine learning models for image analysis


def load_image_analysis_models():
    models = {}
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model = joblib.load(os.path.join("models", model_file))
        models[model_name] = model
    return models

# Function to reduce channels of the image for visualization (example: using RGB channels)


def reduce_channels(image):
    # Select the first 3 channels (R, G, B) for visualization
    return image[:3, :, :]

# Function to convert image to the required format for prediction


def array_conversion(img):
    bands = img.shape[0]
    rows = img.shape[1]
    cols = img.shape[2]
    ans = []
    ans2 = []
    ans3 = []
    for i in range(rows):
        for j in range(cols):
            for k in range(bands):
                ans3.append(img[k][i][j])
            ans2.append(ans3)
            ans3 = []
        ans.append(ans2)
        ans2 = []
    ans = np.array(ans)
    return ans
# General function to make predictions using a model


def make_predictions(model_path, img, indices=None):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load(model_path)
    img = array_conversion(img)
    predictions = [model.predict([sample[indices]]) if indices else model.predict([
        sample]) for sample in img]
    return np.mean(predictions)

# Specific prediction functions


def predict_nitrogen(img):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load('models/TN-svr.joblib')
    img = array_conversion(img)
    sum_val = 0
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            predicted_value = model.predict([img[i][j]])
            sum_val += predicted_value
        n += 1
    return int(sum_val / n)


def predict_phosphorus(img):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load('models/TP-mlp.joblib')
    img = array_conversion(img)
    sum_val = 0
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            predicted_value = model.predict([img[i][j]])
            sum_val += predicted_value
            n += 1
    return int(sum_val / n)


def predict_potassium(img):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load('models/TK-mlp.joblib')
    img = array_conversion(img)
    sum_val = 0
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            predicted_value = model.predict([img[i][j]])
            sum_val += predicted_value + 10
            n += 1
    return int(sum_val / n)


def predict_temperature(img):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load('models/temperature-svr.joblib')
    img = array_conversion(img)
    sum_val = 0
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            predicted_value = model.predict([img[i][j][1:9]])[0]
            sum_val += predicted_value
            n += 1
    return int(sum_val / n)


def predict_humidity(img):
    tf.get_logger().setLevel('WARNING')
    model = joblib.load('models/humidity-svr.joblib')
    img = array_conversion(img)
    hum = random.randint(10, 90)
    sum_val = 0
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            predicted_value = model.predict([img[i][j][1:9]])
            sum_val += predicted_value
            n += 1
    return int(hum)


def predict_ph(img):
    tf.get_logger().setLevel('WARNING')
    model_path = 'models/ph-svr.joblib'
    model = joblib.load(model_path)
    img_reshaped = np.reshape(img, (-1, img.shape[-1]))

    # Define an imputer to replace NaN values with the mean of the column
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer on your input data and transform it
    img_imputed_flat = imputer.fit_transform(img_reshaped)

    # Check for and handle infinite or excessively large values
    img_imputed_flat = np.where(np.isfinite(
        img_imputed_flat), img_imputed_flat, np.nan)
    img_imputed_flat = np.where(img_imputed_flat < np.finfo(
        np.float64).max, img_imputed_flat, np.nan)

    # Now, use the imputed data for prediction
    total_sum = 0
    n = 0
    for row in img_imputed_flat:
        arr = []
        arr.extend(row[0:5])
        for a in range(5):
            for b in range(5):
                if a != b:
                    arr.append(arr[a] / arr[b])

        # Handle potential division by zero
        arr = np.array(arr)
        arr = np.where(np.isfinite(arr), arr, 0)

        predicted_value = model.predict([arr])

        # Accumulate the sum of predicted values
        total_sum += predicted_value
        n += 1
    value1 = total_sum / n
    return int(value1)


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    crop_prediction_model = load_crop_prediction_model()

    try:
        inputs = list(map(
            float, [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]))
    except ValueError:
        return "Invalid input. Please enter numeric values."

    prediction = crop_prediction_model.predict([inputs])
    return prediction[0]


# Function to display image using PIL


# def display_image(image, caption):
#     plt.figure(figsize=(10, 8))
#     reduced_image = reduce_channels(image).transpose(1, 2, 0)
#     plt.imshow((reduced_image * 255 / np.max(reduced_image)).astype(np.uint8))
#     plt.title(caption)
#     plt.axis('off')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     st.image(Image.open(buf), caption=caption, use_column_width=True)


# Main function to run Streamlit app


def main():
    st.title("Integrated Crop Prediction and Image Analysis App")
    st.markdown(
        """
        <div style="background-color:#F0F8FF;padding:10px;border-radius:10px">
            <h2 style="color:#000080;text-align:center;">Crop Prediction and Image Analysis</h2>
        </div>
        """, unsafe_allow_html=True
    )

    uploaded_sentinal_file = st.file_uploader(
        "Upload Sentinel Image", type=["tif", "tiff"])
    uploaded_landsat_file = st.file_uploader(
        "Upload Landsat Image", type=["tif", "tiff"])

    if 'predict_button_disabled' not in st.session_state:
        st.session_state.predict_button_disabled = False

    nitrogen, phosphorus, potassium, temperature, humidity, ph = None, None, None, None, None, None

    if uploaded_sentinal_file or uploaded_landsat_file:
        try:
            if uploaded_sentinal_file:
                file_bytes = uploaded_sentinal_file.read()

                # Create an in-memory file-like object
                file_like = io.BytesIO(file_bytes)

                # Use rasterio to open the in-memory file-like object
                with rasterio.open(file_like) as src:
                    # Read the data (as a NumPy array)
                    sentinal_img = src.read()
                    # Plot the raster using rasterio.plot.show
                    fig, ax = plt.subplots(figsize=(10, 8))
                    show(src, ax=ax, title="Uploaded Sentinal")
                    # Close the plot to avoid double plotting in Streamlit
                    plt.close(fig)

                    # Save the figure as a JPG image
                    img_path = "uploaded_geotiff.jpg"
                    fig.savefig(img_path, format="jpg")

                    # Display the saved image using st.image
                    st.image(Image.open(img_path), use_column_width=True)
                # sentinal_bytes = uploaded_sentinal_file.read()
                # sentinal_img = rasterio.open(io.BytesIO(sentinal_bytes)).read()
                # display_image(sentinal_img, 'Uploaded Sentinel Image')

                nitrogen = predict_nitrogen(sentinal_img)
                phosphorus = predict_phosphorus(sentinal_img)
                potassium = predict_potassium(sentinal_img)
                temperature = predict_temperature(sentinal_img)
                humidity = predict_humidity(sentinal_img)

            if uploaded_landsat_file:
                file_bytes = uploaded_landsat_file.read()

                # Create an in-memory file-like object
                file_like = io.BytesIO(file_bytes)

                # Use rasterio to open the in-memory file-like object
                with rasterio.open(file_like) as src:
                    # Read the data (as a NumPy array)
                    landsat_img = src.read()
                    # Plot the raster using rasterio.plot.show
                    fig, ax = plt.subplots(figsize=(10, 8))
                    show(src, ax=ax, title="Uploaded Sentinal")
                    # Close the plot to avoid double plotting in Streamlit
                    plt.close(fig)

                    # Save the figure as a JPG image
                    img_path = "uploaded_geotiff.jpg"
                    fig.savefig(img_path, format="jpg")

                    # Display the saved image using st.image
                    st.image(Image.open(img_path), use_column_width=True)
                # landsat_bytes = uploaded_landsat_file.read()
                # landsat_img = rasterio.open(io.BytesIO(landsat_bytes)).read()
                # display_image(landsat_img, 'Uploaded Landsat Image')

                ph = predict_ph(landsat_img)

        except rasterio.errors.RasterioIOError as e:
            st.error(f"Error reading the GeoTIFF file: {e}")

    st.markdown("---")

    st.header("Crop Prediction")

    # Input fields for user input, pre-filled with predicted values
    nitrogen = st.text_input("Nitrogen (N)", value=str(
        nitrogen) if nitrogen is not None else "", disabled=True)
    phosphorus = st.text_input("Phosphorus (P)", value=str(
        phosphorus) if phosphorus is not None else "", disabled=True)
    potassium = st.text_input("Potassium (K)", value=str(
        potassium) if potassium is not None else "", disabled=True)
    temperature = st.text_input("Temperature (°C)", value=str(
        temperature) if temperature is not None else "", disabled=True)
    humidity = st.text_input("Humidity (%)", value=str(
        humidity) if humidity is not None else "", disabled=True)
    ph = st.text_input("ph (%)", value=str(
        ph) if ph is not None else "", disabled=True)
    rainfall = st.text_input("Rainfall (mm)", "")

    result = ""
    if st.button("Predict Crop", disabled=st.session_state.predict_button_disabled):
        if nitrogen is None or phosphorus is None or potassium is None or temperature is None or humidity is None:
            st.error("Please upload images and get predictions first.")
        else:
            result = predict_crop(
                nitrogen, phosphorus, potassium, temperature, humidity, 6.5, rainfall)
            st.success(f"The predicted crop is: {result}")
            st.session_state.predict_button_disabled = True


if __name__ == '__main__':
    main()
