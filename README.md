#Crop-Prediction-Using-ML-and-Analysis-of-Soil-Nutrients-using-Satellite-imaging-and-IoT
An Web Application developed using Streamlit to predict the suitable crop for a particular land. This is predicted by using Sentinal and Landsat images.

Architecture used for crop prediction using ML
![image](https://github.com/user-attachments/assets/ade15080-074a-46d3-9910-07ef020b397e)



Architecture used for Analysis of Soil Nutrients using IOT
![image](https://github.com/user-attachments/assets/7980f857-c95c-4188-adc0-7c3b7a369bd8)


IOT Component
* Connection for the circuit(Change needs to be done by removing NRF24L01)
![image](https://github.com/user-attachments/assets/57092fcc-dc04-4c47-b7c6-8d98b505c75c)


* This is the Output of IOT Component
IoT-Output

Architecture used for Satellite Imaging
satelliteimage

Overall Architecture
overall

Dataset
The data used to train the model was collected from the Crop Prediction dataset. The dataset consists of 2200 samples of 22 different crops whose predictions are made using 7 features: nitrogen, phosphorus, potassium, and pH content of the soil, temperature, humidity and rainfall. The dataset is perfectly balanced, with each crop having 100 samples.

crop

Attributes information:
N - Ratio of Nitrogen content in soil

P - Ratio of Phosphorous content in soil

K - Ratio of Potassium content in soil

Temperature - temperature in degree Celsius

Humidity - relative humidity in %

ph - ph value of the soil

Rainfall - rainfall in mm

Procedure
Create an ML model for Crop prediction using the code provided in Crop Prediction module.

We are using the LGBM algorithm since it shows higher accuracy than other algorithms.

Open a new project in Google Earth Engine.

Download Landsat and Sentinal images by using the code which is given in Google Earth Engine module.

Then Create an UI to display using the code in streamlit(new.py) module.

Insert the necessary ML models as per the code and change the file path accordingly. (as per ML models module)

Then after the new.py runs successfully follow the Steps provided in the OUTPUT of this readme.

Accuracy of Crop Prediction Algorithms
accuracy

Confusion Matrix
confusion

Web UI
webui-1 webui-2

OUTPUT
We need to upload Sentinal and landsat images seperately.

output1

output2

Step 1: We provide Sentinal image to get the analysis of soil nutrients such as " nitrogen, phosphorus, potassium, temperature, humidity ".

Step 2: We Provide Landsat image to get the analysis of soil nutrients such as " Ph "

Step 3: Provide Rainfall Input.

Step 4: Click Predict Button to get the Crop Prediction result.

For further doubts reffer Documentation Module
