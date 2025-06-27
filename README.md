*Crop-Prediction-Using-ML-and-Analysis-of-Soil-Nutrients-using-Satellite-imaging-and-IoT
An Web Application developed using Streamlit to predict the suitable crop for a particular land. This is predicted by using Sentinal and Landsat images.

Architecture used for crop prediction using ML
![image](https://github.com/user-attachments/assets/ade15080-074a-46d3-9910-07ef020b397e)



Architecture used for Analysis of Soil Nutrients using IOT
![image](https://github.com/user-attachments/assets/7980f857-c95c-4188-adc0-7c3b7a369bd8)


IOT Component
* Connection for the circuit(Change needs to be done by removing NRF24L01)
![image](https://github.com/user-attachments/assets/57092fcc-dc04-4c47-b7c6-8d98b505c75c)


* This is the Output of IOT Component
![image](https://github.com/user-attachments/assets/fb05dd39-b176-41d8-a617-5c667b2d6ccc)


Architecture used for Satellite Imaging
satelliteimage

Overall Architecture
![image](https://github.com/user-attachments/assets/9a8ccd08-1d43-4a82-8d7c-b2d67884e014)


Dataset
The data used to train the model was collected from the Crop Prediction dataset. The dataset consists of 2200 samples of 22 different crops whose predictions are made using 7 features: nitrogen, phosphorus, potassium, and pH content of the soil, temperature, humidity and rainfall. The dataset is perfectly balanced, with each crop having 100 samples.
![image](https://github.com/user-attachments/assets/1059f6aa-8d5b-40ff-89ef-6b6926966964)


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
![image](https://github.com/user-attachments/assets/34a2fc9b-3032-470b-b222-f2460fd61e37)


Confusion Matrix
![image](https://github.com/user-attachments/assets/92967815-3218-406c-91a5-d92f3f5ab970)


Web UI
![image](https://github.com/user-attachments/assets/ce1571a5-6727-4c10-9ef8-ae858fa1ae18)

![image](https://github.com/user-attachments/assets/aa007c5f-d5dc-476f-9293-7f6265a8ed84)



OUTPUT
We need to upload Sentinal and landsat images seperately.
![image](https://github.com/user-attachments/assets/13604343-4856-4722-9f99-645b441044f5)

![image](https://github.com/user-attachments/assets/73374d4a-b825-43d8-a639-05dd6705442d)


Step 1: We provide Sentinal image to get the analysis of soil nutrients such as " nitrogen, phosphorus, potassium, temperature, humidity ".

Step 2: We Provide Landsat image to get the analysis of soil nutrients such as " Ph "

Step 3: Provide Rainfall Input.

Step 4: Click Predict Button to get the Crop Prediction result.

For further doubts reffer Documentation Module
