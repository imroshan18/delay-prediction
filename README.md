# 🚆 Train Delay Prediction App

An interactive web application built with **Streamlit** that predicts whether a **train will be delayed** based on various journey and operational features.

Try it live here:  
👉 **[Train Delay Prediction App](https://delay-prediction-8q9gkkdjhhkgeadiridajx.streamlit.app/)**


## 🔍 About the Project

The **Train Delay Prediction App** is a machine learning–powered web app that predicts whether a **train is likely to arrive late** based on user inputs.

Typical inputs can include:

- Departure and arrival stations  
- Scheduled departure/arrival time  
- Day of week / date  
- Distance or route info  
- Historical delay behavior  
- Weather or other operational factors

The app provides a simple UI so users can enter trip details and instantly see whether the model predicts **“On Time”** or **“Delayed”**.

---

## ✨ Features

- 🧠 **ML-based prediction** of train delay (On Time vs Delayed)
- 🧾 **Form-based inputs** for journey details
- 📊 Clear **prediction result** and explanation text
- 🌐 **Deployed online** via Streamlit Cloud
- ⚙️ Modular code structure, easy to extend or retrain with new data

---

## 🛠 Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** scikit-learn (classifier model)
- **Data Handling:** pandas, numpy
- **Model Persistence:** pickle / joblib (saved model file)


## 📁 Project Structure


```bash
.
├── app.py                 # Main Streamlit app
├── model.pkl              # Trained ML model
├── requirements.txt       # Python dependencies

'''

