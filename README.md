Driver Drowsiness Detection System

📌 Project Description

This project is a Driver Drowsiness Detection System designed to enhance road safety by identifying signs of driver fatigue in real-time. The system utilizes machine learning models to analyze driver behavior and detect drowsiness, providing alerts when necessary.

🚀 Features

Real-time drowsiness detection using a machine learning model implemented in Python.

Utilizes facial landmarks and eye movement analysis to assess driver fatigue levels.

Integration of DROZY, NTHU, UTA-RLDD datasets to train and validate the model.

🛠 Technologies Used

Python (Machine Learning Model)

OpenCV & Dlib (Facial Landmark Detection)

TensorFlow/Keras (Deep Learning Frameworks)

📂 Dataset

The model is trained on publicly available datasets:

DROZY

NTHU Drowsy Driver Dataset

UTA-RLDD (Real-Life Drowsiness Dataset)

⚡ Installation & Usage

To set up the project locally, follow these steps:

1️⃣ Clone the Repository

git clone https://github.com/alperenkoruyucu/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Model

python detect_drowsiness.py

4️⃣ Run the Web Interface

cd web-interface
npm install
npm start

📌 Future Improvements

Enhance model accuracy with additional datasets and fine-tuning.

Implement real-time video streaming for live detection.

Deploy as a cloud-based service with API integration.

🤝 Contributing

Feel free to submit issues and pull requests to improve the project!
