# 🚁 SkyGuard Nexus: Aerial Threat Classification AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-green)
![Accuracy](https://img.shields.io/badge/Accuracy-98.14%25-brightgreen)

## 📌 Project Overview
As commercial and recreational drones become increasingly common, they pose significant physical and security risks to restricted airspaces, such as commercial airports, military bases, and protected wildlife reserves. Traditional radar systems often struggle to differentiate between a small mechanical drone and a large biological bird. 

**SkyGuard Nexus** is an end-to-end Computer Vision and Deep Learning pipeline designed to solve this. By analyzing raw aerial imagery, the system utilizes a highly optimized **MobileNetV2** architecture to instantly classify the object with **98.14% accuracy**. The project includes a production-ready Streamlit dashboard designed for real-time edge deployment.

## ✨ Key Features
* **🎯 Precision Single Target Scan:** Upload individual aerial images for deep convolutional analysis.
* **📂 High-Volume Batch Processing:** Drag and drop entire folders of drone patrol footage for automated bulk classification.
* **🔍 AI Vision X-Ray (Explainable AI):** Toggle Canny Edge Detection to visualize exactly how the Convolutional Neural Network extracts geometric shapes and metallic edges.
* **📈 Command Center Analytics:** Interactive, real-time data visualizations built with Plotly to track historical detections.
* **🔄 MLOps "Human-in-the-Loop":** Integrated feedback database allowing operators to flag false predictions for future model retraining.

## 🧠 Model Architecture & Performance
Three different neural network architectures were benchmarked to find the optimal balance between accuracy and computational weight for edge-device deployment:

1. **Custom CNN (Baseline):** 81.86% Accuracy
2. **ResNet50 (Transfer Learning):** 72.09% Accuracy
3. **🏆 MobileNetV2 (Transfer Learning): 98.14% Accuracy**

**Why MobileNetV2?** MobileNetV2 was selected as the final deployment model because of its exceptional accuracy and lightweight architecture (utilizing depthwise separable convolutions). This allows the system to run in real-time on edge hardware (like local camera servers) without requiring massive cloud GPU infrastructure.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (cv2), Pillow (PIL)
* **Data Processing:** NumPy, Pandas
* **Frontend & UI:** Streamlit
* **Data Visualization:** Plotly, Seaborn, Matplotlib
