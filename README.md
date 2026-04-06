  # 🦴 XRay Bone Fracture Classification CNN & TransferLearning
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Project-green)
![Tensorflow](https://img.shields.io/badge/TensorFlow-DL-orange)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

#### 🔗 Live Application: https://xay-bone-fracture-detection.streamlit.app/
---

## 📖 1. Project Overview

This project builds a **Deep Learning** system to automatically detect bone fractures from X-ray images, classifying them as **Fractured** or **Normal**.

By combining a **custom CNN** with **Transfer Learning**, the model achieves high diagnostic accuracy. The final solution is deployed as a **Streamlit** web app, providing healthcare professionals with a real-time, AI-powered tool for faster and more reliable diagnosis.

---

## 📉 2. Problem Statement

Manual X-ray analysis is often **slow**, **subjective**, and **prone to error**, especially in high-volume clinics. These delays can lead to diagnostic oversights and slower patient care.

This project develops an **AI-powered system** to:

* **Speed up detection** for instant fracture identification.
* **Reduce human error** by providing a reliable "second opinion."
* **Streamline workflows** to help radiologists prioritize urgent cases.

---
## 🎯 3. Project Objectives

The primary goal is to develop a **robust Deep Learning model** for high-accuracy bone fracture detection.

> **Key Deliverables:**
* ***Model Development:*** Build a custom **CNN** and implement **Transfer Learning** (e.g., ResNet, VGG).
* ***Performance Benchmarking:*** Compare architectures to identify the most reliable model.
* ***Comprehensive Evaluation:*** Measure success using **Accuracy, Precision, Recall, and F1-score**.
* ***Interactive Deployment:*** Launch a **Streamlit** web app for real-time X-ray analysis.

---

## 📊 4. Data Understanding

>  **Dataset Source**
* **Source:** [Kaggle - Bone Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)


> **Dataset Characteristics**
* **Scope:** Includes X-ray images from multiple anatomical regions (lower limbs, upper limbs, lumbar, hips, knees, etc.).
* **Total Images:** 10,580 radiographic images.
* **Format:** Images are organized into standard folders for machine learning workflows:
    * **Train:** 9,246 images (approx. 87%)
    * **Validation:** 828 images (approx. 8%)
    * **Test:** 506 images (approx. 5%)

> **Feature Breakdown**
* **Input Features:** Raw pixel data from grayscale X-ray images of varying sizes and orientations.
* **Target Labels:** Binary classification:
    * `Fractured (0)`: Images containing a bone break or fracture.
    * `Normal (1)`: Images showing intact bone structures without fractures.

---

## ⚙️ 5. Data Pipeline & Generators

To efficiently load and preprocess the image dataset, **Image Data Generators** were used. Generators helped in handling large image datasets without loading everything into memory at once.

- Used `ImageDataGenerator` from TensorFlow/Keras
- Automatically reads images from directory structure
- Assigns labels based on folder names (Fractured / Normal)
- Generates batches of images during training.

> **Implementation Details:**
* ***Preprocessing:*** Applied `rescale=1./255` to normalize pixel values between 0 and 1.
* ***Standardization:*** Set a uniform `target_size` for all X-rays to ensure consistent input dimensions.
* ***Configuration:*** Used `class_mode='binary'` for the two-class output and optimized `batch_size` for training stability.
* ***Partitioning:*** Created distinct generators for **Training**, **Validation**, and **Testing** to maintain strict data leakage prevention.

---
## 🔍 6. Exploratory Data Analysis (EDA)

EDA was conducted to identify key visual patterns and statistical properties of the X-ray dataset.

* ***Sample Visualization:*** Compared **`Fractured`** (cracks/discontinuities) vs. **`Normal`** (smooth/continuous) images to verify data quality.
* ***Class Distribution:*** Analyzed the ratio of samples to prevent model bias and determine if data augmentation or class weighting was necessary.
* ***Image Statistics:*** Computed **`Mean`**, **`Median`**, and **`Std Dev`** of pixel intensities, confirming high variability in brightness.
* ***Pixel Distribution*** Observed a concentration in low-to-medium intensity, highlighting the need for **`Normalization (Rescaling)`**.

---

## 🧪 7. Image Processing & Feature Analysis

Advanced computer vision techniques were used to extract structural features from the radiographs.

* ***Sobel & Canny Edge Detection:*** Highlighted bone boundaries and sharp intensity transitions to isolate fine fracture lines.
* ***HOG (Histogram of Oriented Gradients):*** Extracted features based on gradient orientation, helping the model recognize the "shape" of a break regardless of lighting.
---

## ⚙️ 8. Data Preprocessing

Data preprocessing ensures X-ray images are in an optimized, consistent format for stable training and high diagnostic performance.

> #### 8.1 Resizing & Normalization
* ***Uniform Resizing:*** Standardized all images to fixed dimensions (e.g. 224 X 224) for Uniformity.
* ***Pixel Rescaling:*** Normalized values from [0, 255] to [0, 1] using `rescale=1./255`.


> #### 8.2 Data Augmentation

To improve generalization and reduce overfitting, real-time transformations were applied to the training set:
* ***Techniques:*** Rotation, Width/Height Shifting, Zooming, and Horizontal Flipping.

---

## 🤖 9. Baseline CNN Model

A custom Convolutional Neural Network (CNN) was built as a baseline model to classify X-ray images into Fractured and Non-Fractured categories.

> #### Architecture Overview:

- Convolutional layers for feature extraction
- Activation function: ReLU
- Pooling layers for dimensionality reduction
- Fully connected (Dense) layers for classification
- Output layer with Sigmoid activation (binary classification)
---

## 🚀 10. Transfer Learning 
To maximize diagnostic accuracy, **Transfer Learning** was implemented using three state-of-the-art architectures. By leveraging pre-trained weights from ImageNet, these models effectively extract complex features from X-ray imagery.

> #### **10.1 Architectures Implemented**
* ***MobileNetV2:*** Optimized for efficiency and speed without sacrificing depth.
* ***ResNet152V2:*** A deep residual network designed to eliminate the vanishing gradient problem.
* ***VGG19:*** A classic deep architecture known for its simple, consistent design.


> #### **10.2 Performance Benchmark**

| Model       | Loss   | Accuracy     |
| ----------- | ------ | ------------ |
| MobileNetV2 | 0.0143 | **99.60%** ✅ |
| ResNet152V2 | 0.0402 | 99.01%       |
| VGG19       | 0.2069 | 90.91%       |



---

## **📊 11. Model Comparison**
**Baseline CNN** benchmarked  against three **Transfer Learning** architectures to identify the most reliable diagnostic tool.

> #### **11.1 Performance Summary**

| Model | Accuracy | Loss | Remarks |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | **99.60%** | **0.0143** | **Optimal:** Highest accuracy and most efficient. ✅ |
| **ResNet152V2** | 99.01% | 0.0402 | **Strong:** High accuracy but computationally heavy. |
| **Baseline CNN** | 95.85% | 0.1101 | **Good:** Solid but lacks deep feature extraction. |
| **VGG19** | 90.91% | 0.2069 | **Weak:** Lower accuracy and prone to overfitting. |

>#### **11.2 Comparative Analysis**

Models were evaluated on  Test Set with **Accuracy** and **False Negatives (FN)**, as missing a fracture is a critical clinical failure.

* **`Baseline CNN:`** Strong performance (~95.8%) for a custom-built model. It identifies obvious fractures effectively but lacks the deep feature extraction required for more subtle, complex cases
* **`MobileNetV2:`** Achieved near-perfection (~99.6%) with **zero misclassifications** in testing. Its efficient architecture captures fine bone textures perfectly.
* **`ResNet152V2:`** Delivered strong results (~99.0%) but requires more computational power.
* **`VGG19:`** Showed the highest error rate (~9.1%), failing to generalize as well to X-ray contrast.

## 🏆 12.  Best Model Selection

> **`MobileNetV2`** was selected as the best-performing model.

It achieved the highest accuracy and, more importantly, zero false positives, meaning no fracture cases were missed.

This makes MobileNetV2 the most reliable model for this problem, especially in a medical context where missing a fracture can have serious consequences.

---

## 📌 13. Conclusion

This project demonstrates the effectiveness of **Deep Learning** in detecting bone fractures from X-ray images. Multiple models were evaluated, and **MobileNetV2** achieved the best performance with **99.60% accuracy**.

The use of **transfer learning, data preprocessing, and augmentation** significantly improved results. The final model was successfully deployed using **Streamlit**, enabling real-time predictions.


---

## 👩‍💻 14. Author

> **Shravani More**

Computer Science & Electronics Student 


