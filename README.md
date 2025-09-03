# Wheat Classifier

![License](https://img.shields.io/github/license/M-Tayyab06/Wheat_classifier?color=blue)
![Stars](https://img.shields.io/github/stars/M-Tayyab06/Wheat_classifier?style=social)
![Issues](https://img.shields.io/github/issues/M-Tayyab06/Wheat_classifier)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Wheat_Classifier-yellowgreen?logo=ai)


# Wheat Disease Classification using CNN

This repository contains a Flask-based Streamlit application for classifying wheat diseases using a custom-trained Convolutional Neural Network (CNN). The model can predict three classes: **Normal**, **Fusarium**, and **Unknown Disease**, based on images or videos provided by the user.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Installation and Setup](#installation-and-setup)
4. [Running the Application](#running-the-application)
5. [Usage](#usage)
6. [Model Details](#model-details)

## Project Overview

This project leverages deep learning for image classification to identify wheat diseases. The trained CNN model has been saved and is used by the application for real-time prediction of:
- Normal wheat
- Wheat affected by Fusarium
- Wheat with an unknown disease

The app allows users to upload images or videos for classification.

## Folder Structure

The repository contains the following files:

1. **`.gitignore`**: Specifies files and directories to be ignored by Git.
2. **`best_wheat_classification_model.h5`**: Saved model file in H5 format.
3. **`best_wheat_classification_model.keras`**: Saved model file in Keras format.
4. **`app.py`**: The main Streamlit application file.
5. **`requirements.txt`**: Contains the Python dependencies required to run the application.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment (optional but recommended), the python version should be 3.10.0:
   ```bash
   python3 -m venv env
   source env/bin/activate # For Linux/Mac
   env\Scripts\activate   # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To launch the application, use the following command:
```bash
streamlit run app.py
```
This will start a local Streamlit server and provide a URL to access the application in your browser.

## Usage

1. Open the application in your browser using the provided URL.
2. Upload an image or video file using the upload interface.
3. The application will process the input and classify the wheat condition as one of the three categories:
   - **Normal**
   - **Fusarium**
   - **Unknown Disease**
4. View the results and predictions in real-time.

## Model Details

The CNN model has been trained on a dataset of wheat images to classify them into the following categories:
- **Normal Wheat**: Healthy wheat with no signs of disease.
- **Fusarium**: Wheat affected by Fusarium Head Blight.
- **Unknown Disease**: Wheat showing symptoms not classified as Normal or Fusarium.

The model has been saved in both `.h5` and `.keras` formats to ensure compatibility with various frameworks.

Feel free to contribute to the project by submitting issues or pull requests. Happy coding!

