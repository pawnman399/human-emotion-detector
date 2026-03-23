# human-emotion-detector
This project is a real-time human emotion recognition system built using Deep Learning, Computer Vision, and Streamlit. It detects faces from a webcam feed and classifies emotions such as Angry, Happy, and Sad. 
# Human Emotion Recognition System (Real-Time)

A real-time Human Emotion Recognition System built using Deep Learning, Computer Vision, and Streamlit. This project detects faces from a webcam feed and classifies emotions such as Angry, Happy, and Sad.

---

## Features

* Real-time emotion detection via webcam
* CNN-based deep learning model
* Fast inference using TensorFlow Lite
* Emotion classification: Angry, Happy, Sad
* Confidence score and emotion history tracking
* Interactive UI using Streamlit
* Automated dataset splitting

---

## Model Architecture

* Input: 48×48 grayscale images
* 3 Convolutional layers with MaxPooling
* Fully connected Dense layers
* Dropout for regularization
* Output layer with Softmax activation

---

## Project Structure

```
project/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── split_dataset.py
├── train_emotion_model.py
├── convert_to_tflite.py
├── real
```
