# Drowning Detection Application

A real-time drowning detection application using a YOLOv8-based deep learning model. This project processes video or image inputs to detect events related to drowning, swimming, or being out of water. When a drowning event is detected, the system draws bounding boxes around the detected area and plays an alert sound.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [License](#license)

## Overview

This project implements a real-time detection system that:
- Processes both images and videos to detect drowning events.
- Differentiates between `drowning`, `swimming`, and `out of water` events.
- Plays an alert sound (`sound/alarm.wav`) whenever a drowning event is detected.

The system leverages a YOLOv8 model that has been previously trained using a dataset (e.g., from Roboflow) containing images with the three classes. The application uses OpenCV for video processing and the `playsound` module for audio alerts.

## Features

- **Real-Time Detection:** Process live video streams or input files to detect events.
- **Multi-Class Support:** Identify and display bounding boxes for `drowning`, `swimming`, and `out of water` events.
- **Audible Alerts:** An alarm sound is played when a drowning event is detected.
- **User-Friendly:** Simple command-line interface to specify the input source.

## Prerequisites

- **Python 3.7+**
- Required Python packages:
  - `opencv-python`
  - `ultralytics`
  - `playsound`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies:**

    ```bash
    pip install opencv-python ultralytics playsound
    ```

3. **Model and Sound Setup:**

Ensure that trained YOLOv8 model file (best.pt) is placed in the same directory as app.py.
Place the alert sound file (alarm.wav) inside a folder named sound (i.e., sound/alarm.wav).

## Usage
Run the application by specifying the input file (image or video):

- For an image:
    ```bash
    python app.py --source path/to/your/image.jpg
    ```
- For a video:
    ```bash
    python app.py --source path/to/your/video.mp4
    ```
When executed, the application will display the input with annotated bounding boxes. If a drowning event is detected, an audible alarm will be played.

## Project Structure
    .
    ├── app.py             # Main application file
    ├── best.pt            # Trained YOLOv8 model weights
    ├── sound
    │   └── alarm.wav      # Alert sound file
    └── README.md          # Project documentation

## Model Training
The YOLOv8 model used in this application was trained on a dataset from Roboflow containing images labeled as drowning, swimming, and out of water. You can see the training code in "Drown_detect.ipynb".

## License
This project is licensed under the MIT License.
