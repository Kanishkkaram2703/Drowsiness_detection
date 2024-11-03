# Drowsiness_detection

Here’s a detailed description for your driver drowsiness detection project with GUI and dataset information, ideal for a README file on GitHub.

---

# Driver Drowsiness Detection System

This project aims to provide a real-time solution for detecting driver drowsiness and alerting them to prevent accidents. By analyzing eye state (open or closed) using a machine learning model, this system accurately identifies signs of drowsiness and triggers an alarm when necessary. The project features an interactive GUI and utilizes a dataset from Kaggle for model training and testing.

## Features

- **Real-time Video Capture**: Uses the system camera to capture video of the driver’s face.
- **Eye Detection**: Detects both left and right eyes using pre-trained Haar cascades.
- **Drowsiness Prediction**: Predicts eye state (open or closed) using a CNN model trained on a labeled dataset.
- **Alarm System**: Sounds an alarm when drowsiness is detected, based on consecutive frames showing closed eyes.
- **User-Friendly GUI**: An interactive GUI is provided to control and monitor the drowsiness detection process.

## Dataset

The dataset used in this project is the [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) from Kaggle. It contains labeled images of open and closed eyes, essential for training a machine learning model to detect eye state accurately. The dataset includes a balanced set of images for both classes (`Open` and `Closed`), making it ideal for classification tasks in drowsiness detection.

## Project Structure

- `main.py`: The main script that handles the video capture, GUI, and drowsiness detection logic.
- `drowsiness_.h5`: Pre-trained model for eye state classification.
- `alarm.mp3`: Sound file that plays when drowsiness is detected.
- Haar cascades for face and eye detection: XML files for face and eye detection using OpenCV.

## How It Works

1. **Face and Eye Detection**: The system first detects the driver’s face and locates their eyes using Haar cascades for both left and right eyes.
2. **Eye State Prediction**: Each detected eye is processed and passed through a convolutional neural network (CNN) model to predict whether the eye is open or closed.
3. **Drowsiness Check**: The system tracks consecutive frames where both eyes are predicted as closed. When a threshold (e.g., 10 consecutive frames) is reached, the system determines the driver is drowsy.
4. **Alarm Trigger**: An alarm sound plays to alert the driver, helping them to stay alert.

## GUI Overview

The GUI, built using Tkinter, allows users to start and stop the drowsiness detection system. It displays the live camera feed and the system status, including whether the driver’s eyes are open or closed. In the case of drowsiness detection, the GUI updates with a warning message, and an alarm is triggered.

### Key GUI Components

- **Video Display**: Displays the live video feed from the camera with visual indicators for detected eyes.
- **Status Indicator**: Shows whether the system is actively monitoring and if the driver is drowsy.
- **Control Buttons**: Allows the user to start or stop the drowsiness detection process.

## Model Training and Preprocessing

The CNN model used for eye state classification was trained on the Kaggle dataset mentioned above. The dataset images were preprocessed to match the input size requirements of the model, converted to grayscale, and normalized to improve classification accuracy.

## Requirements

To run this project, install the following Python libraries:

```bash
pip install tensorflow keras opencv-python-headless numpy pillow playsound
```

Ensure that your system has a webcam and audio output for real-time detection and alarm functionality.

## Getting Started

1. Clone this repository.
2. Ensure you have all required libraries installed.
3. Download and place the model file (`drowsiness_.h5`) and Haar cascade XML files in the project directory.
4. Run the application:

```bash
python main.py
```

5. Use the GUI to start the drowsiness detection. The application will display live video, and if the driver’s eyes remain closed for an extended period, it will trigger an alarm.

## Future Enhancements

- **Improved Accuracy**: Experiment with advanced deep learning models to improve drowsiness detection accuracy.
- **Mobile Integration**: Develop a mobile application version for wider usability in vehicles.
- **Additional Facial Cues**: Incorporate other drowsiness cues such as yawning or head position.

## License

This project is open-source and available under the [MIT License](LICENSE).

--- 
