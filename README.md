# SignSense: A Neural Network Approach to Sign Language Recognition 

## Project Overview
This project uses Convolutional Neural Networks (CNN) and OpenCV to create a real-time sign language recognition system. It's designed to bridge the communication gap for the deaf and hard-of-hearing community by translating sign language gestures into text or speech. Our goal is to enhance accessibility and interaction within digital spaces.

## Features
- **Real-Time Detection**: Captures live video streams from a webcam for gesture recognition.
- **High Accuracy**: Utilizes a pre-trained CNN model for accurate sign language gesture classification.
- **User-Friendly**: Easy to use with a straightforward interface for all users.

## Installation

### Prerequisites
- Python 3.x
- Pip Package Manager

### Dependencies
Install the necessary libraries using pip:
```bash
pip install tensorflow keras opencv-python numpy
```
Clone the Repository:
```bash
git clone https://github.com/Kumudkohli/Sign-Language-Recognition
cd Sign-Language-Recognition
```
## Dataset
https://www.kaggle.com/datamunge/sign-language-mnist

## Usage
To launch the sign recognition application, execute:
```bash
python ROIinOpenCV.py
```
Ensure your gestures are within the webcam's field of view. The system will recognize and display the corresponding sign interpretations in real-time.

## Model Training 
The CNN model is trained on a comprehensive sign language dataset. For custom training:

1. Arrange your dataset in the specified format.
2. Modify the model parameters in train_model.py as needed.
3. Execute python train_model.py to commence training.

## Contributing
We welcome contributions to improve the project. Here's how you can contribute:

1. Fork the repository.
2. Create a new branch: git checkout -b feature-branch.
3. Commit your changes: git commit -am 'Add some feature'.
4. Push to the branch: git push origin feature-branch.
5. Submit a pull request.
