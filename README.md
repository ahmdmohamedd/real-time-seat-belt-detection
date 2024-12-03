# Real-Time Seat Belt Detection

## Overview
This project implements a real-time seat belt detection system using Convolutional Neural Networks (CNN). The model processes images and detects whether a person is wearing a seat belt or not. It has been trained on a custom dataset and can be deployed for real-time video feed processing, providing a practical solution for vehicle safety.

## Features
- **Real-time seat belt detection**: Detects if a person is wearing a seat belt from a webcam or video feed.
- **CNN model**: Uses a Convolutional Neural Network (CNN) for binary classification (seat belt or no seat belt).
- **Custom dataset**: The model is trained on a dataset containing images of people with and without seat belts.

## Dataset
The system was trained on a custom dataset consisting of images organized into two categories:
- `with_belt`: Images of people wearing seat belts.
- `without_belt`: Images of people not wearing seat belts.

The dataset is split into training, validation, and testing sets.

## Requirements
To run the project, make sure you have the following installed:
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

You can install the necessary libraries using pip:
```bash
pip install tensorflow opencv-python matplotlib numpy scikit-learn
```

## Usage

### 1. Train the Model
1. Clone this repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/real-time-seat-belt-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd real-time-seat-belt-detection
   ```
3. Open `real_time_seat_belt_detection.ipynb` in a Jupyter notebook or other compatible Python environment.
4. Run the cells to train the model.

### 2. Real-Time Detection
After training the model, you can use the real-time detection script to start the webcam feed and detect whether a person is wearing a seat belt.

Run the following code in the notebook:
```python
# Start the webcam for real-time detection
cap = cv2.VideoCapture(0)  # Use 0 for webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    label = detect_seat_belt(frame, model)
    
    # Display result
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Seat Belt Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. Model Saving and Loading
Once the model is trained, you can save it for later use:
```python
model.save('seat_belt_detection_model.h5')
```
To load the saved model later:
```python
model = models.load_model('seat_belt_detection_model.h5')
```

## Contributing
Feel free to fork this repository, create a branch, and submit pull requests for any improvements or bug fixes. Contributions are always welcome!
