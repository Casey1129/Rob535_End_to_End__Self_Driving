import numpy as np
import cv2
from keras.models import load_model

# Load the pre-trained Keras model
model = load_model('models/model.keras')

def keras_predict(model, image):
    """
    Predicts the steering angle for a given input image using the provided model.

    Parameters:
        model (Model): Pre-trained Keras model.
        image (numpy array): Preprocessed input image.

    Returns:
        steering_angle (float): Predicted steering angle scaled by 100.
    """
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 100  # Scale the prediction
    return steering_angle

def keras_process_image(img):
    """
    Preprocesses the input image for prediction.

    Parameters:
        img (numpy array): Input image.

    Returns:
        processed_img (numpy array): Preprocessed image ready for model input.
    """
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))  # Resize to 40x40
    img = np.array(img, dtype=np.float32)  # Convert to float32
    img = np.reshape(img, (-1, image_x, image_y, 1))  # Reshape for the model
    return img

# Load the steering wheel image and initialize variables
steer = cv2.imread('resources/steering_wheel_image.jpg', 0)  # Load as grayscale
rows, cols = steer.shape
smoothed_angle = 0

# Open the video file
cap = cv2.VideoCapture('resources/run.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    # Preprocess the current frame
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))

    # Predict the steering angle
    steering_angle = keras_predict(model, gray)
    print(f"Predicted Steering Angle: {steering_angle:.2f}")

    # Display the video frame
    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))

    # Smooth the steering angle for display
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)

    # Rotate the steering wheel image
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources and close windows
cap.release()
cv2.destroyAllWindows()
