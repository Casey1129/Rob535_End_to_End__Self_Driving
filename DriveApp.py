import numpy as np
import cv2
from keras.models import load_model
from keras.config import enable_unsafe_deserialization

enable_unsafe_deserialization()  # Allow deserialization of unsafe Lambda layers

model = load_model('models/Autopilot.keras')


def keras_predict(model, image):
    processed = keras_process_image(image)
    # Use `predict` with a batch of size 1
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 100
    return steering_angle


def keras_process_image(img):
    image_x = 40
    image_y = 40
    # Resize and normalize the image
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


# Load the steering wheel image
steer = cv2.imread('resources/steering_wheel_image.jpg', 0)
if steer is None:
    raise FileNotFoundError("Steering wheel image not found. Please check the path: 'resources/steering_wheel_image.jpg'")
rows, cols = steer.shape
smoothed_angle = 0

# Open the video file
cap = cv2.VideoCapture('resources/run.mp4')
if not cap.isOpened():
    raise FileNotFoundError("Video file not found or cannot be opened. Please check the path: 'resources/run.mp4'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Convert the frame to HSV and process the V (grayscale) channel
    try:
        gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 2], (40, 40))
    except Exception as e:
        print(f"Error processing frame: {e}")
        break

    steering_angle = keras_predict(model, gray)
    print(f"Steering Angle: {steering_angle}")
    
    # Display the video frame
    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))

    # Smooth the steering angle for visualization
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
    
    # Rotate the steering wheel image
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
