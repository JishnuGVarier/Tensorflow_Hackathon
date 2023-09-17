import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

# Load your emotion detection model
model = keras.models.load_model('C:\\Users\\heyrg\\Desktop\\hack\\model.h5')


# Create a function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera (you may need to change this)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Frame not captured.")
        break

    # Preprocess the captured frame
    preprocessed_frame = preprocess_image(frame)

    # Perform emotion detection on the preprocessed frame
    emotion_class = model.predict(preprocessed_frame)
    emotion_label = np.argmax(emotion_class)

    # Map the emotion label to a human-readable string (e.g., "happy")
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']   # Replace with your emotion labels
    emotion = emotions[emotion_label]

    # Display the emotion label on the frame
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the emotion label
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
