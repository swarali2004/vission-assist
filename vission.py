import cv2
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
from transformers import pipeline

# Constants for object detection
Known_distance = 30  # Inches
Known_width = 5.7  # Inches

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Load the YOLO model
model = YOLO('yolov8n.pt')  # load an official model
names = model.names  # get class names

# Create an image captioning pipeline using the specified model
image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")


# Function to calculate focal length
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width


# Function to estimate distance
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    return (real_object_width * Focal_Length) / object_width_in_frame


# Function to generate a description of the image using image captioning pipeline
def generate_description(image_path):
    # Generate a caption for the image using the image captioning pipeline
    caption = image_captioning_pipeline(image_path)
    # Extract the caption text from the result
    caption_text = caption[0]['caption']
    return caption_text


# Function for speech generation
def generate_speech(description):
    # Speak out the description
    engine.say(description)
    engine.runAndWait()


# Function to control speech generation based on user commands
def control_speech():
    while True:
        print("Say 'start' to begin or 'stop' to end.")
        command = listen_for_command()
        if command == "start":
            cap = cv2.VideoCapture(0)  # Camera object
            describe_objects(cap)
        elif command == "stop":
            print("Stopping speech generation...")
            engine.stop()
            break
        else:
            print("Sorry, could not understand the command.")


# Function to listen for start/stop command
def listen_for_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print("Received command:", command)
        return command
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your internet connection.")
        return ""


# Function to describe objects detected by YOLO and generate speech
def describe_objects(cap):
    Focal_length_found = None  # Initialize Focal_length_found variable
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Predict objects in the frame using YOLO
        results = model(frame)  # predict on an image
        result = results[0]

        # Calculate focal length
        if Focal_length_found is None:
            Focal_length_found = FocalLength(Known_distance, Known_width, frame.shape[1])

        # Process detected objects
        for box in result.boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            x, y, w, h = cords
            class_id = result.names[box.cls[0].item()]

            # Calculate distance for the detected object
            object_width_in_frame = w
            object_distance = Distance_finder(Focal_length_found, Known_width, object_width_in_frame)
            object_distance = round(object_distance, 2)

            # Generate description for the image using image captioning pipeline
            description = generate_description(frame)  # Use the current frame for image captioning

            # Speak out the description
            generate_speech(description)

            # Draw rectangles around the detected objects
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Object: {class_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy the windows
    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    control_speech()


if __name__ == "__main__":
    main()
