import cv2

# Open the default camera
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
recognizer.read("Trainer.yml")

# List of names corresponding to the IDs
name_list = ["","Dipanshu Singh","RIHAN RAJ","RITIK SINGH","PIYUSH KUMAR"]  # Ensure this matches the IDs used during training

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Predict the ID of the detected face
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Check if the confidence is above the threshold
        if conf < 50:  # Lower confidence means a better match
            if serial < len(name_list):  # Ensure the index is valid
                cv2.putText(frame, name_list[serial], (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            else:
                cv2.putText(frame, "UNKNOWN", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with rectangles around detected faces
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
print("Face recognition completed.")
