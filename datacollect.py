import cv2

# Open the default camera
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Get user ID
id = input("Enter your ID: ")
id = int(id)  
count = 0  

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
        count += 1
        cv2.imwrite('datasets/user.' + str(id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w]) 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with rectangles around detected faces
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed or if 500 images have been captured
    k = cv2.waitKey(1)
    if k == ord('q') or count >= 500:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
print("Dataset collection done...............")