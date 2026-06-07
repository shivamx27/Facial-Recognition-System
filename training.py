import cv2
import numpy as np
from PIL import Image
import os

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "datasets"

def get_image_paths(path):
    # Get a list of image paths in the specified directory
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return image_paths

def load_faces_and_ids(image_paths):
    faces = []
    ids = []
    
    for image_path in image_paths:
        # Open the image and convert it to grayscale
        face_image = Image.open(image_path).convert('L')
        face_np = np.array(face_image, 'uint8')  # Corrected 'unit8' to 'uint8'
        
        # Extract ID from the filename
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Assuming the ID is the second part of the filename
        
        # Append the face image and ID to the lists
        faces.append(face_np)
        ids.append(id)
        
        # Display the training image (optional)
        cv2.imshow("Training", face_np)
        cv2.waitKey(1)  # Wait for 1 ms to allow the image to be displayed
    
    return faces, ids

# Call the function to get image paths
image_paths = get_image_paths(path)
print("Image Paths:", image_paths)

# Load faces and IDs
faces, ids = load_faces_and_ids(image_paths)
print("Faces Loaded:", len(faces))
print("IDs Loaded:", ids)

# Train the recognizer
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write("Trainer.yml")

# Close all OpenCV windows
cv2.destroyAllWindows() 
print("Training completed...................")