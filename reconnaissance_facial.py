import cv2
import face_recognition

# Initialize the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Capture your face
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture Your Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("your_face.jpg", frame)

# Capture another face
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture Another Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("other_face.jpg", frame)

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Load the face images
your_face_image = face_recognition.load_image_file("your_face.jpg")
other_face_image = face_recognition.load_image_file("other_face.jpg")

# Get face encodings
your_face_encodings = face_recognition.face_encodings(your_face_image)
other_face_encodings = face_recognition.face_encodings(other_face_image)

# Check if any face encodings were found
if len(your_face_encodings) == 0 or len(other_face_encodings) == 0:
    print("No face found in one or both of the images.")
else:
    # Compare the face encodings
    match_results = face_recognition.compare_faces([your_face_encodings[0]], other_face_encodings[0])
    if match_results[0]:
        print("The faces match!")
    else:
        print("The faces do not match.")
