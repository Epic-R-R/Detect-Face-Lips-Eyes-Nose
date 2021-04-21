import cv2


CAMERA_DEVICE_ID = 0

# Load the cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")
noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

# Open a handler for the camera
video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)

# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while video_capture.isOpened():
    # Grab a single frame of video (and check if it went ok)
    ok, frame = video_capture.read()
    if not ok:
        logging.error("Could not read frame from camera. Stopping video capture.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    # Detect the Simle
    smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the Lips
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)



    # Detect the Eyes
    eyes = eyeCascade.detectMultiScale(roi_gray)
    # Draw a rectangle around the Eyes
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  
    # Detect the Nose
    nose = noseCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.3,
        minNeighbors=5   
    )
    # Draw a rectangle around the Nose
    for (x, y, w, h) in nose:
        cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()