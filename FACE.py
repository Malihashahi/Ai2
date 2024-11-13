
# import necessary modules
import cv2
import sys

# importing the cascade classifier for face and eye
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# check for input video
# if no input is given, default camera is choosen as input source
if len(sys.argv) == 1:
    cap = 0
else:
    cap = sys.argv[1]

# initialize input head, with source
video = cv2.VideoCapture(cap)

# Run an infinite loop, until user quit(press 'q')
while True:
    
    # reading frame from the video source
    _, frame =video.read()
    
    # cinverting frame to Gray scale to pass on classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces and return coordinates of rectangle
    # This is the section, where you need to work
    # To get more accurate result, you need to play with this parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    # make a rectangle around face detected
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # extract the rectangle, containing face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # As eye without a face doen't make any sense
        # so we search for eye, within the face only
        # this reduces the computational load an also increases accuracy
        # detect eyes and return coordinates of rectangle
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # make a rectangle around face detected
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    # show the processed frame
    cv2.imshow('Output',frame)
    
    # If 'q' pressed => Quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Release the video object
        video.release()
        # close all open windows
        cv2.destroyAllWindows() 
        exit(0)