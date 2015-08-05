#include<stdlib.h>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

import cv2
import sys

cascPath = sys.argv[1]
faceCascade1 = cv2.CascadeClassifier(cascPath)
faceCascade2 = cv2.CascadeClassifier(cascPath)

video_capture1 = cv2.VideoCapture(0)
video_capture2 = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame1 = video_capture1.read()
    ret, frame2 = video_capture2.read()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    faces1 = faceCascade1.detectMultiScale(
        gray1,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE

        # cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    faces2 = faceCascade2.detectMultiScale(
        gray2,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE

        # cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces1&faces2:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video1', frame1)
    cv2.imshow('Video2', frame2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture1.release()
video_capture2.release()
cv2.destroyAllWindows()
