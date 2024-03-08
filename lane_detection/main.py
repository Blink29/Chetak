import cv2 as cv 
import numpy as np

video_capture = cv.VideoCapture('test_vid/vid2.mp4')

if not video_capture.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Couldn't read a frame.")
        break
    frame = cv.resize(frame, (600, 400))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 100)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=5, maxLineGap=300)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 > 350 or y2 > 350):
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
