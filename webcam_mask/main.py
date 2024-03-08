import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), -1)  
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)  
    
    result = cv.bitwise_and(frame, frame, mask=mask_inv)

    cv.imshow('Result', result)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()