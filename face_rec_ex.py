import face_recognition
import os
import cv2

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	for (x, y, w, h) in faces:
		print(x, y, w, h)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		img_item = "known/paddy.png"
		cv2.imwrite(img_item, roi_color)

		color = (255, 0, 0)
		img = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
