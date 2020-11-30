import numpy as numpy
import cv2

i = 0
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

name = input("Enter name:")

while(True):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5)
	for (x, y, w, h) in faces:
		print(x, y, w, h)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)

		if cv2.waitKey(20) & 0xFF == ord('p'):
			if i < 5:
				img_item = f"{name+i+1}.png"
				cv2.imwrite(img_item, roi_color)
				i+=1
				break
			else:
			    break

		# for i in range(0, 5):
		# img_item = f"{name}.png"
		# cv2.imwrite(img_item, roi_color)
		
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

		color = (255, 0, 0) #BGR
		stroke = 2
		img = cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
