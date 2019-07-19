import keras
import cv2
import numpy as np
import matplotlib
print(cv2.__version__)


'''Sketch generating function'''
def sketch(image):
  # Convert image to gray scale
  img_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  # Clean up image using Gaussian Blur
  img_gray_blur = cv2.GaussianBlur(img_gray,(5,5),0)

  #Extract edges
  canny_edges = cv2.Canny(img_gray_blur, 20 , 50)

  # Do an invert binarize the image
  ret , mask = cv2.threshold(canny_edges , 70 , 255 , cv2.THRESH_BINARY_INV)
  return mask

''' 
Initialize webcam . cap is the object provided by VideoCapture
Contains a boolean (ret) indicating if it was successful or not
(frame) contains the image collected from web cam
'''
cap = cv2.VideoCapture(0)

while True:
  ret , frame = cap.read()
  cv2.imshow('Our Live Sketcher' , sketch(frame))
  if cv2.waitKey(1) == 13: # Press Enter  to exit the web cam
    break

cap.release()
cv2.destroyAllWindows()




