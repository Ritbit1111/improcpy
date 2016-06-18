import cv2
import numpy as np

maze = cv2.imread('/home/ritesh/Documents/My Program/p/renw.jpg')   		#Image reading

gray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)			#Gray conversion of original image

gblur = cv2.GaussianBlur(gray, (15,15), 0)				#Gaussian blur
cv2.imshow('Gblur', gblur)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(gblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
						 cv2.THRESH_BINARY_INV, 11, 2)							
														#Adaptive thresholding

kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=8)  			 #closing 


(cnts, _) = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Finding contours

ind = []
count = 0

for i, c in enumerate(cnts):
	p = cv2.arcLength(c,True)
	if p>=1500.000000:							#filtering long contours
		ind.append(i)
		print p
		count += 1
print ind

blank = np.zeros_like(maze)						# blank white screen

for i in ind:
	cv2.drawContours(blank, cnts, i, (255,255,255), -1)    	#drawing filtered contours on blank


kernel = np.ones((3, 3), np.uint8)
masked = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel, iterations=25)
cv2.imshow('masker', masked)							#filling the whole region white 
														#will serve as our mask to filter out maze

masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

final = cv2.bitwise_and(gray,gray, mask = masked)		# cropping out from the original image using mask

re, thresh2 = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
														#Thresholding so that black maze walls get merged to black surroundings and white maze path gets 255 white

#this will leave some white noises 
 
kernel = np.ones((5,5), np.uint8)						
erosion = cv2.erode(thresh2, kernel, iterations = 1)		#so erosion and dilation to remove noises
cv2.imshow('eroded', erosion)
cv2.waitKey(0)

dilation = cv2.dilate((erosion), kernel, iterations = 1)
cv2.imshow('dilated', dilation)
cv2.waitKey(0)

cv2.destroyAllWindows()
