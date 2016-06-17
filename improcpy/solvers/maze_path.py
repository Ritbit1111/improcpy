import cv2
import numpy as np

maze = cv2.imread('p/renw.jpg')   						#Image reading
#print maze.shape
#cv2.imshow('original', maze)
#cv2.waitKey(0)

gray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)			#Gray conversion
print gray.shape
cv2.imshow('gray', gray)
cv2.waitKey(0)

'''gray1 = cv2.bilateralFilter(gray, 11, 150, 50)
cv2.imshow('blurr', gray1)
cv2.waitKey(0)
'''

gblur = cv2.GaussianBlur(gray, (15,15), 0)				#Gaussian blur
cv2.imshow('Gblur', gblur)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(gblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
						 cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('THRESH', thresh)							
														#Adaptive thresholding
cv2.waitKey(0)

#ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
#cv2.imshow('threshotsu', thresh1)
#cv2.waitKey(0)
#mask_inv = cv2.bitwise_not(thresh)
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=8)
cv2.imshow('fill', closing)							#closing
cv2.waitKey(0)

#edge = cv2.Canny(mask_inv, 10, 50)
#cv2.imshow('canny', edge)

(cnts, _) = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts,key = cv2.arcLength, reverse = True)[:10000]
ind = []
count = 0

for i, c in enumerate(cnts):
	p = cv2.arcLength(c,True)
	if p>=1500.000000:
		ind.append(i)
		print p
		count += 1
print ind

blank = np.zeros_like(maze)
for i in ind:
	cv2.drawContours(blank, cnts, i, (255,255,255), -1)
cv2.imshow('fill_contour', blank)
cv2.waitKey(0)



kernel = np.ones((3, 3), np.uint8)
masked = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel, iterations=25)
cv2.imshow('masker', masked)
cv2.waitKey(0)

masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

final = cv2.bitwise_and(gray,gray, mask = masked)
cv2.imshow('cropped', final)
cv2.waitKey(0)

re, thresh2 = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
cv2.imshow('threshed', thresh2)
cv2.waitKey(0)

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(thresh2, kernel, iterations = 1)
cv2.imshow('eroded', erosion)
cv2.waitKey(0)

dilation = cv2.dilate((erosion), kernel, iterations = 1)
cv2.imshow('dilated', dilation)
cv2.waitKey(0)

closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations = 1)
cv2.imshow('closing', closing)
cv2.waitKey(0)

cv2.destroyAllWindows()
