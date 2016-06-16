import cv2
import numpy as np
from operator import itemgetter

maze = cv2.imread('p/renw.jpg', 0)

#gray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(maze, 11, 150, 50)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
mask_inv = cv2.bitwise_not(thresh)

edge = cv2.Canny(mask_inv, 10, 50)
cv2.imshow('canny', edge)

(cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts,key = cv2.arcLength, reverse = True)[:10000]
ind = []
count = 0

for i, c in enumerate(cnts):
	p = cv2.arcLength(c,True)
	if p>=1000.000000:
		ind.append(i)
		count += 1


mask = np.full_like(maze,0)

for i in ind:
	cv2.drawContours(mask, cnts, i, 255, -1) # Draw filled contour in mask
#ret1, bin_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

i = cv2.bitwise_and(maze,maze, mask = mask)
#out = np.zeros_like(maze) # Extract out the object and place into output image
#out[mask == 255] = maze[mask == 255]
#print mask.shape
#print type(mask)
#print maze.shape
cv2.imshow('Output', mask)
cv2.waitKey(0)
'''
(cnt, _) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(cnt):
	p = cv2.arcLength(c,True)
	print p
	print c.shape

mask1 = np.full_like(maze,0)
cv2.drawContours(mask1, cnt, -1, (255,255,255), -25)
cv2.imshow('yo', mask1)
cv2.waitKey(0)
'''
cv2.destroyAllWindows()



