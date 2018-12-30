import cv2
import numpy as np
from matplotlib import pyplot as plt

#C:\Python27\python.exe lab9.py

#Image imported as a greyscale image
imgGray = cv2.imread('shrek.jpg',cv2.IMREAD_GRAYSCALE)
#imgGray = cv2.imread('danny.jpg',cv2.IMREAD_GRAYSCALE)
#Imported with colour
imgOrig = cv2.imread('shrek.jpg')
#imgOrig = cv2.imread('danny.jpg')
#This blurs the grey image
imgBlur = cv2.GaussianBlur(imgGray,(15, 15),0)

#Plots the positions for the 9 pictures
nrows=3
ncols=3

#Plots clear image and adds a title
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

#Plots the grey image and adds a title
plt.subplot(nrows, ncols,2),plt.imshow(imgGray,cmap='gray')
plt.title('Grey'), plt.xticks([]), plt.yticks([])

#Plots the blurred image and adds a title
plt.subplot(nrows, ncols,3),plt.imshow(imgBlur,cmap='gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])

#Sobel x direction
sobelHorizontal = cv2.Sobel(imgBlur,cv2.CV_64F,1,0,ksize=5) 
#Sobel y direction
sobelVertical = cv2.Sobel(imgBlur,cv2.CV_64F,0,1,ksize=5) 
#Joins both sobel directions
sobel=sobelHorizontal+sobelVertical

#Plots sobel x direction
plt.subplot(nrows, ncols,4),plt.imshow(sobelHorizontal,cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#Plots sobel y direction
plt.subplot(nrows, ncols,5),plt.imshow(sobelVertical,cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#Plots both sobels from above
plt.subplot(nrows, ncols,6),plt.imshow(sobel,cmap='gray')
plt.title('Sobel Combined'), plt.xticks([]), plt.yticks([])
#Change grey image to canny and give it params
canny = cv2.Canny(imgGray,100,200)
#Plot the canny position
plt.subplot(nrows, ncols,7),plt.imshow(canny,cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
#Change grey image to canny and give it params
canny2 = cv2.Canny(imgGray,50,80)
#Plot the canny2 position
plt.subplot(nrows, ncols,8),plt.imshow(canny2,cmap='gray')
plt.title('Grey Canny'), plt.xticks([]), plt.yticks([])
#canny threshold
height,width = sobel.shape

#Used for loop
threshold=120
#Will loop through the pixels and if the pixels are above the threshold defined above make it white, otherwise black
for x in range(0,height):
	for y in range(0,width):
		if(sobel[x][y]>threshold):
			sobel[x][y]=255
		else:
			sobel[x][y]=0
#Plots final sobel image | Threshold
plt.subplot(nrows, ncols,9),plt.imshow(sobel,cmap='gray')	
plt.title('Sobel Sum Image'), plt.xticks([]), plt.yticks([])
#Plots all images
plt.show()
