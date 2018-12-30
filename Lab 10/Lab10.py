import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'GMIT1.jpg'
#Image imported with color
img = cv2.imread('GMIT2.JPG')
#Image imported with greyscale
gray = cv2.imread('GMIT2.JPG',cv2.IMREAD_GRAYSCALE)

#Plots the positions for the 9 pictures
nrows=3
ncols=3

#Plots normal image 
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('Normal'), plt.xticks([]), plt.yticks([])
#plot the clear greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray,cmap='gray')
plt.title('Grey'), plt.xticks([]), plt.yticks([])
#Corner Harris image
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#Plot the harris image & Title
plt.subplot(nrows, ncols,3),plt.imshow(dst,cmap='gray')
plt.title('HarrisCornerDetection'), plt.xticks([]), plt.yticks([])
#Deep Copy image
imgHarris = img.copy()
#Display the above copy
plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(imgHarris,cv2.COLOR_BGR2RGB))
plt.title('DeepCopy'), plt.xticks([]), plt.yticks([])

#Loop through every element in the 2d matrix - DST
threshold = 0.001;
for i in range(len(dst)):
 for j in range(len(dst[i])):
	if dst[i][j] > (threshold*dst.max()):
		cv2.circle(imgHarris,(j,i),3,(200, 100, 20),-1)


#Plots the corner harris image
plt.subplot(nrows, ncols,5),plt.imshow(cv2.cvtColor(imgHarris,cv2.COLOR_BGR2RGB))
plt.title('Corners'), plt.xticks([]), plt.yticks([])
#Giving corners values
corners = cv2.goodFeaturesToTrack(gray,95,0.01,10)
#Deep copy image
imgShiTomasi = img.copy()

#Iterate over every pixel + draw a circle at each point higher than threshold
for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi,(x,y),3,(175, 85, 15),-1)

#Plot Shi image
plt.subplot(nrows, ncols,6),plt.imshow(cv2.cvtColor(imgShiTomasi,cv2.COLOR_BGR2RGB))
plt.title('Shi Image'), plt.xticks([]), plt.yticks([])

#Deep Copy #3
imgCopy = img.copy()

#Initiate SIFT detector with a feature limit of 50
sift = cv2.SIFT(50)
kp = sift.detect(gray,None)
#Draw keypoints
imgSift = cv2.drawKeypoints(imgCopy,kp,color=(40, 200, 35), flags = 4)
#Plot sift image
plt.subplot(nrows, ncols,7),plt.imshow(cv2.cvtColor(imgSift,cv2.COLOR_BGR2RGB))
plt.title('SiftImnage'), plt.xticks([]), plt.yticks([])

#Displays all images
plt.show()