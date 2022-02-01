import cv2
import numpy as np
img=cv2.imread(r"C:\Users\91746\Downloads\1299171.jpg")
def cartoonize(img,k):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('edges',edges)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,8)
    data=np.float32(img).reshape((-1,3))
    print(data.shape)
    print(img.shape)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label,center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    #print(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    #cv2.imshow("result",result)
    blurred = cv2.medianBlur(result,3)
    cartoon=cv2.bitwise_and(blurred,blurred,mask=edges)
    return cartoon
cartoonized=cartoonize(img,8)
cv2.imshow('input',img)
cv2.imshow('output',cartoonized)
cv2.waitKey(0)
