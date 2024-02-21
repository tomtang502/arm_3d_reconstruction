import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((14*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:14].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,14), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (8,14), corners2, ret)
    # cv.imwrite(fname[-12:], img)
    cv.imshow('img', img)
    cv.waitKey(500)
# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv.imread('left12.jpg')
h, w = img.shape[:2]
print(mtx)
print(dist)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(mtx)
print(newcameramtx)
for fname in images:
    img = cv.imread(fname)
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult.png', dst)
    cv.imwrite(fname[-12:], dst)
