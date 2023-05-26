import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_pose(matches, keypoints1, keypoints2, camera_matrix, dist_coeffs):
    points2D = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    points3D = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)

    _, rvec, tvec = cv2.solvePnP(points3D, points2D, camera_matrix, dist_coeffs)

    return rvec, tvec

# Load images
img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')
img3 = cv2.imread('img3.png')

# Load 2D-3D arrays
points2D = np.load('vr2d.npy')
points3D = np.load('vr3d.npy')

# correct shape of the 2D and 3D points array
points2D = points2D.reshape(20, 2)
points3D = points3D.reshape(20, 3)


# Camera calibration
fx = 100
fy = 100
cx = 960
cy = 540
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))

feature_detector = cv2.ORB_create()
keypoints1, descriptors1 = feature_detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = feature_detector.detectAndCompute(img2, None)
keypoints3, descriptors3 = feature_detector.detectAndCompute(img3, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1_2 = matcher.match(descriptors1, descriptors2)
matches1_3 = matcher.match(descriptors1, descriptors3)


rvec_1_2, tvec_1_2 = refine_pose(matches1_2, keypoints1, keypoints2, camera_matrix, dist_coeffs)
rvec_1_3, tvec_1_3 = refine_pose(matches1_3, keypoints1, keypoints3, camera_matrix, dist_coeffs)

# Plotting
trajectory = np.array([tvec_1_2.T, tvec_1_3.T])
plt.plot(trajectory[:, 0], trajectory[:, 2], '-o')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Computer Vision Test')
plt.show()
