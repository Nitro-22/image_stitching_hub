import cv2
import numpy as np
from sift_functions import generate_image_pyramid, calculate_dog, SIFT_feature_detection
from helper_functions import quick_resize

def convert_keypoints_to_cv2(keypoints, scale=1):
    kp_cv2 = []
    for octave_index, _, (r, c) in keypoints:
        x = c * (2 ** octave_index)
        y = r * (2 ** octave_index)
        kp_cv2.append(cv2.KeyPoint(x / scale, y / scale, 1))
    return kp_cv2

def get_descriptors(img, kp_cv2):
    sift = cv2.SIFT_create()
    kp_cv2, descriptors = sift.compute(img, kp_cv2)
    return kp_cv2, descriptors

def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def stitch_images(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result