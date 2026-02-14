import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('./dataset/image1.jpg')
img2 = cv2.imread('./dataset/image2.jpg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Keypoints img1:", len(kp1))
print("Keypoints img2:", len(kp2))

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img1 = np.float32([
    [0, 0],
    [w1, 0],
    [w1, h1],
    [0, h1]
]).reshape(-1, 1, 2)

corners_img2 = np.float32([
    [0, 0],
    [w2, 0],
    [w2, h2],
    [0, h2]
]).reshape(-1, 1, 2)

warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

panorama_width  = x_max - x_min
panorama_height = y_max - y_min

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])

panorama = cv2.warpPerspective(
    img1,
    translation @ H,
    (panorama_width, panorama_height)
)

panorama[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

matches_vis = cv2.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    good_matches[:100],   # batasi agar tidak terlalu padat
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(18, 12))

plt.subplot(3, 2, 1)
plt.imshow(img1_rgb)
plt.title("Before: Image 1 (jalan1.jpg)")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(img2_rgb)
plt.title("Before: Image 2 (jalan2.jpg)")
plt.axis('off')


plt.subplot(3, 2, (3, 4))
plt.imshow(matches_vis)
plt.title("SIFT Feature Matches (Stitches)")
plt.axis('off')

plt.subplot(3, 2, (5, 6))
plt.imshow(panorama_rgb)
plt.title("After: Panorama (Expanded Canvas)")
plt.axis('off')

plt.tight_layout()
plt.show()
