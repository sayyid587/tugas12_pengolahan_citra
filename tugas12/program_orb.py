import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MEMBACA GAMBAR
# ==========================================

img1 = cv2.imread('f1.jpg')
img2 = cv2.imread('f2.jpg')

# Mengecek gambar
if img1 is None or img2 is None:
    print("Gambar tidak ditemukan!")
    exit()

# ==========================================
# TAMPILKAN GAMBAR ASLI
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Gambar Asli 1")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Gambar Asli 2")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()

# ==========================================
# TAHAP 1 - EDGE DETECTION
# ==========================================

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 100, 200)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Edge Detection Gambar 1")
plt.imshow(edges1, cmap='gray')

plt.subplot(1,2,2)
plt.title("Edge Detection Gambar 2")
plt.imshow(edges2, cmap='gray')

plt.show()

# ==========================================
# TAHAP 2 - GRAYSCALE
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Grayscale Gambar 1")
plt.imshow(gray1, cmap='gray')

plt.subplot(1,2,2)
plt.title("Grayscale Gambar 2")
plt.imshow(gray2, cmap='gray')

plt.show()

# ==========================================
# TAHAP 3 - KEYPOINT DAN DESCRIPTOR
# ==========================================

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Menggambar keypoint
keypoint_img1 = cv2.drawKeypoints(
    img1,
    kp1,
    None,
    color=(0,255,0),
    flags=0
)

keypoint_img2 = cv2.drawKeypoints(
    img2,
    kp2,
    None,
    color=(0,255,0),
    flags=0
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Keypoint Gambar 1")
plt.imshow(cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Keypoint Gambar 2")
plt.imshow(cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB))

plt.show()

# ==========================================
# TAHAP 4 - FEATURE MATCHING ORB
# ==========================================

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# Urutkan berdasarkan kecocokan terbaik
matches = sorted(matches, key=lambda x: x.distance)

# Gambar hasil matching
matching_result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:20],
    None,
    flags=2
)

plt.figure(figsize=(15,8))
plt.title("ORB Feature Matching")
plt.imshow(cv2.cvtColor(matching_result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()