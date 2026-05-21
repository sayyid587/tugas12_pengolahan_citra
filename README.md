# tugas12_pengolahan_citra



# ORB Feature Matching dengan Python OpenCV

## Deskripsi Project
Project ini merupakan implementasi pengolahan citra digital menggunakan algoritma ORB (Oriented FAST and Rotated BRIEF) pada Python OpenCV. Program digunakan untuk mendeteksi fitur, menampilkan keypoint, dan melakukan pencocokan fitur antara dua gambar.

ORB merupakan alternatif dari algoritma SIFT dan SURF karena memiliki performa yang cepat, ringan, dan bebas lisensi paten.

---

# Library yang Digunakan

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

# Tahapan Program
## 1. Menampilkan Gambar Asli

Pada tahap ini program membaca dua gambar input menggunakan OpenCV sebelum dilakukan proses pengolahan citra.
```python
img1 = cv2.imread('mobil1.jpg')
img2 = cv2.imread('mobil2.jpg')
```
Tujuan:
- Membaca gambar input
- Menampilkan citra asli sebelum diproses

## 2. Edge Detection

Tahap ini menggunakan metode Canny Edge Detection untuk mendeteksi tepi objek pada gambar.
```python
edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 100, 200)
```
Tujuan:
- Menampilkan batas objek
- Mempermudah identifikasi bentuk pada gambar

## 3. Grayscale

Pada tahap grayscale, gambar berwarna diubah menjadi abu-abu agar proses komputasi lebih ringan.
```python
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
```
Tujuan:
- Mengurangi kompleksitas warna
- Mempermudah proses ekstraksi fitur

## 4. Keypoint dan Descriptor

Tahap ini menggunakan algoritma ORB untuk mendeteksi titik penting (keypoint) dan menghasilkan descriptor.
```python
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
```
Tujuan:
- Menemukan titik fitur unik pada gambar
- Membuat descriptor untuk proses pencocokan

## 5. Feature Matching ORB

Pada tahap terakhir dilakukan pencocokan fitur antara dua gambar menggunakan BFMatcher.
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
```
Tujuan:
- Mencocokkan fitur antar gambar
- Menampilkan hubungan keypoint yang mirip

# Code Lengkap
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MEMBACA GAMBAR
# ==========================================

img1 = cv2.imread('mobil1.jpg')
img2 = cv2.imread('mobil2.jpg')

# ==========================================
# MENAMPILKAN GAMBAR ASLI
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
# GRAYSCALE
# ==========================================

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Grayscale 1")
plt.imshow(gray1, cmap='gray')

plt.subplot(1,2,2)
plt.title("Grayscale 2")
plt.imshow(gray2, cmap='gray')

plt.show()

# ==========================================
# EDGE DETECTION
# ==========================================

edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 100, 200)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Edge Detection 1")
plt.imshow(edges1, cmap='gray')

plt.subplot(1,2,2)
plt.title("Edge Detection 2")
plt.imshow(edges2, cmap='gray')

plt.show()

# ==========================================
# ORB KEYPOINT DAN DESCRIPTOR
# ==========================================

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

img_kp1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0))
img_kp2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Keypoint 1")
plt.imshow(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Keypoint 2")
plt.imshow(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB))

plt.show()

# ==========================================
# FEATURE MATCHING
# ==========================================

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

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
```

# Kesimpulan

Berdasarkan hasil percobaan, algoritma ORB mampu mendeteksi dan mencocokkan fitur antar gambar dengan cukup baik. ORB bekerja dengan mendeteksi keypoint dan descriptor kemudian mencocokkannya menggunakan Brute Force Matcher. Metode ini efektif digunakan pada pengolahan citra digital karena cepat, ringan, dan dapat menangani rotasi serta perubahan skala gambar.
