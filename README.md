# UAS-Computer-Vision
# Image Stitching / Panorama using SIFT (OpenCV)

Project ini merupakan implementasi teknik **Image Stitching (Panorama)** menggunakan algoritma **Scale-Invariant Feature Transform (SIFT)** di Python dengan OpenCV.

Program ini mendeteksi fitur pada dua gambar, mencocokkannya, menghitung homografi menggunakan RANSAC, lalu menggabungkannya menjadi satu gambar panorama.

---

## Features

* Deteksi fitur menggunakan SIFT
* Feature Matching menggunakan Brute-Force Matcher (BFMatcher)
* Filtering match dengan Lowe’s Ratio Test
* Estimasi transformasi menggunakan Homography + RANSAC
* Perspective warping dan perluasan kanvas
* Visualisasi keypoints, matches, dan hasil panorama

---

## Algorithm Pipeline

1. Load dua gambar input
2. Konversi ke grayscale
3. Deteksi keypoints & descriptors (SIFT)
4. Pencocokan fitur (BFMatcher)
5. Terapkan Lowe’s Ratio Test
6. Hitung Homography menggunakan RANSAC
7. Warp gambar dan buat panorama

---

## Output

```
Keypoints img1: 15900
Keypoints img2: 19725
Good matches: 3534
```

Hasil ini menunjukkan:

* Kedua gambar memiliki tekstur yang kaya
* Area overlap cukup luas
* Homografi dapat dihitung secara stabil
* Panorama berhasil dibuat dengan baik

---


## Project Structure

```
project-folder/
│
├── panoramic.py
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── README.md
```


## Output Visualization

Program akan menampilkan:

* Original Image 1
* Original Image 2
* SIFT Feature Matches
* Final Panorama (Expanded Canvas)

---

## Concepts Used

* Scale-Invariant Feature Transform (SIFT)
* Feature Descriptor
* Lowe’s Ratio Test
* Homography Matrix
* RANSAC
* Perspective Transformation
* Image Warping

