import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Fungsi untuk menghapus latar belakang gambar
def remove_background(img):
    # Ubah gambar ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize gambar
    resized_image = cv2.resize(img_rgb, (1600, 1200))
    
    # Ubah gambar ke grayscale
    gs = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    blur = cv2.GaussianBlur(gs, (55, 55), 0)
    
    # Thresholding dengan Otsu
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological closing
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    
    # Temukan kontur
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Fungsi untuk menemukan kontur daun
    def find_contour(cnts):
        contains = []
        y_ri, x_ri, _ = resized_image.shape
        for cc in cnts:
            yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
            contains.append(yn)
        return np.argmax(contains)

    # Buat gambar hitam
    black_img = np.empty([1200, 1600, 3], dtype=np.uint8)
    black_img.fill(0)
    
    # Temukan indeks kontur daun dan gambar kontur pada gambar hitam
    index = find_contour(contours)
    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)
    
    # Aplikasikan mask pada gambar asli
    maskedImg = cv2.bitwise_and(resized_image, mask)
    
    # Ubah semua piksel hitam menjadi putih
    final_img = maskedImg
    h, w, _ = final_img.shape
    black_pix = [0, 0, 0]
    white_pix = [255, 255, 255]
    for x in range(w):
        for y in range(h):
            channels_xy = final_img[y, x]
            if all(channels_xy == black_pix):
                final_img[y, x] = white_pix
    
    return final_img

# Fungsi untuk mengekstrak fitur dari gambar
def extract_features_from_image(img):
    img = remove_background(img)  # Tambahkan penghapusan latar belakang di sini
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Fitur warna
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # Fitur tekstur
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = [red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
              contrast, correlation, inverse_diff_moments, entropy]

    return vector

# Fungsi untuk memprediksi gambar baru
def predict_new_image(img, scaler, model):
    # Ekstraksi fitur dari gambar
    features = extract_features_from_image(img)

    # Normalisasi fitur menggunakan scaler yang sama dengan data latih
    features_scaled = scaler.transform([features])

    # Prediksi menggunakan model SVM
    prediction = model.predict(features_scaled)

    # Logging fitur dan prediksi untuk debugging
    #st.write("Fitur yang diekstrak dari gambar:", features)
    #st.write("Fitur yang di-skala:", features_scaled)
    st.write("Hasil prediksi model:", prediction)

    # Interpretasi hasil prediksi dan mengembalikan label kelas
    if prediction == 1:
        return "Kelas 1 (blas)"
    elif prediction == 2:
        return "Kelas 2 (hawar)"
    else:
        return "Kelas 3 (sehat)"

st.title("Klasifikasi Penyakit Tanaman Padi")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        st.image(img, channels="BGR", caption="Gambar yang diunggah")

        # Muat scaler dan model
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('svm_model.pkl')
        
        # Prediksi gambar baru
        predicted_label = predict_new_image(img, scaler, model)
        
        st.write("Prediksi untuk gambar ini:", predicted_label)
    else:
        st.write("Gambar tidak dapat dibaca. Silakan coba gambar lain.")
else:
    st.write("Silakan unggah gambar untuk klasifikasi.")
