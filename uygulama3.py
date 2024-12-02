import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from collections import Counter

# Kaydedilen modeli, scaler'ı ve label encoder'ı yükleme
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Özellik çıkarımı fonksiyonu
def extract_features(data, sr):
    features = {}

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
    features.update(extract_statistical_features(spectral_centroid, 'spectral_centroid'))

    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=data, sr=sr)
    features.update(extract_statistical_features(np.expand_dims(spectral_flux, axis=0), 'spectral_flux'))

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    features.update(extract_statistical_features(spectral_bandwidth, 'spectral_bandwidth'))

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=data)
    features.update(extract_statistical_features(spectral_flatness, 'spectral_flatness'))

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)
    features.update(extract_statistical_features(spectral_rolloff, 'spectral_rolloff'))

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sr)
    features.update(extract_statistical_features(spectral_contrast, 'spectral_contrast'))

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(data)
    features.update(extract_statistical_features(zero_crossing_rate, 'zero_crossing_rate'))

    # Root mean squared error
    rmse = librosa.feature.rms(y=data)
    features.update(extract_statistical_features(rmse, 'rms'))

    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr)
    features.update(extract_statistical_features(chroma_stft, 'chroma_stft'))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=data, sr=sr)
    features['tempo'] = tempo

    # Mel-Frequency Cepstral Coefficients (MFCC)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        features.update(extract_statistical_features(mfcc[i, :], f'mfcc_{i+1}'))

    return pd.DataFrame([features])

# İstatistiksel özellikleri çıkaran yardımcı fonksiyon
def extract_statistical_features(feature_array, feature_name):
    return {
        f'{feature_name}_mean': np.mean(feature_array),
        f'{feature_name}_std': np.std(feature_array),
        f'{feature_name}_min': np.min(feature_array),
        f'{feature_name}_max': np.max(feature_array),
        f'{feature_name}_kurtosis': kurtosis(feature_array, axis=None),
        f'{feature_name}_skew': skew(feature_array, axis=None)
    }

# Segmentleri analiz eden fonksiyon
def analyze_segments(data, sr, segment_duration=5):
    segment_samples = segment_duration * sr
    num_segments = len(data) // segment_samples
    predictions = []

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = data[start:end]
        features_df = extract_features(segment, sr)
        features_scaled = scaler.transform(features_df)
        prediction_encoded = model.predict(features_scaled)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        predictions.append(prediction[0])

    # Duygu durumlarının yüzdesini hesaplama
    total_predictions = len(predictions)
    emotion_counts = Counter(predictions)
    emotion_percentages = {emotion: (count / total_predictions) * 100 for emotion, count in emotion_counts.items()}

    return emotion_percentages, predictions

# Yıldız değerlendirme fonksiyonu
def calculate_star_rating(emotion_percentages):
    if 'mutlu' in emotion_percentages and emotion_percentages['mutlu'] >= 80:
        return 5
    elif 'mutlu' in emotion_percentages and emotion_percentages['mutlu'] >= 60:
        return 4
    elif 'mutlu' in emotion_percentages and emotion_percentages['mutlu'] >= 40:
        return 3
    elif 'sinirli' in emotion_percentages and emotion_percentages['sinirli'] >= 60:
        return 1
    else:
        return 2

# Streamlit arayüzü
st.title("Müşteri Hizmetleri Görüşmesi Analizi")
uploaded_file = st.file_uploader("Bir ses dosyası yükleyin (4 dakikalık görüşme önerilir)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.info("Ses dosyası başarıyla yüklendi. İşleniyor...")
    
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Ses dosyasını yükleyip segmentleri analiz etme
    y, sr = librosa.load("temp_audio.wav", sr=None)
    emotion_percentages, predictions = analyze_segments(y, sr, segment_duration=5)

    # Analiz sonuçlarını gösterme
    st.subheader("Duygu Durumu Yüzdeleri")
    st.write(emotion_percentages)

    # Duygu durumlarının grafiksel gösterimi
    st.bar_chart(pd.Series(emotion_percentages))

    # Yıldız değerlendirmesi
    stars = calculate_star_rating(emotion_percentages)
    st.subheader("Görüşme Yıldız Puanı:")
    st.write("⭐" * stars)

    st.subheader("Tüm Segmentlerin Analizi")
    st.write("Her segment için tahmin edilen duygular:")
    st.write(predictions)

    # Ses dosyasını çalma
    st.audio(uploaded_file, format="audio/wav")

st.markdown("<footer style='text-align: center; font-size: 12px; margin-top: 20px;'>Bu uygulama, müşteri hizmetleri görüşmelerinin duygu durumu analizini yapar.</footer>", unsafe_allow_html=True)
