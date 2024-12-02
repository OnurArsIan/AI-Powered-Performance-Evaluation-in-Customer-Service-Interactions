import os
import pandas as pd
import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# Fonksiyon: Belirtilen dizindeki dosya yollarını ve etiketleri çıkarır
def create_dataframe_from_directory(directory_path, limit=2800):
    paths = []
    labels = []
    
    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            paths.append(file_path)
            label = dirname.split('\\')[-1]
            labels.append(label.lower())
            
            if len(paths) == limit:
                break
        if len(paths) == limit:
            break
    
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    return df

# Train verisi için DataFrame oluşturma
train_df = create_dataframe_from_directory('train_sound')

# Test verisi için DataFrame oluşturma
test_df = create_dataframe_from_directory('test_sound')

# İlk 5 satırı göstermek için
print("Train DataFrame:")
print(train_df.head())

print("\nTest DataFrame:")
print(test_df.head())

# Verileri CSV dosyasına kaydetme
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("\nVeriler CSV dosyalarına kaydedildi: 'train_dataset.csv' ve 'test_dataset.csv'")


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

# Ses dosyasından özellik çıkarımı yapan ana fonksiyon
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_centroid, 'spectral_centroid'))

    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    features.update(extract_statistical_features(np.expand_dims(spectral_flux, axis=0), 'spectral_flux'))

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_bandwidth, 'spectral_bandwidth'))

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features.update(extract_statistical_features(spectral_flatness, 'spectral_flatness'))

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_rolloff, 'spectral_rolloff'))

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_contrast, 'spectral_contrast'))

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    features.update(extract_statistical_features(zero_crossing_rate, 'zero_crossing_rate'))

    # Root mean squared error
    rmse = librosa.feature.rms(y=y)
    features.update(extract_statistical_features(rmse, 'rms'))

    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update(extract_statistical_features(chroma_stft, 'chroma_stft'))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # Mel-Frequency Cepstral Coefficients (MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        features.update(extract_statistical_features(mfcc[i, :], f'mfcc_{i+1}'))

    return features
# Belirtilen dizindeki tüm dosyalar için özellik çıkarımı ve DataFrame oluşturma
def create_features_dataframe(directory_path):
    data = []
    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            features = extract_features(file_path)
            features['file_path'] = file_path
            features['label'] = dirname.split('\\')[-1].lower()  # Klasör adından etiketi çıkar
            data.append(features)
    return pd.DataFrame(data)

# Train ve test data yolları
train_path = 'train_sound'
test_path = 'test_sound'

# DataFrame'leri oluşturma
train_df = create_features_dataframe(train_path)
test_df = create_features_dataframe(test_path)

# DataFrame'leri CSV dosyalarına kaydetme
train_df.to_csv('train_features.csv', index=False)
test_df.to_csv('test_features.csv', index=False)

print("Özellik çıkarımı tamamlandı ve CSV dosyalarına kaydedildi: 'train_features.csv' ve 'test_features.csv'")



# Özellikler ve etiketleri ayırma
X_train = train_df.drop(['label', 'file_path'], axis=1)
y_train = train_df['label']

X_test = test_df.drop(['label', 'file_path'], axis=1)
y_test = test_df['label']
 
# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Veriyi ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost modeli oluşturma ve eğitme
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train_encoded)

# Test veri seti üzerinde tahmin yapma
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Tahminleri orijinal etiketlere dönüştürme
y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

# Model performansını değerlendirme
print("XGBoost Doğruluk Skoru:", accuracy_score(y_test, y_pred_xgb_decoded))
print("XGBoost Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb_decoded))

import joblib

# Modeli kaydetme
joblib.dump(xgb_model, 'xgb_model.pkl')

# Ölçekleyiciyi kaydetme
joblib.dump(scaler, 'scaler.pkl')

# Label encoder'ı kaydetme
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model ve ilgili objeler başarıyla kaydedildi.")
