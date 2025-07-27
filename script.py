#!/usr/bin/env python3
"""
Product Access System Demonstration

This script demonstrates the exact flow:
1. User attempts to access the product prediction model
2. If face is recognized, they proceed to attempt to run a prediction
3. The prediction must be confirmed through voice verification
4. The system determines if the prediction is allowed
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from sklearn.preprocessing import StandardScaler

# Set paths
IMAGE_DIR = 'Images'
AUDIO_DIR = 'Audio'
MODEL_DIR = 'Models'

# Load models
try:
    with open(os.path.join(MODEL_DIR, 'face_recognition_model.pkl'), 'rb') as f:
        face_model = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'voiceprint_verification_model.pkl'), 'rb') as f:
        voice_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'product_recommendation_model.pkl'), 'rb') as f:
        product_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'product_label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please run the model_creation.ipynb notebook first to train and save the models.")
    sys.exit(1)

# Load customer data
try:
    customer_data = pd.read_csv('Data Files/merged_customer_data.csv')
    print("Customer data loaded successfully!")
except FileNotFoundError:
    print("Customer data file not found. Please run the data preprocessing notebook first.")
    sys.exit(1)

# Function to extract image features
def extract_image_features(image_path):
    """Extract features from an image for facial recognition."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract histogram features
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_rgb], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    # Extract HOG features (simplified)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128))
    
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    hog_features = np.concatenate([mag.flatten()[:20], ang.flatten()[:20]])
    
    features = np.concatenate([hist_features, hog_features])
    
    # Match expected feature length
    expected_length = 40
    if len(features) > expected_length:
        features = features[:expected_length]
    elif len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)), 'constant')
    
    return features

# Function to extract audio features
def extract_audio_features(audio_path):
    """Extract features from an audio file for voice verification."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        features = np.concatenate([
            mfccs_mean, mfccs_std, 
            [rolloff_mean, centroid_mean, zcr_mean, rms_mean]
        ])
        
        return features
    
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def product_access_system(face_image_path, voice_sample_path):
    """
    Implements the exact flow:
    1. User attempts to access the product prediction model
    2. If face is recognized, they proceed to attempt to run a prediction
    3. The prediction must be confirmed through voice verification
    4. The system determines if the prediction is allowed
    """
    
    print("\\n" + "="*60)
    print("🔐 PRODUCT PREDICTION MODEL ACCESS ATTEMPT")
    print("="*60)
    
    # Step 1: User attempts to access the product prediction model
    print("\\n👤 User attempting to access product prediction model...")
    print("Processing facial recognition for access control...")
    
    # Display the face image
    img = cv2.imread(face_image_path)
    if img is None:
        print("❌ ACCESS DENIED: Could not process face image")
        return
    
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Face Recognition - Access Control")
    plt.axis('off')
    plt.show()
    
    # Extract face features
    face_features = extract_image_features(face_image_path)
    if face_features is None:
        print("❌ ACCESS DENIED: Face recognition failed")
        return
    
    # Face recognition
    face_prediction = face_model.predict([face_features])[0]
    face_proba = face_model.predict_proba([face_features])[0]
    face_confidence = np.max(face_proba)
    
    print(f"Face Recognition Result: {face_prediction}")
    print(f"Recognition Confidence: {face_confidence:.4f}")
    
    if face_confidence < 0.6:
        print("❌ ACCESS DENIED: Face not recognized with sufficient confidence")
        print("   → User is not authorized to access the product prediction model")
        return
    
    print("✅ FACE RECOGNIZED: Access granted to product prediction model")
    
    # Step 2: User proceeds to attempt to run a prediction
    print("\\n" + "-"*60)
    print("📊 PROCEEDING TO PRODUCT PREDICTION")
    print("-"*60)
    
    # Map member ID to customer ID (updated to show different product categories)
    member_to_customer = {
        'member1': 'A150',  # Books
        'member2': 'A190',  # Sports  
        'member3': 'A137'   # Electronics
    }
    
    if face_prediction not in member_to_customer:
        print("❌ PREDICTION FAILED: No customer data found")
        return
    
    customer_id = member_to_customer[face_prediction]
    customer_row = customer_data[customer_data['customer_id_new'] == customer_id]
    
    if customer_row.empty:
        print("❌ PREDICTION FAILED: Customer data not available")
        return
    
    # Get customer features for prediction
    customer_features = customer_row[['engagement_score', 'purchase_interest_score', 'customer_rating']].values[0]
    
    print(f"Customer Profile: {customer_id}")
    print(f"  • Engagement Score: {customer_features[0]}")
    print(f"  • Purchase Interest: {customer_features[1]}")
    print(f"  • Customer Rating: {customer_features[2]}")
    
    # Generate prediction
    product_prediction_encoded = product_model.predict([customer_features])[0]
    product_prediction = label_encoder.inverse_transform([product_prediction_encoded])[0]
    product_proba = product_model.predict_proba([customer_features])[0]
    product_confidence = np.max(product_proba)
    
    print(f"\\n🎯 PREDICTION GENERATED:")
    print(f"   Recommended Product Category: {product_prediction}")
    print(f"   Prediction Confidence: {product_confidence:.4f}")
    
    # Step 3: The prediction must be confirmed through voice verification
    print("\\n" + "-"*60)
    print("🎤 VOICE CONFIRMATION REQUIRED")
    print("-"*60)
    print("To execute this prediction, voice confirmation is required...")
    
    # Play the audio sample
    try:
        y, sr = librosa.load(voice_sample_path, sr=None)
        print(f"Processing voice sample: {os.path.basename(voice_sample_path)}")
        display(Audio(data=y, rate=sr))
    except Exception as e:
        print(f"Error playing audio: {e}")
    
    # Extract voice features
    voice_features = extract_audio_features(voice_sample_path)
    if voice_features is None:
        print("❌ PREDICTION BLOCKED: Voice verification failed")
        return
    
    # Voice verification
    voice_prediction = voice_model.predict([voice_features])[0]
    voice_proba = voice_model.predict_proba([voice_features])[0]
    voice_confidence = np.max(voice_proba)
    
    print(f"Voice Verification Result: {voice_prediction}")
    print(f"Voice Confidence: {voice_confidence:.4f}")
    
    # Step 4: The system determines if the prediction is allowed
    print("\\n" + "-"*60)
    print("⚖️  SYSTEM DECISION")
    print("-"*60)
    
    if voice_confidence < 0.7:
        print("❌ PREDICTION BLOCKED: Voice not verified with sufficient confidence")
        print("   → Prediction cannot be executed without proper voice confirmation")
        return
    
    if face_prediction != voice_prediction:
        print(f"❌ PREDICTION BLOCKED: Identity mismatch detected")
        print(f"   → Face identity: {face_prediction}")
        print(f"   → Voice identity: {voice_prediction}")
        print("   → Prediction cannot be executed due to security concerns")
        return
    
    # All checks passed - prediction is allowed
    print("✅ ALL VERIFICATIONS PASSED")
    print("✅ VOICE CONFIRMED: Identity verified")
    print("✅ PREDICTION ALLOWED")
    
    print("\\n" + "="*60)
    print("🎉 FINAL PRODUCT RECOMMENDATION")
    print("="*60)
    print(f"Customer: {customer_id} ({face_prediction})")
    print(f"Recommended Product Category: {product_prediction}")
    print(f"Confidence Level: {product_confidence:.4f}")
    print("Status: ✅ APPROVED AND EXECUTED")
    print("="*60)

def run_product_access_demo():
    """Run demonstrations of the product access system."""
    print("\\n" + "="*80)
    print("PRODUCT PREDICTION MODEL ACCESS SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Scenario 1: Successful access and prediction
    print("\\n\\n" + "*"*80)
    print("SCENARIO 1: SUCCESSFUL ACCESS AND PREDICTION")
    print("*"*80)
    print("User with valid face and voice attempts to access product prediction model")
    input("Press Enter to continue...")
    
    face_image_path = os.path.join(IMAGE_DIR, 'member1', 'neutral.jpeg')
    voice_sample_path = os.path.join(AUDIO_DIR, 'member1', 'confirm_1.wav')
    product_access_system(face_image_path, voice_sample_path)
    
    # Scenario 2: Face not recognized - access denied
    print("\\n\\n" + "*"*80)
    print("SCENARIO 2: ACCESS DENIED - FACE NOT RECOGNIZED")
    print("*"*80)
    print("Unknown user attempts to access product prediction model")
    input("Press Enter to continue...")
    
    face_image_path = os.path.join(IMAGE_DIR, 'member2', 'neutral.jpeg')
    voice_sample_path = os.path.join(AUDIO_DIR, 'member1', 'confirm_1.wav')
    
    # Create blurred face for unauthorized attempt
    img = cv2.imread(face_image_path)
    if img is not None:
        blurred_img = cv2.GaussianBlur(img, (99, 99), 30)
        temp_path = 'temp_unauthorized_face.jpg'
        cv2.imwrite(temp_path, blurred_img)
        
        product_access_system(temp_path, voice_sample_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Scenario 3: Face recognized but voice verification fails
    print("\\n\\n" + "*"*80)
    print("SCENARIO 3: PREDICTION BLOCKED - VOICE VERIFICATION FAILED")
    print("*"*80)
    print("Authorized user accesses system but voice confirmation fails")
    input("Press Enter to continue...")
    
    face_image_path = os.path.join(IMAGE_DIR, 'member1', 'neutral.jpeg')
    voice_sample_path = os.path.join(AUDIO_DIR, 'member2', 'confirm_1.wav')
    product_access_system(face_image_path, voice_sample_path)
    
    print("\\n\\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)

# Main function
if __name__ == "__main__":
    run_product_access_demo()