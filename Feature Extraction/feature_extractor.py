import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import cmeans
import h5py
import librosa
import subprocess
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from moviepy.editor import VideoFileClip

class KeyFrameExtractor:
    def __init__(self, video_path, segment_duration=0.5):
        self.video_path = video_path
        self.segment_duration = segment_duration
        self.key_frames = []
        self.key_frame_timestamps = []
        
    def extract_frames_by_segments(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frames per segment
        frames_per_segment = int(fps * self.segment_duration)
        
        # Process video segment by segment
        segment_start_frame = 0
        
        while segment_start_frame < total_frames:
            segment_end_frame = min(segment_start_frame + frames_per_segment, total_frames)
            
            # Extract frames for this segment
            segment_frames = []
            segment_frame_indices = []
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)
            for frame_idx in range(segment_start_frame, segment_end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                segment_frames.append(frame)
                segment_frame_indices.append(frame_idx)
            
            if len(segment_frames) > 0:
                # Process this segment to extract one key frame
                key_frame, key_frame_idx = self.process_segment(segment_frames, segment_frame_indices)
                
                # Calculate timestamp
                timestamp = key_frame_idx / fps
                
                self.key_frames.append(key_frame)
                self.key_frame_timestamps.append(timestamp)
                
            segment_start_frame = segment_end_frame
            
        cap.release()
        print(f"Extracted {len(self.key_frames)} key frames from {self.video_path}")
    
    def extract_color_features(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        return np.concatenate([h_hist, s_hist, v_hist]).flatten()
    
    def extract_texture_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        mag_hist = cv2.calcHist([magnitude.astype(np.float32)], [0], None, [8], [0, np.max(magnitude)])
        ang_hist = cv2.calcHist([angle.astype(np.float32)], [0], None, [8], [0, 360])
        
        mag_hist = cv2.normalize(mag_hist, mag_hist, 0, 1, cv2.NORM_MINMAX)
        ang_hist = cv2.normalize(ang_hist, ang_hist, 0, 1, cv2.NORM_MINMAX)
        
        return np.concatenate([mag_hist, ang_hist]).flatten()
    
    def extract_edge_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        h, w = edges.shape
        cell_h, cell_w = h // 4, w // 4
        cell_features = []
        
        for i in range(4):
            for j in range(4):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_density = np.sum(cell > 0) / (cell.shape[0] * cell.shape[1])
                cell_features.append(cell_density)
        
        return np.array([edge_density] + cell_features)
    
    def process_segment(self, segment_frames, segment_frame_indices):
        if len(segment_frames) == 1:
            return segment_frames[0], segment_frame_indices[0]
            
        # Extract features for all frames in this segment
        features = []
        for frame in segment_frames:
            color_features = self.extract_color_features(frame)
            texture_features = self.extract_texture_features(frame)
            edge_features = self.extract_edge_features(frame)
            
            combined_features = np.concatenate([
                color_features, 
                texture_features, 
                edge_features
            ])
            
            features.append(combined_features)
            
        features = np.array(features)
        
        # Normalize features
        features_normalized = (features - np.min(features, axis=0)) / (
            np.max(features, axis=0) - np.min(features, axis=0) + 1e-10)
        
        # Replace NaN values with 0
        features_normalized = np.nan_to_num(features_normalized, 0)
        
        # Apply Fuzzy C-means with just one cluster
        cntr, u, u0, d, jm, p, fpc = cmeans(
            features_normalized.T, 
            c=1,
            m=2, 
            error=0.005, 
            maxiter=1000, 
            init=None
        )
        
        # Find the frame closest to the cluster center
        distances = np.sum((features_normalized - cntr[0]) ** 2, axis=1)
        closest_frame_idx = np.argmin(distances)
        
        return segment_frames[closest_frame_idx], segment_frame_indices[closest_frame_idx]
    
    def process(self):
        self.extract_frames_by_segments()
        return self.key_frames[:20]  # Return only first 20 keyframes

def extract_audio(video_path, output_wav_path):
    """Extract audio from video file using moviepy instead of ffmpeg"""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_wav_path, 
                                    fps=44100,  # Sample rate
                                    nbytes=2,   # 16-bit audio
                                    codec='pcm_s16le')  # WAV format
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def extract_mfcc_features(video_path, n_mfcc=40, segment_duration=0.5, total_duration=10.0):
    """Extract MFCC features from a video at 0.5 second intervals, limited to 10 seconds"""
    # Create temporary WAV file with proper path handling
    base_name = os.path.splitext(video_path)[0]
    wav_path = f"{base_name}_temp.wav"
    
    try:
        # Extract audio using moviepy
        extract_audio(video_path, wav_path)
        
        # Load audio file
        y, sr = librosa.load(wav_path, sr=None, res_type='kaiser_fast')
        
        # Limit to exactly 10 seconds
        max_samples = int(sr * total_duration)
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Calculate samples per segment
        samples_per_segment = int(sr * segment_duration)
        
        # Calculate number of segments
        num_segments = int(total_duration / segment_duration)
        
        # Extract MFCC for each segment
        mfcc_features = []
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment
            
            # Handle case where audio is shorter than 10 seconds
            if end_sample > len(y):
                break
                
            segment = y[start_sample:end_sample]
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)  # Average across time
            mfcc_features.append(mfcc_mean)
        
        # Ensure we have exactly the expected number of segments
        if len(mfcc_features) < num_segments:
            # Pad with zeros if audio is shorter than 10 seconds
            for _ in range(num_segments - len(mfcc_features)):
                mfcc_features.append(np.zeros(n_mfcc))
        
        return np.array(mfcc_features)
    
    finally:
        # Clean up temporary WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)

def preprocess_for_mobilenet(frames):
    processed_frames = []
    for frame in frames:
        # Resize to MobileNet input size
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB (OpenCV uses BGR, but MobileNet expects RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Preprocess for MobileNet
        preprocessed = preprocess_input(rgb)
        processed_frames.append(preprocessed)
    return np.array(processed_frames)
