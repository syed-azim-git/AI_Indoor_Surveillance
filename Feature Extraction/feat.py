import sys
from feature_extractor import KeyFrameExtractor, extract_mfcc_features, preprocess_for_mobilenet
import h5py
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model

base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

if len(sys.argv) != 3:
    print("Usage: python feat.py <video_path> <output_h5_path>")
    sys.exit(1)

def extract_features_from_single_video(video_path, output_h5_path):
    """Extract visual and audio features from a single video file"""    
    try:
        # Extract keyframes
        print(f"Extracting keyframes from {video_path}...")
        extractor = KeyFrameExtractor(video_path, segment_duration=0.5)
        keyframes = extractor.process()
        
        if len(keyframes) < 20:
            print(f"Warning: Video has fewer than 20 keyframes ({len(keyframes)})")
            # Duplicate the last frame to reach 20 frames
            last_frame = keyframes[-1] if keyframes else None
            if last_frame is not None:
                keyframes.extend([last_frame] * (20 - len(keyframes)))
            else:
                raise Exception("No keyframes extracted")
        
        keyframes = keyframes[:20]
        
        #print("Preprocessing frames for MobileNet...")
        preprocessed_frames = preprocess_for_mobilenet(keyframes)
        
        #print("Extracting visual features using MobileNet...")
        visual_features = base_model.predict(preprocessed_frames, verbose=0)
        
        # Extract MFCC features
        #print("Extracting MFCC audio features...")
        mfcc_features = extract_mfcc_features(video_path, n_mfcc=40)
        
        print(f"Visual features shape: {visual_features.shape}")
        print(f"MFCC features shape: {mfcc_features.shape}")
        
        # Save features to H5 file
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('visual_features', data=visual_features)
            f.create_dataset('audio_features', data=mfcc_features)
            f.attrs['video_path'] = video_path
        
        print(f"Features saved to {output_h5_path}")
        
        # Display a sample keyframe
        '''plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(keyframes[0], cv2.COLOR_BGR2RGB))
        plt.title("Sample Keyframe")
        plt.axis('off')
        plt.show()'''
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None, None

if __name__ == "__main__":
    video_path = sys.argv[1] 
    output_h5_path = sys.argv[2]
    extract_features_from_single_video(video_path, output_h5_path)