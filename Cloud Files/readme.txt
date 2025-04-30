Feature Extraction - have all files in the same folder
Use the feature extraction folder - for environment, requirements are there
Video Path: Path to the video file
Output Path: Path to save the extracted features
Usage: python feat.py <video_path> <output_path>



Model- have all the files in the same folder
Use the model folder- for environment requirements are there
Path to model: Path to the model state dictionary
Path to hello csv: Hello.csv file is the label map
Path to features: Path to the extracted features

Usage: python3 -m inference <path_to_your_model_state_dict> <path_to_hello_csv> <path_to_features>
