import cv2
import pandas as pd
import os

def extract_frames_with_index_names(video_file, csv_file, output_dir):
    # Load data from CSV
    data = pd.read_csv(csv_file, header=None, names=["index", "timestamp", "steering", "throttle"])
    indices = data["index"].values
    timestamps = data["timestamp"].astype(float).values  # Ensure timestamps are floats

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps  # Total duration of the video in seconds

    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {video_duration}s")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames for each timestamp
    for index, target_timestamp in zip(indices, timestamps):
        # Calculate the frame index closest to the target timestamp
        closest_frame_idx = round(target_timestamp * fps)

        # Cap the frame index within valid bounds
        closest_frame_idx = min(max(0, closest_frame_idx), frame_count - 1)

        # Set the video capture to the closest frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, closest_frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at index {closest_frame_idx}")
            continue

        # Generate filename using the index from the CSV
        output_filename = os.path.join(output_dir, f"{index}.jpg")

        # Save the frame as JPG
        cv2.imwrite(output_filename, frame)
        print(f"Saved frame {closest_frame_idx} for timestamp {target_timestamp} as {output_filename}")

    cap.release()
    print("Finished extracting frames.")

# Input files
video_file = "input.mp4"
csv_file = "data_with_index.csv"
output_dir = "frames"

# Extract frames based on CSV timestamps
extract_frames_with_index_names(video_file, csv_file, output_dir)

