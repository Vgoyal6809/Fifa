#!/usr/bin/env python
# coding: utf-8

#  # Part 1:  Video Upload and Frame Extraction

# 1. Install the Required Libraries

# In[19]:


import cv2
import os


# 2. Directory for video uplaod and frame extraction

# In[20]:


# Function to create directory for storing frames
def create_frame_dir(video_file):
    frame_dir = os.path.splitext(video_file)[0] + "_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    return frame_dir

# Function to extract and save frames from the video
def extract_frames(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    # Create directory to store the frames
    frame_dir = create_frame_dir(video_file)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # If no frame is captured, exit
        
        # Save frame as image
        frame_path = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        print(f"Extracted frame {frame_count}")
    
    cap.release()
    print(f"Frame extraction completed. Frames are saved in {frame_dir}")

# Call the function with your video file path
video_file = "Video2.mp4"  # Replace with your video file
extract_frames(video_file)


# # Part 2: Player Detection Using YOLOv5

# Step 1: Install YOLOv5

# In[21]:


import torch
from pathlib import Path


# Step 2: Use YOLOv5 to detect players in frames.

# In[22]:


# Load YOLOv5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Detect players in the extracted frames
def detect_players_in_frames(frame_dir):
    frame_paths = list(Path(frame_dir).glob('*.jpg'))
    
    for frame_path in frame_paths:
        # Perform detection
        results = model(frame_path)
        
        # Display results (bounding boxes around detected players)
        # results.show()  # This will show the image with detection

        # If you want to save the results, uncomment the following line
        # results.save()  # This will save images with bounding boxes to the current directory
        
        # print(f"Detected players in {frame_path}")

# Call the function with the frame directory
frame_dir = os.path.splitext(video_file)[0] + "_frames"  # Replace with your frame directory
detect_players_in_frames(frame_dir)


# # Part 3: Tracking the Player Across Frames

# Step 1: Install DeepSORT

# In[23]:


import numpy as np
import sklearn

print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)



# In[24]:


import sys


# In[25]:


from deep_sort_realtime.deepsort_tracker import DeepSort


# Step 2: Player tracking with DeepSORT

# In[26]:


# Import necessary libraries
import cv2
import torch
from pathlib import Path
import numpy as np

# Initialize the YOLOv5 model (Ensure that it's already set up)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Assuming the DeepSORT object is replaced with a custom tracker or placeholder
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}

    def update(self, bboxes):
        new_objects = {}
        for bbox in bboxes:
            new_objects[self.next_id] = bbox
            self.next_id += 1
        self.objects = new_objects
        return [[*bbox, obj_id] for obj_id, bbox in self.objects.items()]

# Initialize a simple tracker
tracker = SimpleTracker()

# Function to track players across frames
def track_players_in_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is captured

        # Perform detection (similar to part 2)
        results = model(frame)
        bboxes = results.xyxy[0][:, :4].cpu().numpy()  # Bounding boxes
        confs = results.xyxy[0][:, 4].cpu().numpy()    # Confidence scores

        # Perform tracking using a simple tracker (replace deepsort)
        outputs = tracker.update(bboxes)

        # Draw the bounding boxes and track IDs on the frame
        for output in outputs:
            x1, y1, x2, y2, track_id = output[:5]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow("Tracking", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
   

# Call the tracking function
video_file = "Video2.mp4"  # Replace with your video file
track_players_in_video(video_file)


# # Part 4: Pose Estimation Using OpenPose

# Step 1: Install OpenPose

# In[27]:


import cv2
import mediapipe as mp


# Step 2: Implement Pose Estimation on Extracted Frames

# In[28]:


import cv2
import torch
import mediapipe as mp

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load YOLOv5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to perform pose estimation on a single frame
def pose_estimation_on_frame(frame):
    # Convert frame to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)

    # Draw keypoints on the frame
    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return annotated_frame, results.pose_landmarks

# Function to detect players and perform pose estimation
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is captured

        # Perform detection using YOLOv5
        results = model(frame)
        bboxes = results.xyxy[0].cpu().numpy()  # Bounding boxes

        # Perform pose estimation
        annotated_frame, landmarks = pose_estimation_on_frame(frame)

        # Draw bounding boxes for detected players
        for bbox in bboxes:
            # Check the number of values in bbox
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox  # If only bounding box coordinates are returned
                conf = 1.0  # Set default confidence if not available
            elif len(bbox) >= 5:
                x1, y1, x2, y2, conf = bbox[:5]  # Get the first five values
            else:
                continue  # Skip if bbox has fewer than 4 values

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Display the annotated frame with pose landmarks and bounding boxes
        cv2.imshow("Pose Estimation and Player Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function with your video file
video_file = "Video2.mp4"  # Use the video file specified earlier
process_video(video_file)


# # Part 5: Action Recognition

# Step 1: Define Actions and Recognition Logic

# In[29]:


import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to perform pose estimation on a single frame
def pose_estimation_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    annotated_frame = frame.copy()

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return annotated_frame, results.pose_landmarks

# Action recognition function
def recognize_action(landmarks):
    if landmarks is not None:
        # Define keypoints
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Calculate angles
        shoulder_angle_left = calculate_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y)
        )

        shoulder_angle_right = calculate_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )

        hip_angle_left = calculate_angle(
            (left_hip.x, left_hip.y),
            (left_knee.x, left_knee.y),
            (left_ankle.x, left_ankle.y)
        )

        hip_angle_right = calculate_angle(
            (right_hip.x, right_hip.y),
            (right_knee.x, right_knee.y),
            (right_ankle.x, right_ankle.y)
        )

        # Debug outputs for angles
        print(f"Left Shoulder Angle: {shoulder_angle_left}")
        print(f"Right Shoulder Angle: {shoulder_angle_right}")
        print(f"Left Hip Angle: {hip_angle_left}")
        print(f"Right Hip Angle: {hip_angle_right}")

        # Define action recognition based on calculated angles
        if shoulder_angle_left < 30 and shoulder_angle_right < 30:  # Shooting
            return "Action: Shooting"
        elif shoulder_angle_left > 150 and shoulder_angle_right > 150:  # Running
            return "Action: Running"
        elif 30 <= shoulder_angle_left <= 150 and 30 <= shoulder_angle_right <= 150:  # Neutral position
            return "Action: Neutral"
        elif hip_angle_left < 30 and hip_angle_right < 30:  # Jumping
            return "Action: Jumping"
        elif hip_angle_left > 160 and hip_angle_right > 160:  # Standing
            return "Action: Standing"
        elif 60 <= hip_angle_left <= 120 and 60 <= hip_angle_right <= 120:  # Bending
            return "Action: Bending"
        elif 45 <= hip_angle_left <= 90 and 45 <= hip_angle_right <= 90:  # Walking
            return "Action: Walking"
        else:
            return "Action: Unknown"  # Default case

    return "Action: Unknown"

# Process all frames for pose estimation and action recognition
def process_pose_estimation_on_video_with_action_recognition(video_file):
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, landmarks = pose_estimation_on_frame(frame)

        # Debug output for landmarks
        if landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                print(f"Landmark {idx}: ({landmark.x}, {landmark.y})")

        action = recognize_action(landmarks)
        print(action)

        cv2.imshow("Pose Estimation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the action recognition function with your video
video_file = "Video2.mp4"  # Use the video file specified earlier
process_pose_estimation_on_video_with_action_recognition(video_file)


# # Part 6: Generating Feedback for Players

# Step 1: Generate Feedback Based on Recognized Actions

# In[30]:


# Part 6: Generating Feedback for Players

def generate_feedback(action):
    feedback = ""
    
    # Provide feedback based on the recognized action
    if action == "Action: Shooting":
        feedback = "Feedback: Focus on improving your shooting technique. Aim for accuracy."
    elif action == "Action: Running":
        feedback = "Feedback: Maintain a steady pace and good form while running."
    elif action == "Action: Neutral":
        feedback = "Feedback: Keep practicing your movements for better fluidity."
    elif action == "Action: Jumping":
        feedback = "Feedback: Focus on explosive power and proper landing technique."
    elif action == "Action: Standing":
        feedback = "Feedback: Maintain a balanced and stable stance. Avoid slouching."
    elif action == "Action: Bending":
        feedback = "Feedback: Ensure your knees are aligned with your toes and your back is straight."
    elif action == "Action: Walking":
        feedback = "Feedback: Practice smooth and controlled strides to enhance your walking efficiency."
    else:
        feedback = "Feedback: Keep practicing your actions for better performance."
    
    return feedback

# Modify the video processing function to include feedback generation
def process_pose_estimation_on_video_with_feedback(video_file):
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform pose estimation
        annotated_frame, landmarks = pose_estimation_on_frame(frame)

        # Recognize action
        action = recognize_action(landmarks)
        print(action)

        # Generate feedback
        feedback = generate_feedback(action)
        print(feedback)

        # Display the result
        cv2.imshow("Pose Estimation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the feedback generation function with your video
video_file = "Video2.mp4"  # Use the video file specified earlier
process_pose_estimation_on_video_with_feedback(video_file)


# # Now Simple Implementation of getting video and anlysis them

# 1. Testing and Validation

# In[31]:


def analyze_video(video_path):
    # Placeholder for video analysis logic
    # Replace this with actual video analysis code
    # For now, let's just return a dummy feedback
    return f"Analysis completed for {video_path}"

def test_videos(video_list):
    results = []
    for video_path in video_list:
        feedback = analyze_video(video_path)  # Call the video analysis function
        results.append((video_path, feedback))
    return results

# Sample usage with your specified video file
video_list = ['Video2.mp4']  # Use your specific video file
results = test_videos(video_list)
for video, feedback in results:
    print(f"Feedback for {video}: {feedback}")


# In[32]:


action_threshold = 0.5  # Initial threshold

def set_action_threshold(new_threshold):
    global action_threshold
    action_threshold = new_threshold

# Example of updating the threshold
set_action_threshold(0.6)


# 2. Data Logging

# In[33]:


import csv
from datetime import datetime

# Function to log feedback for each frame with timestamp
def log_feedback(video_path, action, feedback, frame_number):
    with open('feedback_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), video_path, frame_number, action, feedback])

# Example of how the function would be used within your video processing loop
def process_pose_estimation_on_video_with_feedback(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_number = 0  # Initialize frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform pose estimation
        annotated_frame, landmarks = pose_estimation_on_frame(frame)

        # Recognize action
        action = recognize_action(landmarks)
        print(action)

        # Generate feedback
        feedback = generate_feedback(action)
        print(feedback)

        # Log the feedback for this frame
        log_feedback(video_file, action, feedback, frame_number)

        # Display the result
        cv2.imshow("Pose Estimation", annotated_frame)
        
        # Increment frame number
        frame_number += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Sample logging within video processing function
video_file = "Video2.mp4"  # Use the video file specified earlier
process_pose_estimation_on_video_with_feedback(video_file)



# In[34]:


import sqlite3
from datetime import datetime

# Function to initialize the database with the appropriate table
def initialize_db():
    conn = sqlite3.connect('player_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            video_path TEXT,
            frame_number INTEGER,
            action TEXT,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert performance data into the database
def insert_performance(video_path, frame_number, action, feedback):
    conn = sqlite3.connect('player_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO performance (timestamp, video_path, frame_number, action, feedback) 
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), video_path, frame_number, action, feedback))
    conn.commit()
    conn.close()

# Initialize the database
initialize_db()

# Modified video processing function to log actions and feedback frame by frame
def process_pose_estimation_on_video_with_feedback(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_number = 0  # Initialize frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform pose estimation
        annotated_frame, landmarks = pose_estimation_on_frame(frame)

        # Recognize action
        action = recognize_action(landmarks)
        print(action)

        # Generate feedback
        feedback = generate_feedback(action)
        print(feedback)

        # Insert the feedback for this frame into the database
        insert_performance(video_file, frame_number, action, feedback)

        # Display the result
        cv2.imshow("Pose Estimation", annotated_frame)
        
        # Increment frame number
        frame_number += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the video processing function
video_file = "Video2.mp4"  # Use the video file specified earlier
process_pose_estimation_on_video_with_feedback(video_file)


# 3. User Interface

# In[ ]:


import cv2
import mediapipe as mp
import torch
from datetime import datetime
import numpy as np
import tkinter as tka
from tkinter import ttk, filedialog, messagebox
import pytesseract

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the correct Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Initialize drawing utilities for Mediapipe
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv5 model
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

# Helper function: Calculate angle
def calculate_angle(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    return abs(angle) if angle >= 0 else abs(angle + 360)

# Recognize action
def recognize_action(landmarks):
    if landmarks is not None:
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        shoulder_angle = calculate_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y)
        )

        if shoulder_angle < 30:
            return "Shooting", "Improve shooting accuracy."
        elif 30 <= shoulder_angle < 45:
            return "Dribbling", "Maintain low center of gravity."
        elif 45 <= shoulder_angle < 75:
            return "Standing", "Maintain balance and posture."
        elif 75 <= shoulder_angle < 105:
            return "Jumping", "Enhance explosive power and landing."
        elif 105 <= shoulder_angle < 130:
            return "Bending", "Align knees and back properly."
        elif 130 <= shoulder_angle < 150:
            return "Walking", "Focus on smooth strides."
        elif 150 <= shoulder_angle <= 180:
            return "Running", "Keep steady pace and form."
    return None, None

# Recognize player ID using OCR
def recognize_player_id(cropped_frame):
    grayscale_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(grayscale_frame, config='--psm 8')
    player_id = ''.join(filter(str.isdigit, text))
    return player_id if player_id else "Unknown"

# Analyze video
def analyze_video(video_path, player_data):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        bbox_list = results.xywh[0].cpu().numpy()

        for bbox in bbox_list:
            x1, y1, w, h, conf, cls = bbox[:6]
            if conf > 0.5:
                x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
                cropped_frame = frame[y1:y2, x1:x2]
                player_id = recognize_player_id(cropped_frame)
                if player_id == "Unknown":
                    continue

                # Initialize player data if not present
                if player_id not in player_data:
                    player_data[player_id] = {
                        "actions": {},
                        "passes": 0,
                        "max_speed": 0,
                        "avg_speed": 0,
                        "speed_sum": 0,
                        "speed_count": 0,
                    }

                result = pose.process(cropped_frame)
                if result.pose_landmarks:
                    action, feedback = recognize_action(result.pose_landmarks)

                    if action:
                        # Count actions
                        player_data[player_id]["actions"].setdefault(action, []).append(feedback)

                        # Simulate passes and speed calculation
                        player_data[player_id]["passes"] += 1
                        speed = np.random.uniform(1.5, 7.0)  # Simulated speed in m/s
                        speed_kmh = speed * 3.6  # Convert to km/h
                        player_data[player_id]["max_speed"] = max(player_data[player_id]["max_speed"], speed_kmh)
                        player_data[player_id]["speed_sum"] += speed_kmh
                        player_data[player_id]["speed_count"] += 1

    # Calculate average speed
    for player in player_data.values():
        if player["speed_count"] > 0:
            player["avg_speed"] = player["speed_sum"] / player["speed_count"]

    data = player_data
    cap.release()

# Display stats for players with scroll and color customization
def show_stats(player_data, selected_ids):
    stats_window = tka.Toplevel()
    stats_window.title("Player Stats")
    stats_window.geometry("800x600")

    # Scrollable frame
    canvas = tka.Canvas(stats_window)
    scrollbar = tka.Scrollbar(stats_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tka.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Enable mouse wheel scrolling
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

    tka.Label(scrollable_frame, text="Player Stats Analysis", font=("Arial", 16, "bold")).pack(pady=10)

    for pid in selected_ids:
        if pid not in player_data:
            continue

        tka.Label(scrollable_frame, text=f"Player ID: {pid}", font=("Arial", 14, "bold")).pack(pady=5)

        # Display max speed and average speed
        tka.Label(
            scrollable_frame,
            text=f"  Max Speed: {player_data[pid]['max_speed']:.2f} km/h\n  Avg Speed: {player_data[pid]['avg_speed']:.2f} km/h",
            font=("Arial", 12),
        ).pack(pady=5)

        # Create table for actions
        tree = ttk.Treeview(scrollable_frame, columns=("Action", "Feedback", "Count"), show="headings", height=5)
        tree.heading("Action", text="Action")
        tree.heading("Feedback", text="Feedback")
        tree.heading("Count", text="Count")
        tree.pack(fill="x", pady=5)

        actions = player_data[pid]["actions"]
        for i, (action, feedback_list) in enumerate(actions.items()):
            for feedback in set(feedback_list):
                tree.insert(
                    "",
                    "end",
                    values=(action, feedback, feedback_list.count(feedback)),
                    tags=("oddrow" if i % 2 == 0 else "evenrow",),
                )
                tree.tag_configure("oddrow", background="#f0f8ff")
                tree.tag_configure("evenrow", background="#e6e6fa")

# Select Player Stats
def select_player(player_data):
    selection_window = tka.Toplevel()
    selection_window.title("Select Player ID")
    selection_window.geometry("300x300")

    tka.Label(selection_window, text="Select Player IDs (Ctrl+Click for multiple):").pack(pady=10)

    player_listbox = tka.Listbox(selection_window, selectmode="multiple")
    player_listbox.pack(fill="both", expand=True, pady=10)

    # Add player IDs to the listbox
    for player_id in player_data.keys():
        player_listbox.insert("end", player_id)

    def view_selected_stats():
        selected_indices = player_listbox.curselection()
        selected_ids = [player_listbox.get(i) for i in selected_indices]
        if not selected_ids:
            messagebox.showinfo("Info", "No player selected.")
            return
        show_stats(player_data, selected_ids)
        selection_window.destroy()

    tka.Button(selection_window, text="View Stats", command=view_selected_stats).pack(pady=10)


