import os
import yaml
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

# Set Qt platform for OpenCV on Wayland systems (avoids GUI errors)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load dataset directory path from YAML config
with open('/home/keys/me_meow/code/python_projects/sign_language_analyzer/src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize MediaPipe models
mp_hands = mp.solutions.hands                           # Hand landmark detector
mp_drawing = mp.solutions.drawing_utils                 # Drawing utility
mp_drawing_styles = mp.solutions.drawing_styles         # Predefined landmark/connection styles

# Create hand tracker instance
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Read dataset directory from config
DATA_DIR = config['directory']['dir']

data = []   # Will hold normalized hand landmark features
labels = [] # Corresponding labels (folder names)

# Go through each folder in dataset (each folder = 1 class)
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        # Convert image to RGB (MediaPipe expects RGB, OpenCV loads BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks in the image
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []  # stores normalized x/y pairs
                x_ = []        # list of all x coords
                y_ = []        # list of all y coords

                # First pass: collect raw x and y values
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Second pass: normalize relative to top-left of hand bbox
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Append the processed data and label
                data.append(data_aux)
                labels.append(label)

                # Optionally visualize the landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Preview the image with drawn landmarks
        cv2.imshow("Preview", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Save the dataset to a pickle file for future training
with open('/home/keys/me_meow/code/python_projects/sign_language_analyzer/lib//data/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Clean up OpenCV windows
cv2.destroyAllWindows()
