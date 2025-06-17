import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
file_path = ".\\Landmark_Model\\hand_landmarker.task"  # Replace with your file name

model = os.path.abspath(file_path)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


'''
To Do:
- clean code to helpers
- make pointer finger follow finger
- normalize coordinates for finger (future for gesture recognition


'''




def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def draw_point(image, landmarks):
    image_rows, image_cols, _ = image.shape
    for landmark in landmarks:
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    # White circle border
        circle_border_radius = 3
        circle_radius = 2
        thickness = 2
        WHITE_COLOR = (224, 224, 224)
        RED_COLOR = (0, 0, 225)
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                   thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, circle_radius,
                   RED_COLOR, thickness)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    )


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        landmark_point = calc_landmark_list(image, hand_landmarks)
        index_finger_hand = [hand_landmarks.landmark[0], hand_landmarks.landmark[5], hand_landmarks.landmark[6], hand_landmarks.landmark[7], hand_landmarks.landmark[8]]
        draw_point(image, index_finger_hand)

        index_finger = [landmark_point[0], landmark_point[5], landmark_point[6], landmark_point[7], landmark_point[8]]
        print("index finger: ", index_finger)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
