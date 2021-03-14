import requests
import json
import cv2
import base64
import mediapipe as mp

def write_to_file(save_path, data):
  with open(save_path, "wb") as f:
    f.write(base64.b64decode(data))

def lambda_handler(event, context):
    write_to_file("/tmp/photo.jpg", event["body"])
    
    image = cv2.imread("/tmp/photo.jpg")
    mp_hands = mp.solutions.hands

    # For static images:
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return {
        "statusCode": 200,
        "headers": {},
        "body":  "no hands found"
        }

    if(len(results.multi_handedness) == 1):
        isMultiHand = False
    else:
        isMultiHand = True
    # results.multi_handedness[0] is first detected hand
    if(results.multi_handedness[0].classification[0].index == 0):  # Index 0 is Left, 1 is Right
        rightHandFirst = False
    else:
        rightHandFirst = True

    if results.multi_hand_landmarks:

        rightHandPoints = []
        leftHandPoints = []

        for hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if(rightHandFirst):                       # First hand (0) is Right, Second hand (1) is Left
                if(hand == 0): 
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        rightHandPoints.append((landmark.x, landmark.y))
                else:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        leftHandPoints.append((landmark.x, landmark.y))
            else:                                     # First hand (0) is Left, Second hand (1) is Right
                if(hand == 0):
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        leftHandPoints.append((landmark.x, landmark.y))
                else: 
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        rightHandPoints.append((landmark.x, landmark.y))
    hands.close()

    body = {"rightHandPoints" : rightHandPoints , "leftHandPoints" :leftHandPoints}
    return {
      "statusCode": 200,
      "body":  body
    }