import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def finger_state(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = {}
        
    def up(tip,base): return lm[tip].y< lm[base].y
    fingers['thumb'] = lm[4].x < lm[3].x
    fingers['index'] = up(8,5)
    fingers['middle'] = up(12,9)
    fingers['ring'] = up(16,13)
    fingers['pinky'] = up(20,17)
    return fingers
def classify_gesture(fingers):
        if all(fingers.values()):
             return "open palm!"
        if not any(fingers.values()):
             return "fist!"
        if fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and fingers['pinky']:
             return "rock star!"
        if not fingers['thumb'] and fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
             return "peace!"
        if fingers['thumb'] and not fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
             return "thumbs up!"
        return "Unknown gesture"
import os
import pyautogui

def do_action(gesture):
    if gesture == "thumbs up!":
        print("Play!")
        os.system("open /Applications/Spotify.app")
    elif gesture == "peace!":
        print("Pause")
        pyautogui.press("space")
    elif gesture == "fist!":
        print("stop!")
        pyautogui.hotkey("control", "w")
    elif gesture == "rock star!":
        print("volume up!")
        pyautogui.press("volumeup")
    elif gesture == "open palm":
        print("mute!")
        pyautogui.press("volumemute")  
cap = cv2.VideoCapture(1)
with mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    if i >= len(result.multi_handedness):
                        continue
                    label = result.multi_handedness[i].classification[0].label
                    mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                    fingers = finger_state(hand_landmarks)
                    gesture = classify_gesture(fingers)
                    last_gesture = None
                    gesture_cooldown = 10
                    gesture_frame_count = 0
                    if gesture != last_gesture or gesture_frame_count > gesture_cooldown:
                        do_action(gesture)
                        last_gesture = gesture
                        gesture_frame_count = 0
                    else:
                        gesture_frame_count += 1
                    full_label = f"{label} hand: {gesture}"
                    cv2.putText(frame,full_label,(10,50+ i *40),cv2.FONT_HERSHEY_SIMPLEX,1,(225,225,225),2) 
            cv2.imshow("gesture detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break    
cap.release()
cv2.destroyAllWindows()


