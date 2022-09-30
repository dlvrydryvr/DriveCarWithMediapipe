import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark
import json
import threading

NAME_LIST = []
global_thread = None

hand_index_txt= r''

for name in HandLandmark:
    name = str(name).split('.')[-1]
    NAME_LIST.append(name)

with open(hand_index_txt, 'wt') as f:
    for i,n in enumerate(NAME_LIST):
        f.write(n)
        if i != len(NAME_LIST)-1:
            f.write('\n')


def write_json_file(left_right:str, pose_dict):
    # alternatives of flipping image
    if left_right == "Left":
        left_right = "Right"
    elif left_right == "Right":
        left_right = "Left"

    # Writing .json
    try:
        hand_data_json = '' + f'/hand_{left_right}.json'
        with open(hand_data_json, "w") as outfile:
            json_object = json.dumps(pose_dict, indent=4)
            outfile.write(json_object)
    except Exception as e:
        print("Failed To Access", e)
        global_thread.killed = True

def detect_hand():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
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

                # counting for discriminating Left-Right
                ## see write_json_file.
                count = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                    
                    pose_dict = {}
                    for i in range(len(hand_landmarks.landmark)):
                        # Unreal Coordinate System : X Forward, Z Up
                        # MP Coordinate System     : https://google.github.io/mediapipe/solutions/objectron.html#coordinate-systems
                        pose_dict[NAME_LIST[i]] = {'X': hand_landmarks.landmark[i].z , 'Y': hand_landmarks.landmark[i].x , 'Z':hand_landmarks.landmark[i].y * -1}
                    
                    # Don't miss anything!
                    assert(i==len(HandLandmark)-1)

                    left_right = results.multi_handedness[count].classification[0].label                
                    write_json_file(left_right, pose_dict)
                    count += 1
                
            # Flip the image horizontally for a selfie-view display
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        global_thread.killed = True

if __name__ == "__main__":
  global_thread = threading.Thread(target=detect_hand)
  global_thread.start()