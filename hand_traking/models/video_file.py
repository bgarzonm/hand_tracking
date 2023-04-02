import cv2
import mediapipe as mp
import time
import csv

mp_hand = mp.solutions.hands

hands = mp_hand.Hands()

mp_drawing_utils = mp.solutions.drawing_utils
prev_time = 0

mp_drawing_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture('../data/video_file.mp4') 


with open('hand_landmarks.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'ID', 'X', 'Y', 'Z'])

    while True:
        success, img = cap.read()

        if not success:
            break

        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = landmark.x * w, landmark.y * h, landmark.z
                    writer.writerow([hand_idx, lm_idx, cx, cy, cz])

                    mp_drawing_utils.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_drawing_style.get_default_hand_landmarks_style(),
                        mp_drawing_style.get_default_hand_connections_style()
                    )

        current_time = time.time()
        FPS = int(1 / (current_time - prev_time))
        prev_time = current_time

        cv2.putText(img, str(FPS), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
