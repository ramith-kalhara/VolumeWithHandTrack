import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()  # (-96.0, 0.0)
min_vol = vol_range[0]
max_vol = vol_range[1]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip for mirror image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    lmList = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            # If landmarks are detected
            if lmList:
                # Index finger tip: id 8, Thumb tip: id 4, Middle tip: id 12
                x1, y1 = lmList[8][1], lmList[8][2]    # Index
                x2, y2 = lmList[12][1], lmList[12][2]  # Middle

                # Draw circles on tips
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Distance between index and middle fingers
                length = hypot(x2 - x1, y2 - y1)

                # Convert length to volume
                vol = np.interp(length, [30, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                # Display volume level
                vol_bar = np.interp(length, [30, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)

    cv2.imshow("Hand Tracking Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
