import cv2
import mediapipe as mp
import time

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to detect SOS gesture (waving hand 3 times)
def detect_sos_gesture(landmarks):
    global wave_count, last_wave_time, hand_moving
    current_time = time.time()

    # Calculate movement (based on x-coordinate of hand landmarks)
    wrist_x = landmarks[0].x
    hand_direction = 'right' if wrist_x > 0.5 else 'left'

    if hand_moving is None:
        hand_moving = hand_direction
    elif hand_moving != hand_direction:
        hand_moving = hand_direction
        if current_time - last_wave_time < 2:  # Ensure rapid waving within 2 seconds
            wave_count += 1
        else:
            wave_count = 1  # Reset if too slow
        last_wave_time = current_time

    # SOS detected if waved 3 times
    if wave_count >= 3:
        return True
    return False


# Camera setup
cap = cv2.VideoCapture(0)
wave_count = 0
last_wave_time = 0
hand_moving = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # If hand landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect SOS gesture
                if detect_sos_gesture(hand_landmarks.landmark):
                    print("SOS Gesture Detected! Sending alert...")
                    cv2.putText(frame, "SOS Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Trigger alert logic here (sound alarm, send message, etc.)
                    wave_count = 0  # Reset after alert

        # Display the frame with hand landmarks
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
