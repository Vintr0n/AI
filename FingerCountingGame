import cv2
import mediapipe as mp
import random
import pygame
import time

correct_answer = False

pygame.mixer.init()
sound = pygame.mixer.Sound('correct.mp3')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hold_threshold = 1.2

random_number = random.randint(1, 10)

# For webcam input:
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

        # Initially set finger count to 0 for each cap
        fingerCount = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                handLandmarks = []

                # Fill list with x and y positions of each landmark
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                # Test conditions for each finger: Count is increased if finger is 
                # considered raised.
                # Thumb: TIP x position must be greater or lower than IP x position, 
                # depending on hand label.
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount = fingerCount+1

                # Other fingers: TIP y position must be lower than PIP y position, 
                # as image origin is in the upper left corner.
                if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                    fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                    fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                    fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                    fingerCount = fingerCount+1

                # Draw hand landmarks 
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display finger count
        if fingerCount != 0:
            cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
            
        cv2.putText(image, str(random_number), (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
        
        if fingerCount != 0:
            if prev_finger_count is None or fingerCount != prev_finger_count:
                # Start timer when a new finger position is detected
                start_time = time.time()
            elif time.time() - start_time >= hold_threshold:
                # Store finger count if hold time exceeds threshold
                stored_finger_count = fingerCount
                if fingerCount == random_number and not correct_answer:
                    #cv2.putText(image, str("Correct!"), (550, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 204, 0), 10)
                    sound.play()
                    #pygame.time.wait(int(sound.get_length() * 1000))
                    #correct_answer = True
                    cv2.putText(image, str(stored_finger_count), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                    random_number = random.randint(1, 10)
        else:
            # Reset timer and previous finger count when hand position is not detected
            start_time = None
            prev_finger_count = None

        # Update previous finger count
        prev_finger_count = fingerCount
            
        # Display image
        cv2.imshow('Finger counting game', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# Clean up
cap.release()
cv2.destroyAllWindows()
