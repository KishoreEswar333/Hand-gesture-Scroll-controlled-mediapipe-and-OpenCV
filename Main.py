import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe hands module and drawing module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate hand gesture value
def calculate_gesture_value(hand_landmarks):
    if hand_landmarks is not None:
        # Get the landmarks for the thumb tip and the pinky tip
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Calculate the distance between the thumb tip and pinky tip
        distance = abs(thumb_tip.x - pinky_tip.x) + abs(thumb_tip.y - pinky_tip.y)

        # Normalize the distance to a value between 0 and 1 with one decimal place
        gesture_value = round(distance / 0.6, 1)  # Adjust the division factor as needed

        # Cap the gesture value to ensure it stays between 0 and 1
        gesture_value = max(0, min(1, gesture_value))

        return gesture_value
    return None

# Function to scroll based on gesture value range
def scroll_based_on_gesture(current_gesture_value):
    if 0.8 <= current_gesture_value <= 1.0:
        # Scroll up
        print("Scrolling Up")
        pyautogui.scroll(30)  # Adjust this value for scrolling up speed
    elif 0.4 <= current_gesture_value <= 0.6:
        # Scroll down
        print("Scrolling Down")
        pyautogui.scroll(-30)  # Adjust this value for scrolling down speed

# Function to detect hand movement direction
def detect_hand_movement(current_landmarks, previous_landmarks, threshold=0.005):
    if current_landmarks and previous_landmarks:
        current_x = current_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        current_y = current_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        previous_x = previous_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        previous_y = previous_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

        dx = current_x - previous_x
        dy = current_y - previous_y

        if abs(dx) > threshold or abs(dy) > threshold:
            if abs(dx) > abs(dy):
                if dx > threshold:
                    return "Hand is moving right"
                elif dx < -threshold:
                    return "Hand is moving left"
            else:
                if dy > threshold:
                    return "Hand is moving down"
                elif dy < -threshold:
                    return "Hand is moving up"
        else:
            return "No significant hand movement"
    return "No hand movement"

# Main function to process video stream
def main():
    cap = cv2.VideoCapture(0)
    previous_landmarks = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a more intuitive view
        image = cv2.flip(image, 1)

        # Convert the image to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe hands module
        results = hands.process(rgb_image)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the hand gesture value
                gesture_value = calculate_gesture_value(hand_landmarks)

                # Print the hand gesture value in the terminal with 2 decimal points
                if gesture_value is not None:
                    print(f"Hand Gesture Value: {gesture_value:.2f}")

                    # Scroll based on gesture value range
                    scroll_based_on_gesture(gesture_value)

                # Detect and print hand movement direction
                movement = detect_hand_movement(hand_landmarks, previous_landmarks)
                print(movement)

                # Update previous landmarks
                previous_landmarks = hand_landmarks

                # Render hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the processed image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit when Esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
