# vision.py
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        # This is the standard entry point
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.cap = cv2.VideoCapture(0)

    def get_hand_positions(self):
        success, frame = self.cap.read()
        if not success:
            return None, []
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                hands_data.append({
                    'label': label,
                    'landmarks': landmarks
                })
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, hands_data

    def get_pinch_distance(self, hand_landmarks):
        """Calculates normalized distance between Thumb (4) and MIDDLE Finger (12)."""
        # Reference scale (Wrist to Middle Base)
        wrist = hand_landmarks[0]
        middle_base = hand_landmarks[9]
        hand_size = ((wrist[0] - middle_base[0])**2 + (wrist[1] - middle_base[1])**2)**0.5
        
        # Pinch check: Thumb (4) to Middle Tip (12)
        thumb_tip = hand_landmarks[4]
        middle_tip = hand_landmarks[12] # Changed from 8 to 12
        raw_dist = ((thumb_tip[0] - middle_tip[0])**2 + (thumb_tip[1] - middle_tip[1])**2)**0.5
        
        return raw_dist / hand_size

    def release(self):
        self.cap.release()