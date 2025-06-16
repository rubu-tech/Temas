import cv2
import numpy as np
import mediapipe as mp
from rubu import temas
import time

# ---------------------------- Movement Parameters ----------------------------

# Pixel thresholds for movement sensitivity (x = horizontal, y = vertical)
TH_X_FINE   = 30
TH_X_COARSE = 70
TH_Y_FINE   = 30
TH_Y_COARSE = 70
MOVE_DELAY  = 0.001  # Minimum time between movement commands (in seconds)
_last_move  = 0.0    # Timestamp of last movement

# Move robot if the center position is off-center
def move_if_needed(dx: int, dy: int, control) -> None:
    global _last_move
    now = time.time()
    if now - _last_move < MOVE_DELAY:
        return

    moved = False
    # Horizontal movement
    if   dx >  TH_X_COARSE:  control.move_right();       moved = True
    elif dx >  TH_X_FINE:    control.move_right_fine();  moved = True
    elif dx < -TH_X_COARSE:  control.move_left();        moved = True
    elif dx < -TH_X_FINE:    control.move_left_fine();   moved = True

    # Vertical movement
    if   dy >  TH_Y_COARSE:  control.move_down();        moved = True
    elif dy >  TH_Y_FINE:    control.move_down_fine();   moved = True
    elif dy < -TH_Y_COARSE:  control.move_up();          moved = True
    elif dy < -TH_Y_FINE:    control.move_up_fine();     moved = True

    if moved:
        _last_move = now

# ---------------------------- Pose Tracker Class ----------------------------

class PoseTracker():
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def get_pose(self, frame, visible=False):
        # Convert to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.pose_tracker = self.pose.process(img_rgb)

        # Draw landmarks if requested
        if self.pose_tracker.pose_landmarks and visible:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self.pose_tracker.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame

    def get_position(self, frame, position=0, visible=False):
        # Get 2D positions of all body landmarks
        self.list_landmarks = []
        video_height = frame.shape[0]
        video_width = frame.shape[1]

        if self.pose_tracker.pose_landmarks:
            landmark_values = self.pose_tracker.pose_landmarks.landmark
            for i in range(len(landmark_values)):
                x = int(landmark_values[i].x * video_width)
                y = int(landmark_values[i].y * video_height)
                self.list_landmarks.append([i, x, y])

                # Draw a circle on a specific landmark if requested
                if visible and i == position:
                    cv2.circle(frame, (x, y), 30, (0, 0, 255), 2)

        return self.list_landmarks

# ---------------------------- Main Program ----------------------------

def main():
    temas.Connect(hostname="temas")
    control = temas.Control()
    control.move_home()
    camera = temas.Camera()
    camera.start_thread()
    tracker = PoseTracker()

    while True:
        if camera.kill:
            break

        try:
            frame = camera.get_frame()
            if frame is None:
                continue

            video_height = frame.shape[0]
            video_width = frame.shape[1]

            # Process the frame for pose estimation
            frame = tracker.get_pose(frame, visible=True)
            landmarks = tracker.get_position(frame, position=0, visible=False)

            if len(landmarks) > 12:
                # Get shoulder landmark positions (11 = left, 12 = right)
                x1 = landmarks[11][1]
                x2 = landmarks[12][1]
                y1 = landmarks[11][2]
                y2 = landmarks[12][2]

                # Calculate center point between shoulders
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Calculate delta from image center
                dx = x_center - (video_width // 2)
                dy = y_center - (video_height // 2)

                # Move robot if needed based on delta
                move_if_needed(dx, dy, control)

                # Visual feedback
                #print(f'x: {x_center}, y: {y_center}')

            cv2.imshow('Image', frame)

        except Exception as e:
            print("Error:", e)
            pass

        if cv2.waitKey(1) == ord('q'):
            break

    control.move_home()
    camera.stop_thread()

if __name__ == "__main__":
    main()
