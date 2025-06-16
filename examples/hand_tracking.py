import cv2
import mediapipe as mp
import time
from rubu import temas

# ---------------------------- Control Parameters ----------------------------

# Movement thresholds for fine and coarse adjustments
TH_X_FINE   = 30
TH_X_COARSE = 60
TH_Y_FINE   = 30
TH_Y_COARSE = 60
MOVE_DELAY  = 0.001  # Minimum time (in seconds) between movement commands
_last_move  = 0.0    # Timestamp of the last movement

# Function to trigger robot movement based on x/y deltas from center
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

# ---------------------------- Hand Tracker Class ----------------------------

class HandTracker():
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()

    def get_hand(self, frame, visible=False):
        # Convert image to RGB for Mediapipe processing
        self.video_height = frame.shape[0]
        self.video_width = frame.shape[1]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.hand_tracker = self.hands.process(img_rgb)

        # Draw landmarks if requested
        if self.hand_tracker.multi_hand_landmarks:
            for i in self.hand_tracker.multi_hand_landmarks:
                if visible:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, i, mp.solutions.hands.HAND_CONNECTIONS
                    )
        return frame

    def get_position(self):
        # Returns all landmark coordinates of the detected hand
        self.list_landmarks = []
        if self.hand_tracker.multi_hand_landmarks:
            landmark_values = self.hand_tracker.multi_hand_landmarks[0].landmark
            for i in range(len(landmark_values)):
                x = int(landmark_values[i].x * self.video_width)
                y = int(landmark_values[i].y * self.video_height)
                self.list_landmarks.append([i, x, y])
        return self.list_landmarks

# ---------------------------- Main Function ----------------------------

def main():
    temas.Connect(hostname="temas")
    control = temas.Control()
    control.move_home()
    camera = temas.Camera()
    camera.start_thread()
    tracker = HandTracker()

    ids = [8, 12, 16, 20]  # IDs of fingertips (index to pinky)
    n = 0
    image_capture = False
    video_recording = False

    # Set up video writer for recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))

    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                continue

            video_height = frame.shape[0]
            video_width = frame.shape[1]

            # Detect hand and landmarks
            frame = tracker.get_hand(frame, visible=False)
            list_landmarks = tracker.get_position()

            if len(list_landmarks) != 0:
                fingers = []

                # Determine which fingers are raised
                for i in range(4):
                    if list_landmarks[ids[i]][2] < list_landmarks[ids[i] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                print(fingers)

                # Only index finger raised → move robot
                if fingers == [1, 0, 0, 0]:
                    x_pos = list_landmarks[8][1]
                    y_pos = list_landmarks[8][2]
                    dx = x_pos - video_width // 2
                    dy = y_pos - video_height // 2
                    move_if_needed(dx, dy, control)

                    # Visual feedback for active tracking
                    cv2.rectangle(frame, (x_pos - 25, y_pos - 25), (x_pos + 25, y_pos + 25), (0, 255, 0), 1)

                # Index + middle finger up → take photo
                if fingers == [1, 1, 0, 0]:
                    image_capture = True

                # Middle + ring + pinky finger up → start recording
                if fingers == [0, 1, 1, 1]:
                    video_recording = True

                # Only pinky up → stop recording
                if fingers == [0, 0, 0, 1]:
                    video_recording = False

            # Save image after short delay
            if image_capture:
                n += 1
                if n == 30:
                    cv2.imwrite("img.jpg", frame)
                    image_capture = False
                    n = 0

            # Write frame to video if recording
            if video_recording:
                out.write(frame)

            # Show the processed image
            cv2.imshow("Image", frame)

        except Exception as e:
            print("Error:", e)
            pass

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    out.release()
    camera.stop_thread()

if __name__ == "__main__":
    main()
