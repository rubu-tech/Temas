# -*- coding: utf-8 -*-
"""
Temas Face Tracker
------------------------
 • Detects the face using MediaPipe FaceMesh from the Temas RGB camera
 • Rotates the robot head to center the face in the frame
 • Runs fully locally – no web UI, no <model-viewer>, no cloud

Author: Muhammed, May 2025
"""
from __future__ import annotations
import time, logging, cv2, numpy as np, mediapipe as mp
from rubu import temas
from collections import deque

# ---------------------------------------------------------------------------- Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")

# ---------------------------------------------------------------------------- Connect to Temas Robot
IP = temas.Connect(hostname="temas").ip or "192.168.0.4"
CTRL = temas.Control(ip_address=IP)
logging.info("Connected to Temas @ %s", IP)

# ---------------------------------------------------------------------------- Movement thresholds and control
TH_X_FINE   = 30     # Fine threshold in X
TH_X_COARSE = 60     # Coarse threshold in X
TH_Y_FINE   = 30     # Fine threshold in Y
TH_Y_COARSE = 60     # Coarse threshold in Y
MOVE_DELAY  = 0.001  # Minimum delay between movement commands
_last_move  = 0.0

def move_if_needed(delta_x: int, delta_y: int) -> None:
    """Send robot movement commands based on face position offset."""
    global _last_move
    now = time.time()
    if now - _last_move < MOVE_DELAY:
        return

    moved = False
    # Horizontal movement
    if   delta_x >  TH_X_COARSE:  CTRL.move_right();       moved = True
    elif delta_x >  TH_X_FINE:    CTRL.move_right_fine();  moved = True
    elif delta_x < -TH_X_COARSE:  CTRL.move_left();        moved = True
    elif delta_x < -TH_X_FINE:    CTRL.move_left_fine();   moved = True

    # Vertical movement
    if   delta_y >  TH_Y_COARSE:  CTRL.move_down();        moved = True
    elif delta_y >  TH_Y_FINE:    CTRL.move_down_fine();   moved = True
    elif delta_y < -TH_Y_COARSE:  CTRL.move_up();          moved = True
    elif delta_y < -TH_Y_FINE:    CTRL.move_up_fine();     moved = True

    if moved:
        _last_move = now

# ---------------------------------------------------------------------------- MediaPipe FaceMesh Setup
mp_draw = mp.solutions.drawing_utils
mp_mesh = mp.solutions.face_mesh
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# ---------------------------------------------------------------------------- Camera setup
camera_vis = temas.Camera(port=8081)   # RGB camera
camera_tof = temas.Camera(port=8084)   # Time-of-Flight (depth) camera
camera_vis.start_thread()
camera_tof.start_thread()

# ---------------------------------------------------------------------------- Color map for ToF visualization
color_map = np.zeros((256, 1, 3), dtype=np.uint8)
color_map[:128, 0, 0] = np.linspace(255, 0, 128)   # Blue → Cyan → Green
color_map[:128, 0, 1] = np.linspace(0, 255, 128)
color_map[128:, 0, 1] = np.linspace(255, 0, 128)   # Green → Yellow → Red
color_map[128:, 0, 2] = np.linspace(0, 255, 128)

# ---------------------------------------------------------------------------- Tracking buffers
center_history = deque(maxlen=5)       # Last 5 face center positions
distance_history = deque(maxlen=5)     # Last 5 distance values (cm)

# ---------------------------------------------------------------------------- Main loop
with mp_mesh.FaceMesh(max_num_faces=1,
                      refine_landmarks=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as mesh:

    while True:
        img_vis = camera_vis.get_frame()
        if img_vis is not None:
            h, w = img_vis.shape[:2]
            rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                # --- Compute face center
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                cx, cy = int(sum(xs) / len(xs) * w), int(sum(ys) / len(ys) * h)

                # Add to history buffer
                center_history.append((cx, cy))

                # Draw face landmarks
                mp_draw.draw_landmarks(img_vis, res.multi_face_landmarks[0],
                                       mp_mesh.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=draw_spec,
                                       connection_drawing_spec=draw_spec)
                
                # Calculate smoothed average center if buffer is full
                if len(center_history) == center_history.maxlen:
                    avg_cx = int(sum(p[0] for p in center_history) / len(center_history))
                    avg_cy = int(sum(p[1] for p in center_history) / len(center_history))

                    # Draw visual indicator at smoothed center
                    cv2.circle(img_vis, (avg_cx, avg_cy), 6, (255, 0, 0), -1)  # Outer white circle
                    cv2.circle(img_vis, (avg_cx, avg_cy), 3, (0, 0, 255), -1)  # Inner red dot

                    # Calculate offset from image center and move robot
                    dx = avg_cx - w // 2
                    dy = avg_cy - h // 2
                    move_if_needed(dx, dy)

        img_tof = camera_tof.get_frame()
        if img_tof is not None and len(center_history) == center_history.maxlen:
            # --- Crop and resize ToF image
            img_tof_crop = img_tof[0:180, 30:210]
            img_tof = cv2.resize(img_tof_crop, (720, 720))

            # Convert to grayscale and apply color map
            gray = cv2.cvtColor(img_tof, cv2.COLOR_BGR2GRAY)
            norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_mapped_frame = cv2.applyColorMap(norm_gray, color_map)

            # Prepare inverted grayscale frame for distance lookup
            img_tof_distance = cv2.bitwise_not(img_tof_crop[:, :, 0])
            img_tof_distance = cv2.resize(img_tof_distance, (720, 720))

            avg_cx, avg_cy = center_history[-1]
            cx_tof = avg_cx - 360  # Adjust to match ToF frame alignment
            cy_tof = avg_cy

            # --- Compute average depth in 10x10 patch around center
            max_distance_cm = 3.9 * 100  # 3.9m in cm
            z_patch = img_tof_distance[cy_tof-5:cy_tof+5, cx_tof-5:cx_tof+5]
            z_cm = z_patch * (max_distance_cm / 255.0)
            raw_distance = np.mean(z_cm)

            # Update smoothed distance history
            distance_history.append(raw_distance)

            # Use smoothed distance if available
            if len(distance_history) == distance_history.maxlen:
                distance = np.round(np.mean(distance_history))

                # Display distance and draw indicator
                cv2.putText(img_vis, f"ToF Distance [cm]: {distance}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

                # Draw corresponding dot in ToF frame
                cv2.circle(color_mapped_frame, (cx_tof, cy_tof), 6, (255, 255, 255), -1)
                cv2.circle(color_mapped_frame, (cx_tof, cy_tof), 3, (0, 0, 255), -1)

                #ToF Kamera
                # if color_mapped_frame is not None:
                #     cv2.imshow("Temas ToF Camera", color_mapped_frame)

        # Display final image
        if img_vis is not None:
            cv2.imshow("Temas Visual Camera", img_vis)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

# ---------------------------------------------------------------------------- Cleanup
camera_vis.stop_thread()
camera_tof.stop_thread()
cv2.destroyAllWindows()
