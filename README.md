# Description

Control and monitor Temas hardware over TCP/IP with a simple Python API.
Optimized for robotics, lab automation, and test systems requiring accurate motion and sensor feedback.


---
## Features

- TCP/IP socket communication to Temas devices
- Distance measurement (laser sensor)
- Precise pan and tilt positioning
- Real-time camera stream (visual and ToF)
- Easy-to-use object-oriented API
- Built-in threading support for camera streams
- Point cloud scanning
- Adjustable camera settings (exposure time, brightness, saturation, contrast, gain, lens position)
---


## 📦 Installation

Install the package easily using pip:

```bash
pip install rubu
```

## Example Code

```python
import cv2
from rubu import temas

# Connect to the device (via hostname or IP address)
device = temas.Connect(hostname="temas")
# Alternatively: device = temas.Connect(ip_address="192.168.4.4")

# Initialize control class
control = temas.Control()

# Measure distance (laser, in cm)
distance = control.distance()
print(f"Measured distance: {distance} cm")

# Move to a specific position (Pan, Tilt)
control.move_pos(60, 30)

# Initialize camera (Visual Port: 8081, ToF Port: 8084)
camera = temas.Camera(port=8081)
camera.start_thread()

# Adjust camera settings
camera.set_exposure_time(32000)
camera.set_brightness(50)
camera.set_contrast(3)

while True:
    try:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow('Visual Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera stream...")
            break
    except Exception as e:
        print(f"Error retrieving camera frame: {e}")

# Reset camera and control
control.move_home()
camera.stop_thread()
cv2.destroyAllWindows()
print("Program terminated.")
```
