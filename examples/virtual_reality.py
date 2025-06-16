# Import required Kivy modules
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

# Import gyroscope access via Plyer (cross-platform)
from plyer import gyroscope

# Import temas communication module
from rubu import temas

# For running background threads
from threading import Thread

# For streaming video over HTTP
import urllib.request

# OpenCV and NumPy for image processing
import cv2
import numpy as np

# For converting radians to degrees
from math import pi

# Delay execution at the start
import time
time.sleep(10)

# Class to handle MJPEG video streaming inside a Kivy Image widget
class MJPEGStream(Image):
    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.url = url                # MJPEG stream URL
        self.frame = None            # Stores the latest decoded frame
        self._thread = Thread(target=self._update_stream, daemon=True)
        self._thread.start()         # Start background thread to fetch frames

    def _update_stream(self):
        """Continuously fetch and decode frames from MJPEG stream."""
        try:
            stream = urllib.request.urlopen(self.url)
            bytes_data = b''
            while True:
                bytes_data += stream.read(1024)  # Read stream in chunks
                a = bytes_data.find(b'\xff\xd8')  # Start of JPEG
                b = bytes_data.find(b'\xff\xd9')  # End of JPEG
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]        # Extract JPEG frame
                    bytes_data = bytes_data[b+2:]  # Remove used bytes
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (640, 360))    # Resize frame
                        img = cv2.flip(img, 1)               # Mirror horizontally
                        self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        except Exception as e:
            print("Stream error:", e)

    def get_texture(self):
        """Convert the current frame into a Kivy Texture."""
        if self.frame is not None:
            h, w, _ = self.frame.shape
            texture = Texture.create(size=(w, h), colorfmt='rgb')
            texture.flip_vertical()
            texture.wrap = 'clamp_to_edge'
            texture.mag_filter = 'nearest'
            texture.blit_buffer(self.frame.flatten(), colorfmt='rgb', bufferfmt='ubyte')
            return texture
        return None

# Main application class
class VRGyroApp(App):
    def build(self):
        # PUT HERE YOUR IP ADDRESS
        self.ip = '192.168.0.4'
        self.winkel = [0, 0, 0]       # Stores rotation angles (x, y, z)
        self.dt = 0.033               # Time step (~30 updates per second)
        self.control = None
        self.initialisiert = False
        self.last_pos = [None, None] # Last sent position to avoid duplicates

        # Initialize robot connection shortly after start
        Clock.schedule_once(self.init_connection, 0.5)

        # Try to enable gyroscope
        try:
            gyroscope.enable()
        except NotImplementedError:
            print("Gyroscope not available")

        # Schedule reading gyroscope data repeatedly
        Clock.schedule_interval(self.read_gyro, self.dt)

        # Create horizontal layout for side-by-side images (VR-like effect)
        layout = BoxLayout(orientation='horizontal')
        stream_url = 'http://'+self.ip+':8081/stream.mjpg'
        mjpeg = MJPEGStream(stream_url)

        # Create two image views for left and right eye
        left_image = Image()
        right_image = Image()
        layout.add_widget(left_image)
        layout.add_widget(right_image)

        # Function to update textures from the MJPEG stream
        def update_shared_texture(dt):
            tex = mjpeg.get_texture()
            if tex:
                left_image.texture = tex
                right_image.texture = tex

        # Update both images at ~30 FPS
        Clock.schedule_interval(update_shared_texture, 1.0 / 30)

        # Add the MJPEG stream widget (hidden, only for background decoding)
        mjpeg.opacity = 0
        mjpeg.size_hint = (0, 0)
        layout.add_widget(mjpeg)

        return layout

    def init_connection(self, dt):
        # Try to connect to the remote robot system
        try:
            temas.Connect(ip_address=self.ip)  # NOTE: Missing actual IP assignment!
            self.control = temas.Control()
            self.control.move_pos(0, 0)  # Move to neutral position
            self.initialisiert = True
            print("Connection established")
        except Exception as e:
            print(f"Connection error: {e}")

    def read_gyro(self, dt):
        # Read and process gyroscope data to control position
        if not self.initialisiert:
            return

        data = gyroscope.rotation
        if data and all(v is not None for v in data):
            for i in range(3):
                # Convert from radians to degrees and integrate angle
                self.winkel[i] += data[i] * self.dt * (180 / pi)

            x, y, z = map(int, self.winkel)
            y1 = -y  # Invert Y-axis for control logic

            # Only send position updates if within defined limits
            if (-60 <= x <= 60) and (-30 <= y1 <= 90):
                if [x, y1] != self.last_pos:
                    try:
                        self.control.move_pos(x, y1)  # Send movement command
                        self.last_pos = [x, y1]       # Save last sent position
                    except Exception as e:
                        print(f"Error in move_pos: {e}")
        else:
            print("Gyro data not available")


# Entry point to run the app
if __name__ == '__main__':
    VRGyroApp().run()
