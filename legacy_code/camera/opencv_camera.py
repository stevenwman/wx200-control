import cv2
import time

class OpenCVCamera:
    """OpenCV-based camera capture fallback."""
    
    def __init__(self, device=0, width=1920, height=1080, fps=30):
        """
        Initialize OpenCV camera.
        
        Args:
            device: Camera device index or path (e.g., 0, 1, '/dev/video0')
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frame rate
        """
        self.device = device
        # Handle string device paths if they are just numbers (e.g. "0")
        if isinstance(self.device, str):
            if self.device.isdigit():
                self.device = int(self.device)
            elif self.device.startswith('/dev/video'):
                try:
                    self.device = int(self.device.replace('/dev/video', ''))
                except ValueError:
                    pass # Keep as string if it's not a simple /dev/videoN
        
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def start(self):
        """Start the camera."""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device: {self.device}")
            
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Warmup
        time.sleep(1.0)
        
    def read(self):
        """
        Read a frame.
        
        Returns:
            (success, frame): Tuple of (bool, numpy array or None)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def release(self):
        """Stop and cleanup."""
        if self.cap:
            self.cap.release()
            self.cap = None
