import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # Calculate the region of interest (ROI) in the source image
    roi = src[x:min(x+h, rows), y:min(y+w, cols)]

    # Ensure that the shapes are compatible for broadcasting
    overlay = overlay[:roi.shape[0], :roi.shape[1]]

    # Create a mask using the alpha channel of the overlay
    mask = overlay[:, :, 3] / 255.0

    # Blend the images using the mask
    src[x:min(x+h, rows), y:min(y+w, cols)][:, :, :3] = (
        mask[:, :, None] * overlay[:, :, :3] +
        (1.0 - mask[:, :, None]) * roi
    )


def face_blur(x, y, w, h, frame):
    tmpx, tmpy = x - 15, y - 35
    if h > 0 and w > 0:
        roi_color = frame[tmpy:int(tmpy * 1.5) + h, tmpx:int(tmpx * 1.5) + w]
        # batu = cv2.resize(batman, (w + 20, h + 35))
        # transparentOverlay(roi_color, batu)
        iron = cv2.resize(ironman, (w + 20, h + 35))
        transparentOverlay(roi_color, iron)

if __name__ == "__main__":
    window_title = "faceMask"
    face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    
    # Use the gstreamer pipeline for Jetson Nano
    gstreamer_pipe = gstreamer_pipeline()
    video_capture = cv2.VideoCapture(gstreamer_pipe, cv2.CAP_GSTREAMER)
    
    if not video_capture.isOpened():
        print("Unable to open camera")
        exit()
    
    batman = cv2.imread('batman_1.png', -1)
    ironman = cv2.imread('iron_man_2.png', -1)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_blur(x, y, w, h, frame)

        cv2.imshow(window_title, frame)
        
        keyCode = cv2.waitKey(10) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord('q'):
            break

    cv2.destroyAllWindows()
