import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=960,
    capture_height=540,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
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

def radial_distortion(image, strength=1.0, center=None, radius = 60, speed = 1.5):
    h, w = image.shape[:2]
    time = cv2.getTickCount() / cv2.getTickFrequency()
    if center is None:
        center_x, center_y = w / 2 + radius * np.cos(speed * time), h / 2 + radius * np.sin(speed * time)
    else:
        center_x, center_y = center

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    factor = 1 + strength * r**2
    factor[r > radius] = 1

    map_x = (dx / factor + center_x).astype(np.float32)
    map_y = (dy / factor + center_y).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

def swirl_distortion(image, center=None, strength=1.0, radius=100):
    h, w = image.shape[:2]
    if center is None:
        center_x, center_y = w / 2, h / 2
    else:
        center_x, center_y = center

    x, y = np.meshgrid(np.arange(w), np.arange(h))  #np.indices((h,w), dtype=np.float32)
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    mask = r <= radius
    swirl_strength = np.clip(strength * (radius - r) / radius, 0, strength)
    theta += swirl_strength * mask

    map_x = center_x + r * np.cos(theta)
    map_y = center_y + r * np.sin(theta)

    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

# def wave_distortion(image, wave_length=20, amplitude=5, axis='x'):
#     h, w = image.shape[:2]
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
    
#     if axis == 'x':
#         map_y = y + amplitude * np.sin(2 * np.pi * x / wave_length)
#         map_x = x
#     else:
#         map_x = x + amplitude * np.sin(2 * np.pi * y / wave_length)
#         map_y = y

#     return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    window_title = "faceDistort"
    # Use the gstreamer pipeline for Jetson Nano
    gstreamer_pipe = gstreamer_pipeline()
    video_capture = cv2.VideoCapture(gstreamer_pipe, cv2.CAP_GSTREAMER) #gstreamer_pipe, cv2.CAP_GSTREAMER
    
    if not video_capture.isOpened():
        print("Unable to open camera")
        exit()

    face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            # tmpx, tmpy = x - w//2, y - h//2
            roi = frame[y:y + h, x:x + w]
            radu = np.random.randint(20,90)
            print(radu)
            radial_distort = radial_distortion(roi, strength=0.00045, radius = radu)
            # swirl_distort = swirl_distortion(roi, strength=1.999, radius=60)
            # wave_distort = wave_distortion(roi, wave_length=30, amplitude=5, axis='x')
            frame[y:y + h, x:x + w] = radial_distort

        cv2.imshow(window_title, frame)
        
        keyCode = cv2.waitKey(10) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()