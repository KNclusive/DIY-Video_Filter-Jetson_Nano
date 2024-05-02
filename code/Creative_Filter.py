import cv2
import numpy as np
from pphumanseg import PPHumanSeg

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA]
]

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
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

def frame_carton():
    color = cv2.bilateralFilter(frame, 9, 9, 7)
    blur = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 15)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    frame_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, frame_edge)
    blurred = cv2.bilateralFilter(cartoon, d=7, sigmaColor=200,sigmaSpace=200)
    return blurred

def Mask():
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _image = cv2.resize(image, dsize=(192, 192))
    result = model.infer(_image)
    result = cv2.resize(result[0, :, :], dsize=(w, h))
    _, threshold = cv2.threshold(result,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_mask = threshold.astype(np.uint8)
    mask = cv2.merge([binary_mask, binary_mask, binary_mask])
    return mask

if __name__ == "__main__":
    window_title = "faceblur"
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath="human_segmentation_pphumanseg_2023mar.onnx", backendId=backend_id, targetId=target_id)
    # Use the gstreamer pipeline for Jetson Nano
    gstreamer_pipe = gstreamer_pipeline()
    video_capture = cv2.VideoCapture(gstreamer_pipe, cv2.CAP_GSTREAMER)
    
    if not video_capture.isOpened():
        print("Unable to open camera")
        exit()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_cart = frame_carton()
        mask = Mask()

        result_frame = cv2.bitwise_and(frame_cart,mask)
        RS2 = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        result_frame = cv2.add(RS2, result_frame)

        cv2.imshow(window_title, result_frame)
        
        keyCode = cv2.waitKey(10) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()