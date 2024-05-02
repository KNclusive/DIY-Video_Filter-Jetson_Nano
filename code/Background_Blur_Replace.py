import numpy as np
import cv2
from pphumanseg import PPHumanSeg

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA]
]

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
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


if __name__ == '__main__':
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath="human_segmentation_pphumanseg_2023mar.onnx", backendId=backend_id, targetId=target_id)
    window_title = "output"
    video_capture = cv2.VideoCapture(0) # gstreamer_pipeline(flip_method =2), cv2.CAP_GSTREAMER

    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            img_bcg = cv2.imread("beach.jpg")
            while True:
                ret, frame = video_capture.read()
                h, w, _ = frame.shape
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _image = cv2.resize(image, dsize=(192, 192))
                result = model.infer(_image)
                result = cv2.resize(result[0, :, :], dsize=(w, h))
                _, threshold = cv2.threshold(result,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                blur_input = cv2.GaussianBlur(frame,(91,91),cv2.BORDER_DEFAULT)

                binary_mask = threshold.astype(np.uint8)
                mask = cv2.merge([binary_mask, binary_mask, binary_mask])
                img_bcg = cv2.resize(img_bcg,(w, h))

                 # This is bcg rplacement
                result_frame = cv2.bitwise_and(frame,mask)
                RS2 = cv2.bitwise_and(img_bcg, cv2.bitwise_not(mask))
                result_frame = cv2.add(result_frame, RS2)

                # this is blurring
                result_blur_frame = cv2.bitwise_and(frame,mask)

                # Replace the background using the mask
                RS2_blur = cv2.bitwise_and(blur_input, cv2.bitwise_not(mask))
                result_frame_blur = cv2.add(result_blur_frame, RS2_blur)
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow("Segmentation with Background Blur", result_frame_blur)
                    cv2.imshow("Segmentation with Background Replacement", result_frame)
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")