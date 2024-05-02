import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
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


def face_detect():
    window_title = "Face Detect"
    beach = cv2.imread('beach.jpg')
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret, frame = video_capture.read()
                beach_resized = cv2.resize(beach,(frame.shape[1],frame.shape[0]))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                inverted_gray = cv2.bitwise_not(gray)
                # blurred_frame = cv2.GaussianBlur(inverted_gray, (11, 11), 0) # One Viable option is to blur inverted image and pass to mask.
                # edges = cv2.Canny(gray, 100, 200)

                # mask = cv2.inRange(inverted_gray, 140, 255)
                _, threshold = cv2.threshold(inverted_gray,140,255,cv2.THRESH_OTSU)

                # mask_with_blur = cv2.medianBlur(threshold, 11) # Second Viable option

                # kernel = np.ones((9, 9), np.uint8) # Best Viable option Till now Start
                # mask_erode = cv2.erode(mask, kernel, iterations=5)
                # mask_dilate = cv2.dilate(mask_erode, kernel, iterations=4)
                # mask = mask_dilate # Best Viable option Till now end
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Try TREE as well
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # cv2.drawContours(frame, [largest_contour],-1, (0, 255, 0), 2)    # Working Important

                    contour_mask = cv2.drawContours( np.zeros_like(gray), [largest_contour], -1, (255), thickness=cv2.FILLED)

                    # For Background Blur
                    # blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0) # For Blurring Background
                    # frame_outside_contour = cv2.bitwise_and(blurred_frame, blurred_frame, mask=~contour_mask)
                    # result_frame = cv2.bitwise_and(frame, frame, mask=contour_mask)
                    # result_frame = cv2.add(result_frame, frame_outside_contour)

                    # For Background Replacement
                    frame_outside_contour = cv2.bitwise_and(beach_resized, beach_resized, mask=~contour_mask)
                    result_frame = cv2.bitwise_and(frame, frame, mask=contour_mask)
                    result_frame = cv2.add(result_frame, frame_outside_contour)

                    concat_image = np.concatenate((frame, result_frame), axis=1) #cv2.cvtColor(contour_mask, cv2.COLOR_BGR2RGB)

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, concat_image)
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


if __name__ == "__main__":
    face_detect()
