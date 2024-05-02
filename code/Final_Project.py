# Importing all the libraries nessesary for Running the code
import streamlit as st
import cv2
import numpy as np
from pphumanseg import PPHumanSeg
import time

# Defining gstreamer pipeline which will be used to read frames from rasberrypie camera on Jetson nano
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
        "width=(int){}, height=(int){}, framerate=(fraction){}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        .format(
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Function to read faces from Har-Cascade "frontal_face_xml" (Code was provided with jetson nano)
def Face_Cascade(frame, switch):
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml") #E:\\EmbJetson\\haarcascade_frontalface_default.xml
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if switch:
        for (x,y,w,h) in faces:
            roi = frame[y:y + h, x:x + w]
        return roi
    else:
        return faces

# Function to read images used in this project (Mask images are PNG format supporting alpha channel)
def read_images():
    images = {}
    images['batman_mask'] = cv2.imread('batman_1.png', -1)
    images['beach_background'] = cv2.imread('beach.jpg', -1)
    images['iron_man_mask'] = cv2.imread('iron_man_2.png', -1)
    return images

# Initiating model PPHuman Segmentation
def model_instantiate():
    backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA]] # Initiating Flags for GPU and CPU initiation

    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1] # Setting GPU and CPU running flags, These flags will determine wether the code will run on GPU or CPU
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath="human_segmentation_pphumanseg_2023mar.onnx", backendId=backend_id, targetId=target_id)
    return model

# Below is a function used to calculate Human Segmentation Mask
def Mask(frame):
    frame = frame.copy()
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting BGR image read from Gstreamer to RGb to pass to PPHuman Model
    _image = cv2.resize(image, dsize=(192, 192)) # Sizing the image down to get inference from model
    result = model.infer(_image) # Calling model for mask inference
    result = cv2.resize(result[0, :, :], dsize=(w, h)) # Resize the inference image back to frame size
    _, threshold = cv2.threshold(result,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Threshold the image for Mask detail
    binary_mask = threshold.astype(np.uint8) # Reduce dimentonality of image for the purpose of fast processing
    mask = cv2.merge([binary_mask, binary_mask, binary_mask]) # Make The mask 3 channel image for overlaying purpose
    return mask

# Below is a function which will be used for Radial Distortion
def radial_distortion(image, strength=0.00045, center=None, radius = 60, speed = 1.5):
    h, w = image.shape[:2]
    time = cv2.getTickCount() / cv2.getTickFrequency() # Enables the rotaional dynamic center over face
    if center is None:
        center_x, center_y = w / 2 + radius * np.cos(speed * time), h / 2 + radius * np.sin(speed * time)
    else:
        center_x, center_y = center

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2) # Finding randius using pythagoras formulae and dynamic dynamic distance from center
    factor = 1 + strength * r**2
    factor[r > radius] = 1

    map_x = (dx / factor + center_x).astype(np.float32)
    map_y = (dy / factor + center_y).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR) # returning Remaped face here to main program

# Below is the function which will be used for Swirl distortion
def swirl_distortion(image, center=None, strength=2.0, radius=60):
    h, w = image.shape[:2]
    if center is None:
        center_x, center_y = w / 2, h / 2
    else:
        center_x, center_y = center

    x, y = np.meshgrid(np.arange(w), np.arange(h))  #np.indices((h,w), dtype=np.float32)
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx) # finding theta as distortion coefficient
    mask = r <= radius
    swirl_strength = np.clip(strength * (radius - r) / radius, 0, strength) # clipping the range of distortion to limit between face border
    theta += swirl_strength * mask

    map_x = center_x + r * np.cos(theta)
    map_y = center_y + r * np.sin(theta)

    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

# Below is a function which is used to overlay face mask over face
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

# Function that initializes the Maks and calls the Overlay function
def face_mask(x, y, w, h, frame, mask_switch):
    tmpx, tmpy = x - 15, y - 35 # Custom Size defined for the purpose of scaling mask to entire head.
    if h > 0 and w > 0:
        roi_color = frame[tmpy:int(tmpy * 1.5) + h, tmpx:int(tmpx * 1.5) + w]
        if mask_switch:
            iron = cv2.resize(images['iron_man_mask'], (w + 20, h + 35))
            transparentOverlay(roi_color, iron)
        else:
            batu = cv2.resize(images['batman_mask'], (w + 20, h + 35))
            transparentOverlay(roi_color, batu)

# Below is a Function used to Blur the background as well as replace the background.
def Blur_Replace(frame, Blur, img_bcg):
    frame = frame.copy()
    mask = Mask(frame) # Calling Function Mask for the purpose of finidn mask from human Segmentation model
    blur_input = cv2.GaussianBlur(frame,(91,91),cv2.BORDER_DEFAULT) # Blur the frame entirely for purpose of Backgroud blur
    img_bcg = cv2.resize(img_bcg,(frame.shape[1], frame.shape[0])) # Resize new background image to Frame shape

    if Blur:
        # this is blurring
        result_blur_frame = cv2.bitwise_and(frame,mask) # Take features common to frame and mask from Frame
        RS2_blur = cv2.bitwise_and(blur_input, cv2.bitwise_not(mask)) # Take all features except the mask from blurred Frame
        result_frame_blur = cv2.add(result_blur_frame, RS2_blur) # Overlay the above extracted image on top of each other for Background blur effect
        return result_frame_blur
    else:
        # This is bcg rplacement
        result_frame = cv2.bitwise_and(frame,mask)
        RS2 = cv2.bitwise_and(img_bcg, cv2.bitwise_not(mask)) # Take all features except the mask from Background image
        result_frame = cv2.add(result_frame, RS2) # Overlay the above extracted image on top of each other for Background Replacement effect
        return result_frame

# Below is creative Filter Function Used
def frame_carton(frame):
    frame = frame.copy()
    color = cv2.bilateralFilter(frame, 9, 9, 7) # Used Bilateral Filter For Smoothing effect maintaining the edges
    blur = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 15) # Applied median filter over bilateral to remove Impulse Noise/ Salt-and-pepper Noise(Sharp features Noise).
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2) # Applying Adaptive threshold for enhancing edges
    frame_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # converting Gray to RGB to show true colour
    cartoon = cv2.bitwise_and(color, frame_edge)
    blurred = cv2.bilateralFilter(cartoon, d=7, sigmaColor=200,sigmaSpace=200) # Reapplying bilateral filter to smooth further while focusing to smooth colour more in order to achieve Cortoon effect
    return blurred

# The below function Assembels the effect of a Cartoonized image
def cartoonized(frame):
    frame = frame.copy() # create copy of frame
    frame_cart = frame_carton(frame) # create cartoonized frame
    mask = Mask(frame) # Call mask 

    result_frame = cv2.bitwise_and(frame_cart,mask) # Overlay Cartoonized Human or Normal frame
    RS2 = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    result_frame = cv2.add(RS2, result_frame)
    return result_frame

# The below function is used to display frames over Streamlit
def show_stream(frame_placeholder, frame, title = "Karan-Shanu - Original Stream"):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the frame on Streamlit instead of using cv2.imshow
    frame_placeholder.image(frame, caption=title, channels="RGB")
    # Sleep for a short period to give the illusion of a video stream
    time.sleep(0.033)  # Around 30 frames per second

def main():
    st.sidebar.title("Functionality")
    functionality = st.sidebar.selectbox("Choose functionality:", 
                                         ["Original", "Face Distortion Radial", "Face Distortion Swirl", "Face Masking IronMan", "Face Masking BatMan", "Background Blur", "Background Replace", "Creative Filter-Cartoonization"])
    
    gstreamer_pipe = gstreamer_pipeline() # Initializing Gstreamer object
    video_capture = cv2.VideoCapture(gstreamer_pipe, cv2.CAP_GSTREAMER) #gstreamer_pipe, cv2.CAP_GSTREAMER
    frame_placeholder = st.empty() # Creating Empty Placehoder for Image display

    if not video_capture.isOpened():
        print("Unable to open camera")
        exit()
    while True:
        ret, frame = video_capture.read() # read frames
        if not ret:
            break

        # Sidebar parameters for each functionality
        if functionality == "Original":
            show_stream(frame_placeholder, frame, title= "Karan-Shanu - Original Stream")
        elif functionality == "Face Distortion Radial":
            frame1 = frame.copy()
            faces = Face_Cascade(frame1, False)
            for (x,y,w,h) in faces:
                roi_color = frame1[y:y + h, x:x + w]
                roi_color1 = radial_distortion(roi_color)
                frame1[y:y + h, x:x + w] = roi_color1
            show_stream(frame_placeholder, frame1, title= "Karan-Shanu RadialDistortion")
        elif functionality == "Face Distortion Swirl":
            frame1 = frame.copy()
            faces = Face_Cascade(frame1, False)
            for (x,y,w,h) in faces:
                roi_color = frame1[y:y + h, x:x + w]
                roi_color1 = swirl_distortion(roi_color)
                frame1[y:y + h, x:x + w] = roi_color1
            show_stream(frame_placeholder, frame1, title= "Karan-Shanu SwirlDistortion")
        elif functionality == "Face Masking IronMan":
            faces = Face_Cascade(frame, False)
            for (x, y, w, h) in faces:
                face_mask(x, y, w, h, frame, True)
            show_stream(frame_placeholder, frame, title= "FaceMasking IronMan")
        elif functionality == "Face Masking BatMan":
            faces = Face_Cascade(frame, False)
            for (x, y, w, h) in faces:
                face_mask(x, y, w, h, frame, False)
            show_stream(frame_placeholder, frame, title= "Karan-Shanu FaceMasking BatMan")
        elif functionality == "Background Blur":
            frame = Blur_Replace(frame, True, images['beach_background'])
            show_stream(frame_placeholder, frame, title= "Karan-Shanu Background Blur")
        elif functionality == "Background Replace":
            frame = Blur_Replace(frame, False, images['beach_background'])
            show_stream(frame_placeholder, frame, title= "Karan-Shanu Background Replace")
        elif functionality == "Creative Filter-Cartoonization":
            frame = cartoonized(frame)
            show_stream(frame_placeholder, frame, title= "Karan-Shanu Cartoonization")
         
        keyCode = cv2.waitKey(10) & 0xFF # Ask keyboard interrupt
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord('q'):
            break # Break if Key == q
        
    video_capture.release() # Destroy all frames
    cv2.destroyAllWindows() # Cloose all windows
    return frame # return Frame

# Run the main function
if __name__ == "__main__":
    images = read_images()
    model = model_instantiate()
    main()