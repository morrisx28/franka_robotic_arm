from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np # Import NumPy library
 
# Initialize the camera
camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (640, 480)
 
# Set the number of frames per second
camera.framerate = 30
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
# Initialize the first frame of the video stream
first_frame = None
 
# Create kernel for morphological operation. You can tweak
# the dimensions of the kernel.
# e.g. instead of 20, 20, you can try 30, 30
kernel = np.ones((20,20),np.uint8)
 
# Centimeter to pixel conversion factor
# I measured 36.0 cm across the width of the field of view of the camera.
CM_TO_PIXEL = 36.0 / 640
 
# Define the rotation matrix from the robotic base frame (frame 0)
# to the camera frame (frame c).
rot_angle = 180 # angle between axes in degrees
rot_angle = np.deg2rad(rot_angle)
rot_mat_0_c = np.array([[1, 0, 0],
                        [0, np.cos(rot_angle), -np.sin(rot_angle)],
                        [0, np.sin(rot_angle), np.cos(rot_angle)]])
 
# Define the displacement vector from frame 0 to frame c
disp_vec_0_c = np.array([[-17.8],
                         [24.4], # This was originally 23.0 but I modified it for accuracy
                         [0.0]])
 
# Row vector for bottom of homogeneous transformation matrix
extra_row_homgen = np.array([[0, 0, 0, 1]])
 
# Create the homogeneous transformation matrix from frame 0 to frame c
homgen_0_c = np.concatenate((rot_mat_0_c, disp_vec_0_c), axis=1) # side by side
homgen_0_c = np.concatenate((homgen_0_c, extra_row_homgen), axis=0) # one above the other
 
# Initialize coordinates in the robotic base frame
coord_base_frame = np.array([[0.0],
                             [0.0],
                             [0.0],
                             [1]])
 
# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
     
    # Grab the raw NumPy array representing the image
    image = frame.array
 
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Close gaps using closing
    gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
       
    # Remove salt and pepper noise with a median filter
    gray = cv2.medianBlur(gray,5)
     
    # If first frame, we need to initialize it.
    if first_frame is None:
         
      first_frame = gray
       
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
       
      # Go to top of for loop
      continue
       
    # Calculate the absolute difference between the current frame
    # and the first frame
    absolute_difference = cv2.absdiff(first_frame, gray)
 
    # If a pixel is less than ##, it is considered black (background). 
    # Otherwise, it is white (foreground). 255 is upper limit.
    # Modify the number after absolute_difference as you see fit.
    _, absolute_difference = cv2.threshold(absolute_difference, 95, 255, cv2.THRESH_BINARY)
 
    # Find the contours of the object inside the binary image
    contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
  
    # If there are no countours
    if len(areas) < 1:
  
      # Display the resulting frame
      cv2.imshow('Frame',image)
  
      # Wait for keyPress for 1 millisecond
      key = cv2.waitKey(1) & 0xFF
  
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
     
      # If "q" is pressed on the keyboard, 
      # exit this loop
      if key == ord("q"):
        break
     
      # Go to the top of the for loop
      continue
  
    else:
         
      # Find the largest moving object in the image
      max_index = np.argmax(areas)
       
    # Draw the bounding box
    cnt = contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
  
    # Draw circle in the center of the bounding box
    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(image,(x2,y2),4,(0,255,0),-1)
     
    # Calculate the center of the bounding box in centimeter coordinates
    # instead of pixel coordinates
    x2_cm = x2 * CM_TO_PIXEL
    y2_cm = y2 * CM_TO_PIXEL
     
    # Coordinates of the object in the camera reference frame
    cam_ref_coord = np.array([[x2_cm],
                              [y2_cm],
                              [0.0],
                              [1]])
     
    # Coordinates of the object in base reference frame
    coord_base_frame = homgen_0_c @ cam_ref_coord
  
    # Print the centroid coordinates (we'll use the center of the
    # bounding box) on the image
    text = "x: " + str(coord_base_frame[0][0]) + ", y: " + str(coord_base_frame[1][0])
    cv2.putText(image, text, (x2 - 10, y2 - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
    # Display the resulting frame
    cv2.imshow("Frame",image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
  
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If "q" is pressed on the keyboard, 
    # exit this loop
    if key == ord("q"):
      break
 
# Close down windows
cv2.destroyAllWindows()