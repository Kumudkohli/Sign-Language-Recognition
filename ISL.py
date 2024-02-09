import cv2
import numpy as np

# Load the image
cap = cv2.imread('a.png', 0)
if cap is None:
    print("Error loading image")
else:
    # Apply binary threshold
    retval, threshold = cv2.threshold(cap, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', threshold)

    # Edge detection
    edges = cv2.Canny(threshold, 100, 200)
    cv2.imshow('Edges', edges)

    # Assuming lower and upper bounds for skin color in HSV
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # Convert to HSV and apply skin mask
    # First, convert the grayscale back to BGR as cvtColor expects a color image for conversion to HSV
    cap_color = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
    converted = cv2.cvtColor(cap_color, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # Apply a series of erosions and dilations to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # Apply the skin mask to the original image (converted to color)
    skin = cv2.bitwise_and(cap_color, cap_color, mask=skinMask)
    cv2.imshow('Skin', skin)

    # Wait for key press and then destroy all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

