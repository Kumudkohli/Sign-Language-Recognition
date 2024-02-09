import cv2
import numpy as np
import pickle

def build_squares(img):
    """
    Draw squares on the image to guide the user for placing their hand.
    This helps in capturing a histogram of the hand.
    """
    x, y, w, h = 420, 140, 10, 10
    d = 10
    crop = None
    for i in range(10):  # 10 rows of squares
        imgCrop = None
        for j in range(5):  # 5 columns of squares
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    """
    Capture a hand histogram by guiding the user with squares to place their hand.
    The histogram is then saved for future use.
    """
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC = False
    while True:
        ret, img = cam.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        if not flagPressedC:
            imgCrop = build_squares(img)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            break
        
        if flagPressedC:
            show_histogram_result(img, hist)
        
        cv2.imshow("Set hand histogram", img)
    
    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)

def show_histogram_result(img, hist):
    """
    Apply the histogram to the current frame and show the thresholded image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.merge((thresh, thresh, thresh))
    cv2.imshow("Thresh", thresh)

if __name__ == "__main__":
    get_hand_hist()
