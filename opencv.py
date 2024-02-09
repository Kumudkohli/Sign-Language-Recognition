import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import pyramid_reduce

# Load the pre-trained Keras model
model = load_model('model.h5')

def get_square(image, square_size):
    """
    Resize image to a square of given size, maintaining aspect ratio.
    """
    height, width = image.shape
    differ = max(height, width)
    differ += 4

    # Create a square canvas and place the image in the center
    mask = np.zeros((differ, differ), dtype="uint8")
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

    # Downsample or resize to the target size
    if differ / square_size > 1:
        mask = pyramid_reduce(mask, downscale=differ / square_size, multichannel=False)
    else:
        mask = cv2.resize(mask, (square_size, square_size), interpolation=cv2.INTER_AREA)
    
    return mask * 255  # Scale back to 0-255 range if pyramid_reduce was used

def keras_predict(model, image):
    """
    Predict the class of the preprocessed image.
    """
    data = np.asarray(image, dtype="float32")
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    pred_probab = model.predict(data)[0]
    pred_class = np.argmax(pred_probab)
    return np.max(pred_probab), pred_class

def keras_process_image(img):
    """
    Preprocess the image for prediction.
    """
    img = get_square(img, 28)  # Resize and reshape the image
    img = np.reshape(img, (28, 28))  # Ensure the image is 28x28
    return img

def crop_image(image, x, y, width, height):
    """
    Crop the region of interest from the image.
    """
    return image[y:y + height, x:x + width]

# Main loop for capturing and processing video frames
def main():
    cam_capture = cv2.VideoCapture(0)
    
    while True:
        _, image_frame = cam_capture.read()
        if not _:
            break
        
        # Crop the region of interest and preprocess
        cropped_img = crop_image(image_frame, 300, 300, 300, 300)
        image_grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(image_grayscale, (15, 15), 0)
        resized_img = cv2.resize(image_blurred, (28, 28))  # Resize for the model
        ar = resized_img.reshape(1, 28*28)  # Flatten the image for the model

        # Predict the class of the cropped and processed image
        pred_probab, pred_class = keras_predict(model, ar)
        print(f"Predicted Class: {pred_class}, Probability: {pred_probab}")
        
        # Display the images
        cv2.imshow("Cropped Image", cropped_img)
        cv2.imshow("Processed Image", resized_img)

        # Break the loop with 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cam_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

