import cv2
import numpy as np
from skimage.transform import resize, pyramid_reduce
import numpy as np
from keras.models import load_model
import cv2
import PIL
from PIL import Image

# Load the pre-trained CNN model
model = load_model('CNNmodel.h5')

def prediction(pred):
    """
    Convert a predicted class into the corresponding ASCII character.
    """
    return chr(pred + 65)

def keras_predict(model, image):
    """
    Predict the class of the preprocessed image using the loaded CNN model.
    """
    data = np.asarray(image, dtype="float32")
    pred_probab = model.predict(data)[0]
    pred_class = np.argmax(pred_probab)
    return np.max(pred_probab), pred_class

def keras_process_image(img):
    """
    Preprocess the image for prediction, resizing it to 28x28 pixels.
    Uses pyramid_reduce for downsampling if necessary.
    """
    # Convert to grayscale using PIL for consistency with skimage usage
    img = Image.fromarray(img).convert('L')
    img = np.array(img)

    # Use pyramid_reduce for downsampling if the image is larger than the target size
    if img.shape[0] > 28 or img.shape[1] > 28:
        img = pyramid_reduce(img, downscale=2, multichannel=False)
    
    # Ensure the image is resized to 28x28 for the model
    img = resize(img, (28, 28), mode='reflect', anti_aliasing=True)
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    return img

def crop_image(image, x, y, width, height):
    """
    Crop a region of interest from the image.
    """
    return image[y:y + height, x:x + width]

def main():
    cam_capture = cv2.VideoCapture(0)
    
    while True:
        ret, image_frame = cam_capture.read()
        if not ret:
            break
        
        # Crop the region of interest
        cropped_img = crop_image(image_frame, 300, 300, 300, 300)
        
        # Convert cropped image to grayscale
        image_grayscale_blurred = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale_blurred, (15, 15), 0)
        
        # Process the image for the model
        processed_img = keras_process_image(image_grayscale_blurred)
        
        # Predict the class and convert to ASCII character
        _, pred_class = keras_predict(model, processed_img)
        curr = prediction(pred_class)
        
        # Display the prediction on the frame
        cv2.putText(image_frame, curr, (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Show the frame and the processed image
        cv2.imshow("Frame", image_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cam_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
