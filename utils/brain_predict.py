from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]
#load trained brain model
brain_model = load_model(
    "models/brain_model.keras",
    safe_mode=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)


#load and preprocess uploaded image
def load_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array,axis=0)

    return img_array
    
#predict class
def predict_brain(image_path):

    img_array = load_image(image_path)

    predictions = brain_model.predict(
        img_array,
        verbose=0
    )
    predicted_index=np.argmax(predictions[0])
    predicted_class=CLASS_NAMES[predicted_index]
    confidence=float(np.max(predictions[0])*100)

    return {
        "class":predicted_class,
        "confidence":confidence
    }