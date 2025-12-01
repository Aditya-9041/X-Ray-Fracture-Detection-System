import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
try:
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("--- ERROR ---")
    print("LIME or scikit-image is not installed.")
    print("Please run: pip install lime scikit-image")
    sys.exit()

IMAGE_SIZE = (224, 224)
MODEL_PATH = r"C:\Users\adity\OneDrive\Desktop\Fracture Detection Model\fracture_model_multi_branch.h5"
model = load_model(MODEL_PATH)

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy)
model.summary()

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, IMAGE_SIZE)
    array = resized.astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return img_rgb, array

def generate_lime_explanation(model, original_rgb_img):
    print("\n ... ... ...Generating LIME explanation... ... ...")
    explainer = lime_image.LimeImageExplainer()
    def lime_predict_fn(images):
        processed_imgs = []
        for img in images:
            resized = cv2.resize(img, IMAGE_SIZE)
            array = resized.astype("float32") / 255.0
            processed_imgs.append(array)

        batch = np.array(processed_imgs)
        return model.predict(batch)

    explanation = explainer.explain_instance(original_rgb_img, 
        lime_predict_fn, top_labels=2,         
        hide_color=0,num_samples=1000)
    
    print(" LIME explanation generated.")

    try:
        image_class_0, mask_class_0 = explanation.get_image_and_mask(0,positive_only=False, num_features=10,hide_rest=False)
        lime_img_class_0 = mark_boundaries(image_class_0, mask_class_0)
        
    except KeyError:
        print("Could not get explanation for Class 0")
        lime_img_class_0 = original_rgb_img

    try:
        image_class_1, mask_class_1 = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False)
        lime_img_class_1 = mark_boundaries(image_class_1, mask_class_1)
        
    except KeyError:
        print("Could not get explanation for Class 1")
        lime_img_class_1 = original_rgb_img 
    return lime_img_class_0, lime_img_class_1

def predict_fracture(img_path):

    orig, processed = preprocess_image(img_path)
    preds = model.predict(processed)
    class_index = np.argmax(preds)
    if class_index == 0:
        label = "Bone is FRACTURED"
    else:
        label = "Bone is NOT Fractured"
    print(f"\n Prediction: {label}")
    print(f" (Raw Class Index: {class_index})")
    print(f" Probabilities: {preds[0]}")

    lime_img_fractured, lime_img_not_fractured = generate_lime_explanation(model, orig)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original X-Ray")
    plt.imshow(orig) 
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("LIME: All Regions for 'FRACTURED'")
    plt.imshow(lime_img_fractured) 
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("LIME: All Regions for 'NOT FRACTURED'")
    plt.imshow(lime_img_not_fractured) 
    plt.axis("off")
    plt.show()

predict_fracture(r"C:\Users\adity\OneDrive\Pictures\Camera Roll\c0166563-800px-wm.jpg")