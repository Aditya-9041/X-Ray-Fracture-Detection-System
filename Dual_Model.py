import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate  # <-- Added new layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16

import numpy as np
from collections import Counter

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = r"D:\BRAVE downloads\Fracture DATA\bone fracture detection.v4-v4.yolov8\train" 
valid_dir = r"D:\BRAVE downloads\Fracture DATA\bone fracture detection.v4-v4.yolov8\valid"
test_dir = r"D:\BRAVE downloads\Fracture DATA\bone fracture detection.v4-v4.yolov8\test"

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.1,
    height_shift_range=0.1,shear_range=0.1,zoom_range=0.2,horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir,
    target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode='categorical',shuffle=True )

valid_data = valid_datagen.flow_from_directory(valid_dir,
    target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode='categorical',shuffle=False)

test_data = test_datagen.flow_from_directory(test_dir,
    target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode='categorical',shuffle=False)

counter = Counter(train_data.classes)
total = float(sum(counter.values()))
num_classes = len(counter)

class_weights = {}
for class_index, count in counter.items():
    weight = (total / (num_classes * count))
    class_weights[class_index] = weight

print(f"--- Found {num_classes} classes ---")
print(f"Class counts: {counter}")
print(f"Calculated class weights: {class_weights}")
print("Using these weights to force the model to learn the rare class.")
print("-" * 30)

input_layer = Input(shape=(*IMAGE_SIZE, 3), name="image_input")

base_model = VGG16(input_shape=(*IMAGE_SIZE, 3),
                       include_top=False,
                       weights='imagenet')
base_model.trainable = False  
base_model._name = "VGG16_Base" 

vgg_branch = base_model(input_layer)
vgg_features = GlobalAveragePooling2D(name="vgg_gap")(vgg_branch)

custom_branch = Conv2D(16, (3, 3), activation='relu', padding='same', name="custom_conv1")(input_layer)
custom_branch = MaxPooling2D((2, 2), name="custom_pool1")(custom_branch)
custom_branch = Conv2D(32, (3, 3), activation='relu', padding='same', name="custom_conv2")(custom_branch)
custom_branch = MaxPooling2D((2, 2), name="custom_pool2")(custom_branch)
custom_branch = Conv2D(64, (3, 3), activation='relu', padding='same', name="custom_conv3")(custom_branch)
custom_branch = MaxPooling2D((2, 2), name="custom_pool3")(custom_branch)
custom_features = GlobalAveragePooling2D(name="custom_gap")(custom_branch)

merged_features = Concatenate(name="merge_layer")([vgg_features, custom_features])

x = Dense(128, activation='relu', name="final_dense_1")(merged_features)
x = Dropout(0.5, name="final_dropout")(x) 
output = Dense(train_data.num_classes, activation='softmax', name="final_output")(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy',metrics=['accuracy'])

print("\n--- Model Summary(Showing VGG16 + Custom Branch) ---")
model.summary()
print("-" * 30)

early_stopping = EarlyStopping(monitor='val_loss',
        patience=5,restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        patience=2,factor=0.1)

callbacks = [early_stopping, reduce_lr]

history = model.fit(train_data,validation_data=valid_data,
        epochs=50,callbacks=callbacks,class_weight=class_weights)

save_path = r"C:\Users\adity\OneDrive\Desktop\Fracture Detection Model\fracture_model_multi_branch.h5" 
model.save(save_path)

print(f"\n--- Multi-branch model saved to: {save_path} ---")
print("\n--- Evaluating new model on test data ---")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
