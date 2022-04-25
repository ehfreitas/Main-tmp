from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('./data/dog_images/train')
valid_files, valid_targets = load_dataset('./data/dog_images/valid')
test_files, test_targets = load_dataset('./data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("./data/dog_images/train/*/"))]

# print statistics about the dataset
#print('There are %d total dog categories.' % len(dog_names))
#print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
#print('There are %d training dog images.' % len(train_files))
#print('There are %d validation dog images.' % len(valid_files))
#print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("./data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
#print('\nThere are %d total human images.' % len(human_files))

# DETECTING HUMANS

import cv2                
import matplotlib.pyplot as plt                        

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
#for (x,y,w,h) in faces:
    # add bounding box to color image
#    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
#cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
#plt.imshow(cv_rgb)
#plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
#humans = 0
#for human_file in human_files_short:
#    tmp = face_detector(human_file)
#    if tmp > 0:
#        humans += 1
#print('\nDetected humans in {}% of human_files_short.'.format(humans/len(human_files_short)*100))

#humans = 0
#for dog_file in dog_files_short:
#    tmp = face_detector(dog_file)
#    if tmp > 0:
#        humans += 1
#print('Detected humans in {}% of dog_files_short.'.format(humans/len(dog_files_short)*100))

# DETECTING DOGS

from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from tensorflow.keras.applications.resnet50 import preprocess_input#, decode_predictions
#from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
dogs = 0
for human_file in human_files_short:
    tmp = dog_detector(human_file)
    if tmp > 0:
        dogs += 1
print('\nDetected dogs in {}% of human_files_short.'.format(dogs/len(human_files_short)*100))

dogs = 0
for dog_file in dog_files_short:
    tmp = dog_detector(dog_file)
    if tmp > 0:
        dogs += 1
print('Detected dogs in {}% of dog_files_short.'.format(dogs/len(dog_files_short)*100))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('./models/DogResnet50Data.npz')
train_resnet50 = bottleneck_features['train']
valid_resnet50 = bottleneck_features['valid']
test_resnet50 = bottleneck_features['test']

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

### TODO: Define your architecture.
resnet50_model = Sequential()
resnet50_model.add(GlobalAveragePooling2D(input_shape=train_resnet50.shape[1:]))
resnet50_model.add(Dense(133, activation='softmax'))

resnet50_model.summary()

### TODO: Compile the model.
resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint 

### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='./models/weights.best.ResNet50.hdf5', 
                               verbose=1, save_best_only=True)

resnet50_model.fit(train_resnet50, train_targets, 
                   validation_data=(valid_resnet50, valid_targets),
                   epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

### TODO: Load the model weights with the best validation loss.
resnet50_model.load_weights('./models/weights.best.ResNet50.hdf5')

# save the model to disk
#resnet50_model.save(filepath='./models/ResNet50.h5')

### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
resnet50_predictions = [np.argmax(resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(resnet50_predictions)==np.argmax(test_targets, axis=1))/len(resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

from extract_bottleneck_features import *

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def Resnet50_predict_dogbreed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    print(bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[1].replace('_', ' ')

print(Resnet50_predict_dogbreed('./models/amstaff.jpg'))
