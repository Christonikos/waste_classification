#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses transfer learning (INCEPTION V3) to utilize a pretrained CNN
and classify in a binary manner images of waste. 

@author: Christos Nikolaos ZACHAROPOULOS
"""

# =============================================================================
# MODULES
# =============================================================================
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import random
# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil


# =============================================================================
# ALLIASES
# =============================================================================
join=os.path.join
see=os.listdir
real=os.path.realpath
exist=os.path.isdir
make=os.makedirs

def create(dir_):
    if not exist(dir_):
        make(dir_)

def move_imgs_to_validation(source_path, destination_path, percentage=10):
    print(f'Moving {percentage}% of data from: {source_path} to {destination_path}')
    data=see(source_path)
    k = len(data) * percentage // 100
    indices = random.sample(range(len(data)), k)
    for i in indices:
        source=join(source_path,data[i])
        shutil.move(source, destination_path) 




# =============================================================================
# DIRECTORIES
# =============================================================================
# path to data, stored in a different dir than that of the code
path2data=join(os.sep,'media','cz257680','Transcend2',
               'ISSonDL_Kaggle_challenge', 'dataset' )
# path to Inception weights
path2weights=real(join('..','weights'))



## TRAIN & TEST DIRS
# training directory
train_dir=join(path2data, 'TRAIN')
train_organic=join(train_dir,'O')
train_recycable=join(train_dir,'R')

## The dataset does not contain a validation directory, but we will create one
# to hyper-tune our model before testing it. 
valid_dir=join(path2data, 'VALID')
valid_organic=join(valid_dir,'O')
valid_recycable=join(valid_dir,'R')

# make the validation directories
list(map(create,[valid_dir, valid_organic, valid_recycable]))
## now move 20% of images from the training directory to the validation directory
# get the names from the Organic folder, sample and select 10% without replacement
# if the validation directories are not empty, skip this operation
if len(see(valid_organic))==0:
    # move images from the ORGANIC folder
    move_imgs_to_validation(train_organic, valid_organic, percentage=10)
    # move images from the RECYCABLE folder
    move_imgs_to_validation(train_recycable, valid_recycable, percentage=10)    

# testing directory
test_dir=join(path2data, 'TEST')
# to ease how the generator will handle the data (since they are unlabeled),
# we need to create another folder inside 'TEST' and move all the folders in
# there
embedded_dir=join(test_dir,'test_images')
if not exist(embedded_dir):
    print('Moving embedded sentences to a new directory.')
    make(embedded_dir)
    # get the names of the existing files inside the TEST dir
    files=[join(test_dir,i) for i in see(test_dir)]
    # move them to the new directory
    [shutil.move(f, embedded_dir) for f in files] 
    

# =============================================================================
#LOAD THE INCEPTION MODEL, LOCK THE DEEP LAYERS AND EXTRACT ONE OF THE HEAD-LAYERS 
# =============================================================================
# load the local weights
local_weights=join(path2weights,see(path2weights)[0])
# Get the pretrained model and set the weights to None (we are using the local instance of the weights)
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), # the images will be later normalized to fit this size
                                include_top = False, 
                                weights = None)
# add the local weights
pre_trained_model.load_weights(local_weights)
# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False
  
# =============================================================================
# Get one of the last layers (called 'mixed7)  
# =============================================================================
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# =============================================================================
# SET UP THE MODEL      
# =============================================================================

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x) # 1024
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)   

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()      
      
# =============================================================================
# SET UP THE TRAIN & TEST GENERATORS BASED ON THE DIRS      
# =============================================================================
# ~~~~~~~~~~~~~~~~~~~~~~
## TRAINING GENERATOR ##
# ~~~~~~~~~~~~~~~~~~~~~~
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., # this will take care of the different sizes
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 12,
                                                    color_mode="rgb",
                                                    shuffle=True,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150),
                                                    seed=42)     
# ~~~~~~~~~~~~~~~~~~~~~~~~
## VALIDATION GENERATOR ##
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator(rescale=1/255)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=12,
    class_mode="binary",
    shuffle=True,
    seed=42)

# ~~~~~~~~~~~~~~~~~~~~~~
## TESTING GENERATOR ##
# ~~~~~~~~~~~~~~~~~~~~~~
# Note that the testing data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
    
    
# =============================================================================
# CALLBACK CLASS TO STOP TRAINING ONCE WE'VE REACHED MORE THAN 0.97 ACCURACY 
# =============================================================================
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_acc')>0.97):
      print("\nReached 97.0% validation accuracy so cancelling training!")
      self.model.stop_training = True

my_calback_object = MyCallback()            


# =============================================================================
# FITTING/TRAINING THE MODEL
# =============================================================================
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size      

history = model.fit_generator(
            train_generator,
            validation_data = valid_generator,
            steps_per_epoch = STEP_SIZE_TRAIN,
            epochs = 100,
            validation_steps = STEP_SIZE_VALID,
            verbose = 2,
            callbacks=[my_calback_object])      

# =============================================================================
# EVALUATE THE MODEL
# =============================================================================
model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID)

# =============================================================================
# PREDICT THE OUTPUT
# =============================================================================

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

recycable=pred>=0.5
organic=pred<=0.5

epochs = range(len(accuracy))
plt.plot(epochs,accuracy)
plt.plot(epochs,val_accuracy)
plt.ylim([0.5, 1])
plt.axhline(0.5, linestyle='--', label='chance')
plt.plot(np.argmax(val_accuracy), np.max(val_accuracy),
         marker='*', markersize=12, label=np.max(val_accuracy))
plt.legend()
plt.show()


predicted_class_indices=np.argmax(pred,axis=1)
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# %%
## EVAL
preds = model.predict_generator(test_generator)
preds_cls_idx = preds.argmax(axis=-1)


idx_to_cls = {v: k for k, v in train_generator.class_indices.items()}
preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
filenames_to_cls = list(zip(test_generator.filenames, preds_cls))      
      
      
unique, counts = np.unique(preds_cls, return_counts=True)
dict(zip(unique, counts))      