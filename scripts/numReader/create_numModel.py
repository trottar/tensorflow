#!/usr/bin/env python

# From video https://www.youtube.com/watch?v=wQ8BIBpya2k&t=605s

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 sized images of hand-written digits 0-9, each pixel of the 28x28 pixel image is assigned a bin value (28x28=784 bins) for the initial layer

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Loading data of mnist into arrays defined by x/y_train 

# Normalize image, so that the pixel color value is denoted by values up to unity
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Begin modeling
model = tf.keras.models.Sequential() # Sequential model (of two types) this is the most common type
model.add(tf.keras.layers.Flatten()) # Flattens image??? No clue, layer = "neutron", I'm assuming this just means that the image is being "flattened to the number line (i.e it is converted to the initial layer)

# Hidden layers (connections between initial values and final results, each connection has a weight to distinguish them) [128 units in layers (or 128 neutrons in the layer), activation is the "sigmoid" function discussed which determines (by percentage) which image it is most likely to be (sort of determines if the neutron fired or not)
# relu is the default type of function (rectified linear)
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # First layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # Second layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # Final layer, softmax is probability distro
# End of model

# This will train the model
# optimizer = 'adam' is the default optimizer, loss is the degree of error, metrics only looking at accuracy
# loss of error is calculated by adding up the square of the differences and seeing the result
# Example of loss: The number is 3 so...
#    0          1         2          3        ...
# {(0.02-0)^2+(0.3-0)^2+(0.06-0)^2+(0.97-1)^2+...}=loss of 0.03 (larger this value the more the program doesn't know)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# To actually train the model
model.fit(x_train,y_train,epochs=3) # Epochs is a full pass through entire training dataset, so epoch=3 is passing the full training twice

model.save('numReader.model',include_optimizer=False) # Save model for future predictions

# Calculate the validation loss and accuracy to assure that the model is not just "memorizing" the data set
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss,val_acc) # This should be similar to the loss and accuracy for model.fit but not the same and not too great of a divergence

# To load model
#new_model = tf.keras.models.load_model('test_numReader.model')
#new_model.compile()

predictions = model.predict([x_test]) # To make predictions (note: always takes a list), outputs a probability distro

# To get a value for prediction, i.e if it predicts a 7 and the image is a 7
for x in range(11):             # Gives range of numbers 0 to 10
    print("I think the number is " + str(np.argmax(predictions[x])))
    plt.imshow(x_test[x])           # Prints image 
    plt.show()

# print(x_train[0])               # print array elements

# plt.imshow(x_train[0], cmap = plt.cm.binary) # plt image of array data
# plt.show()
