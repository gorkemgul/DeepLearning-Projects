#%%Libraries
import tensorflow as tf 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import transform
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#%%Data Preparation
#directory
dataDir = r'C:\Users\gorke\Desktop\cloth_dataset'
#data generating
trainDatagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.1)
testDatagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.1)
#ffd
trainDatagen = trainDatagen.flow_from_directory(dataDir, target_size = (224,224), subset = 'training', batch_size = 4)
testDatagen = testDatagen.flow_from_directory(dataDir, target_size = (224,224), subset = 'validation', batch_size = 4)
#%%An example visualization
for i in range(5):
    img,label = testDatagen.next()
    print(img.shape)
    plt.imshow(img[0])
    print(label[0])
    plt.show()
#%%Sequential Model Creating & Model Summary
model = Sequential([
    #conv & maxpooling 
    layers.Conv2D(4, (3,3), activation = 'relu', input_shape = (224,224,3)),
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(8, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(16, (3,3), activation = 'relu'),
    
    #flatten
    layers.Flatten(),
    
    #fully connected
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(2, activation = 'softmax')
    ])

#Model summary
model.summary()
#%%Model Training
#optimizer 
optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.01)
#loss
loss = tf.keras.losses.CategoricalCrossentropy()
#model compiling
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
#training
history = model.fit(trainDatagen, epochs = 5, verbose = 1, validation_data = testDatagen)
#evaluating model
model.evaluate(testDatagen)
#%%Model Testing
#Model Testing on Data Set
for i in range(5):
    img,labels = testDatagen.next()
    a = model.predict(img)
    np.argmax(a[0])
    plt.imshow(img[0])
    if np.argmax(a[0]) == 0:
        print('jeans')
    else:
        print('tshirt')
        plt.show()
#Model Testing on 1 Picture
def testpic(path):
    img = Image.open(path)
    img = np.array(img).astype('float32')/255
    img = transform.resize(img, (224,224,3))
    img = np.expand_dims(img, axis = 0)
    print(testDatagen.class_indices)
    return img        

#testing the picture
img = testpic('testpic1.jpg')
pred = model.predict(img)
print(pred)
print(np.argmax(pred))
#%%Model Visualization
#data preparing
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = range(1, len(acc)+1)
#Visualization
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
plt.plot(epoch, acc, label = 'Training accuracy', color = 'orange')
plt.plot(epoch, val_acc, label = 'Validation accuracy', color = 'blue')
plt.title('Training accuracy vs Validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epoch, loss, label = 'Training Loss', color = 'purple')
plt.plot(epoch, val_loss, label = 'Validation Loss', color = 'green')
plt.title('Training loss vs Validation loss')
plt.legend()
plt.show()
