from pre_process_dataset import *
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

training_images = train_data_with_label()
tr_img_data = np.array([i[0] for i in
                        training_images]).reshape(-1, 64, 64, 3)
tr_lbl_data = np.array([i[1] for i in training_images])


model = Sequential()

# model.add(InputLayer(input_shape=[64,64,3]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters= 100, kernel_size=5, strides=1, padding= 'same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax', input_dim=[64,64,3]))
optimizer = Adam(lr=1e-3)


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=50, batch_size=50)
model.summary()

# serialize model to JSON
model.save('Model/model.h5')
print("Saved model to disk")