from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.datasets import mnist

#download data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#crete arrays for learning

X_train = X_train.reshape((60000,28,28,1)).astype('float32')/255
X_test = X_test.reshape((10000,28,28,1)).astype('float32')/255

Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

# ml
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, 
    epochs=12, verbose=1, validation_data=(X_test, Y_test))

score_eval = model.evaluate(X_test, Y_test)

print(score_eval)

# Save model
model.save('/home/alex/urfu_linux/mlops/labs/lab3/ml_for_microservice/model.h5')
