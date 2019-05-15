import keras
import numpy as np

if __name__ == '__main__':
    size=100

    model = keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None,
                                                 input_shape=None, pooling=None, classes=1000)
    print(np.argmax(model.predict(np.random.random((1, size, size, 3)))))

    x_train = np.random.random((2000, size, size, 3))
    y_train = keras.utils.to_categorical(np.random.randint(1000, size=(2000, 1)), num_classes=1000)

    model.compile(loss='categorical_crossentropy',optimizer='RMSprop')
    model.fit(x_train, y_train, batch_size=320, epochs=10)

    x_test = np.random.random((20, size, size, 3))
    y_test = keras.utils.to_categorical(np.random.randint(1000, size=(20, 1)), num_classes=1000)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)
