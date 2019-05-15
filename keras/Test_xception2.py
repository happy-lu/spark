import keras
import numpy as np

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

if __name__ == '__main__':
    size = 100

    base_model = keras.applications.xception.Xception(include_top=False, input_tensor=None,
                                                      input_shape=(size, size, 3), pooling='avg', classes=20)

    # 添加全局平均池化层
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(2048, activation='relu')(x)
    keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                  beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                  beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                                  gamma_constraint=None)

    # 添加一个分类器，假设我们有10个类
    predictions = Dense(10, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # model = base_model
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))

    # x_train = np.random.random((100, size, size, 3))
    # y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    # model.fit(x_train, y_train, batch_size=32, epochs=10)

    x_test = np.random.random((20, size, size, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)
