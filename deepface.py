#%%
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, LocallyConnected2D, Flatten, Dense, Dropout, Input

class DeepFace(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.face_input = Input(shape=(152,152,3), name='face_input')
        self.conv1 = Conv2D(32, (11, 11), input_shape=(152, 152, 3), padding='valid', strides=1, name='C1', activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', name='M2')
        self.conv3 = Conv2D(16, (9, 9), input_shape=(71, 71, 32), padding='valid', strides=1, name='C3', activation='relu')
        self.lc4 = LocallyConnected2D(16, (9, 9), input_shape = (63, 63, 16), name='L4', activation='relu')
        self.lc5 = LocallyConnected2D(16, (7, 7), input_shape = (55, 55, 16), name='L5', activation='relu', strides=2)
        self.lc6 = LocallyConnected2D(16, (5, 5), input_shape = (25, 25, 16), name='L6', activation='relu')
        self.flatten = Flatten(name='Flatten')
        self.fc7 =  Dense(4096, activation='relu', name='F7')
        self.dropout = Dropout(0.5, name='Dropout')
        self.fc8 = Dense(1, activation='sigmoid', name='F8')  # 2, softmax

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.lc4(x)
        x = self.lc5(x)
        x = self.lc6(x)
        x = self.flatten(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = self.fc8(x)

        return x

    def summary_model(self):
        inputs = Input(shape=(152,152,3))
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepFace').summary()


class DeepFaceConv(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, (11, 11), input_shape=(152, 152, 3), padding='valid', strides=1, name='C1', activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', name='M2')
        self.conv3 = Conv2D(16, (9, 9), input_shape=(71, 71, 32), padding='valid', strides=1, name='C3', activation='relu')
        # self.lc4 = LocallyConnected2D(16, (9, 9), input_shape = (63, 63, 16), name='L4', activation='relu')
        # self.lc5 = LocallyConnected2D(16, (7, 7), input_shape = (55, 55, 16), name='L5', activation='relu', strides=2)
        # self.lc6 = LocallyConnected2D(16, (5, 5), input_shape = (25, 25, 16), name='L6', activation='relu')
        self.lc4 = Conv2D(16, (9, 9), input_shape = (63, 63, 16), padding='valid', strides=1, name='L4', activation='tanh')
        self.lc5 = Conv2D(16, (7, 7), input_shape = (55, 55, 16), padding='valid', strides=2, name='L5', activation='tanh')
        self.lc6 = Conv2D(16, (5, 5), input_shape = (25, 25, 16), padding='valid', strides=1, name='L6', activation='tanh')
        self.flatten = Flatten(name='Flatten')
        self.fc7 =  Dense(4096, activation='relu', name='F7')
        self.dropout = Dropout(0.5, name='Dropout')
        self.fc8 = Dense(1, activation='sigmoid', name='F8')  # 2, softmax

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.lc4(x)
        x = self.lc5(x)
        x = self.lc6(x)
        x = self.flatten(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = self.fc8(x)

        return x

    def summary_model(self):
        inputs = Input(shape=(152,152,3))
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepFace').summary()


class DeepFaceCross(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.face_input = Input(shape=(152,152,3), name='face_input')
        self.conv1 = Conv2D(32, (11, 11), input_shape=(152, 152, 3), padding='valid', strides=1, name='C1', activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', name='M2')
        self.conv3 = Conv2D(16, (9, 9), input_shape=(71, 71, 32), padding='valid', strides=1, name='C3', activation='relu')
        self.lc4 = LocallyConnected2D(16, (9, 9), input_shape = (63, 63, 16), name='L4', activation='relu')
        self.lc5 = LocallyConnected2D(16, (7, 7), input_shape = (55, 55, 16), name='L5', activation='relu', strides=2)
        self.lc6 = LocallyConnected2D(16, (5, 5), input_shape = (25, 25, 16), name='L6', activation='relu')
        self.flatten = Flatten(name='Flatten')
        self.fc7 =  Dense(4096, activation='relu', name='F7')
        self.dropout = Dropout(0.5, name='Dropout')
        self.fc8 = Dense(1, activation='sigmoid', name='F8')  # 2, softmax

    def call(self, inputs, cross=False):
        x = self.conv1(inputs)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.lc4(x)
        x = self.lc5(x)
        x = self.lc6(x)
        x = self.flatten(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = self.fc8(x)

        if cross:
            import numpy as np
            res = x.numpy() # .item()
            bs = res.shape[0]
            res = np.array([np.append(1 - res, res)]).reshape(bs, -1)
            res = tf.convert_to_tensor(res)
            # return tf.convert_to_tensor([[res, 1-res]])
            return res
        else:
            return x




