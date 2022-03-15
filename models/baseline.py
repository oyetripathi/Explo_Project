from mimetypes import init
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras import Model

class Baseline_UNet():
    def __init__(self, input_shape, n_filters=32):
        self.input_shape = input_shape
        self.n_filters = n_filters

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x1 = self.EncoderBlock(inputs, self.n_filters, 0)
        x2 = self.EncoderBlock(x1[0], self.n_filters*2, 0)
        x3 = self.EncoderBlock(x2[0], self.n_filters*4, 0)
        x4 = self.EncoderBlock(x3[0], self.n_filters*8, 0.3)
        x5 = self.EncoderBlock(x4[0], self.n_filters*16, 0.3, False)

        y1 = self.DecoderBlock(x5[0], x4[1], self.n_filters*8)
        y2 = self.DecoderBlock(y1, x3[1], self.n_filters*4)
        y3 = self.DecoderBlock(y2, x2[1], self.n_filters*2)
        y4 = self.DecoderBlock(y3, x1[1], self.n_filters)

        conv = Conv2D(self.n_filters, 3, activation='relu', padding='same')(y4)
        conv = Conv2D(3, 1, padding='same')(conv)

        model = Model(inputs=inputs, outputs=conv)
        return model


    @staticmethod
    def EncoderBlock(inputs, n_fliters, dropout_prob=0.3, max_pooling=True):
        conv = Conv2D(n_fliters, 3, activation='relu', padding='same')(inputs)
        conv = Conv2D(n_fliters, 3, activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv, training=False)

        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)

        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2,2))(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

    @staticmethod
    def DecoderBlock(prev_layer_input, skip_layer_input, n_filters=32):
        up =  Conv2DTranspose(n_filters, (3,3), strides=2, padding='same')(prev_layer_input)
        merge = concatenate([up, skip_layer_input], axis=-1)
        conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
        conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
        return conv

model = Baseline_UNet((128,128,1), 32).build_model()
print(model.summary())
print("hello world")
