#conding=gbk
from keras import utils
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import load_model
#from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import h5py
import scipy.io as sci
import time
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "5,0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

stime1=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(stime1+"------begin")

num_classes = 3
adam = Adam(lr=1e-4)
kernel_size = 9
data=h5py.File('astralcull_13.mat', 'r')
xtrain_0=np.transpose(data['p1_4d'][:])
xtrain_1=np.transpose(xtrain_0,(3,0,1,2))
x_train1 = xtrain_1.reshape(-1, 20, 13, 1).astype('float32')
ytrain_1=np.transpose(data['t1_double'][:])            #
y_train = utils.to_categorical(ytrain_1.T-1, num_classes)#
#测试集
matr =h5py.File('casp10_13.mat', 'r')
xtest_0=np.transpose(matr['p1_4d'][:])
xtest_1=np.transpose(xtest_0,(3,0,1,2))
x_test1 = xtest_1.reshape(-1,  20,13, 1).astype('float32')
ytest_1=np.transpose(matr['t1_double'][:])
y_test = utils.to_categorical(ytest_1.T-1, num_classes)

####
data=h5py.File('astralcull_19.mat', 'r')
xtrain_0=np.transpose(data['p1_4d'][:])
xtrain_1=np.transpose(xtrain_0,(3,0,1,2))
x_train2 = xtrain_1.reshape(-1, 20, 19, 1).astype('float32')
#测试集
matr =h5py.File('casp10_19.mat', 'r')
xtest_0=np.transpose(matr['p1_4d'][:])
xtest_1=np.transpose(xtest_0,(3,0,1,2))
x_test2 = xtest_1.reshape(-1,  20,19, 1).astype('float32')

####
data=h5py.File('astralcull_27.mat', 'r')
xtrain_0=np.transpose(data['p1_4d'][:])
xtrain_1=np.transpose(xtrain_0,(3,0,1,2))
x_train3 = xtrain_1.reshape(-1, 20, 27, 1).astype('float32')
#测试集
matr =h5py.File('casp10_27.mat', 'r')
xtest_0=np.transpose(matr['p1_4d'][:])
xtest_1=np.transpose(xtest_0,(3,0,1,2))
x_test3 = xtest_1.reshape(-1,  20,27, 1).astype('float32')

# channel 1
inputs1 = Input(shape=(20, 13, 1))
conv1 = Conv2D(filters=430, kernel_size=12, padding='same',activation='relu')(inputs1)
channel_axis = 1 if K.image_data_format() == "channels_first" else -1
channel1 = conv1._keras_shape[channel_axis]
shared_layer_one1 = Dense(channel1 // 8,activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
shared_layer_two1 = Dense(channel1,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
avg_pool1 = GlobalAveragePooling2D()(conv1)
avg_pool1 = Reshape((1, 1, channel1))(avg_pool1)
assert avg_pool1._keras_shape[1:] == (1, 1, channel1)
avg_pool1 = shared_layer_one1(avg_pool1)
assert avg_pool1._keras_shape[1:] == (1, 1, channel1 // 8)
avg_pool1 = shared_layer_two1(avg_pool1)
assert avg_pool1._keras_shape[1:] == (1, 1, channel1)

max_pool1 = GlobalMaxPooling2D()(conv1)
max_pool1 = Reshape((1, 1, channel1))(max_pool1)
assert max_pool1._keras_shape[1:] == (1, 1, channel1)
max_pool1 = shared_layer_one1(max_pool1)
assert max_pool1._keras_shape[1:] == (1, 1, channel1 // 8)
max_pool1 = shared_layer_two1(max_pool1)
assert max_pool1._keras_shape[1:] == (1, 1, channel1)

cbam_feature1 = Add()([avg_pool1, max_pool1])
cbam_feature1 = Activation('sigmoid')(cbam_feature1)

if K.image_data_format() == "channels_first":
    cbam_feature1 = Permute((3, 1, 2))(cbam_feature1)
cbam_feature11 = multiply([conv1, cbam_feature1])
if K.image_data_format() == "channels_first":
    channel1 = cbam_feature11._keras_shape[1]
    cbam_feature1 = Permute((2, 3, 1))(cbam_feature11)
else:
    channel1 = cbam_feature11._keras_shape[-1]
    cbam_feature1 = cbam_feature11
avg_pool1 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature1)
assert avg_pool1._keras_shape[-1] == 1
max_pool1 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature1)
assert max_pool1._keras_shape[-1] == 1
concat1 = Concatenate(axis=3)([avg_pool1, max_pool1])
assert concat1._keras_shape[-1] == 2
cbam_feature1 = Conv2D(filters=1, kernel_size=kernel_size,strides=1, padding='same',activation='sigmoid', kernel_initializer='he_normal',use_bias=False)(concat1)
assert cbam_feature1._keras_shape[-1] == 1
if K.image_data_format() == "channels_first":
    cbam_feature1 = Permute((3, 1, 2))(cbam_feature1)
x1=multiply([cbam_feature11, cbam_feature1])

conv1= Conv2D(filters=530, kernel_size=3, activation='relu')(x1)
conv1=Dropout(0.5)(conv1)
flat1 = Flatten()(conv1)
# channel 2
inputs2 = Input(shape=(20, 19, 1))
conv2 = Conv2D(filters=430, kernel_size=18, padding='same',activation='relu')(inputs2)
channel_axis = 1 if K.image_data_format() == "channels_first" else -1
channel2 = conv2._keras_shape[channel_axis]
shared_layer_one2 = Dense(channel2 // 8,activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
shared_layer_two2 = Dense(channel2,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
avg_pool2 = GlobalAveragePooling2D()(conv2)
avg_pool2 = Reshape((1, 1, channel2))(avg_pool2)
assert avg_pool2._keras_shape[1:] == (1, 1, channel2)
avg_pool2 = shared_layer_one2(avg_pool2)
assert avg_pool2._keras_shape[1:] == (1, 1, channel2 // 8)
avg_pool2 = shared_layer_two2(avg_pool2)
assert avg_pool2._keras_shape[1:] == (1, 1, channel2)
max_pool2 = GlobalMaxPooling2D()(conv2)
max_pool2 = Reshape((1, 1, channel2))(max_pool2)
assert max_pool2._keras_shape[1:] == (1, 1, channel2)
max_pool2 = shared_layer_one2(max_pool2)
assert max_pool2._keras_shape[1:] == (1, 1, channel2 // 8)
max_pool2 = shared_layer_two2(max_pool2)
assert max_pool2._keras_shape[1:] == (1, 1, channel2)

cbam_feature2 = Add()([avg_pool2, max_pool2])
cbam_feature2 = Activation('sigmoid')(cbam_feature2)
if K.image_data_format() == "channels_first":
    cbam_feature2 = Permute((3, 1, 2))(cbam_feature2)
cbam_feature12 = multiply([conv2, cbam_feature2])
if K.image_data_format() == "channels_first":
    channel2 = cbam_feature12._keras_shape[1]
    cbam_feature2 = Permute((2, 3, 1))(cbam_feature12)
else:
    channel2 = cbam_feature12._keras_shape[-1]
    cbam_feature2 = cbam_feature12
avg_pool2 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature2)
assert avg_pool2._keras_shape[-1] == 1
max_pool2 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature2)
assert max_pool2._keras_shape[-1] == 1
concat2 = Concatenate(axis=3)([avg_pool2, max_pool2])
assert concat2._keras_shape[-1] == 2
cbam_feature2 = Conv2D(filters=1, kernel_size=kernel_size,strides=1,padding='same',activation='sigmoid',kernel_initializer='he_normal',use_bias=False)(concat2)
assert cbam_feature2._keras_shape[-1] == 1
if K.image_data_format() == "channels_first":
    cbam_feature2 = Permute((3, 1, 2))(cbam_feature2)
x2=multiply([cbam_feature12, cbam_feature2])
conv2 = Conv2D(filters=530, kernel_size=6, activation='relu')(x2)
conv2=Dropout(0.5)(conv2)
flat2 = Flatten()(conv2)
# channel 3
inputs3 = Input(shape=(20, 27, 1))
conv3 = Conv2D(filters=430, kernel_size=26, padding='same',activation='relu')(inputs3)
channel_axis = 1 if K.image_data_format() == "channels_first" else -1
channel3 = conv3._keras_shape[channel_axis]
shared_layer_one = Dense(channel3 // 8,activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
shared_layer_two = Dense(channel3,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
avg_pool3 = GlobalAveragePooling2D()(conv3)
avg_pool3 = Reshape((1, 1, channel3))(avg_pool3)
assert avg_pool3._keras_shape[1:] == (1, 1, channel3)
avg_pool3 = shared_layer_one(avg_pool3)
assert avg_pool3._keras_shape[1:] == (1, 1, channel3 // 8)
avg_pool3 = shared_layer_two(avg_pool3)
assert avg_pool3._keras_shape[1:] == (1, 1, channel3)
max_pool3 = GlobalMaxPooling2D()(conv3)
max_pool3 = Reshape((1, 1, channel3))(max_pool3)
assert max_pool3._keras_shape[1:] == (1, 1, channel3)
max_pool3 = shared_layer_one(max_pool3)
assert max_pool3._keras_shape[1:] == (1, 1, channel3 // 8)
max_pool3 = shared_layer_two(max_pool3)
assert max_pool3._keras_shape[1:] == (1, 1, channel3)
cbam_feature3 = Add()([avg_pool3, max_pool3])
cbam_feature3 = Activation('sigmoid')(cbam_feature3)
if K.image_data_format() == "channels_first":
    cbam_feature3 = Permute((3, 1, 2))(cbam_feature3)
cbam_feature13 = multiply([conv3, cbam_feature3])
if K.image_data_format() == "channels_first":
    channel3 = cbam_feature13._keras_shape[1]
    cbam_feature3 = Permute((2, 3, 1))(cbam_feature13)
else:
    channel3 = cbam_feature13._keras_shape[-1]
    cbam_feature3 = cbam_feature13
avg_pool3 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature3)
assert avg_pool3._keras_shape[-1] == 1
max_pool3 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature3)
assert max_pool3._keras_shape[-1] == 1
concat3 = Concatenate(axis=3)([avg_pool3, max_pool3])
assert concat3._keras_shape[-1] == 2
cbam_feature3 = Conv2D(filters=1,kernel_size=kernel_size,strides=1,padding='same', activation='sigmoid',kernel_initializer='he_normal',use_bias=False)(concat3)
assert cbam_feature3._keras_shape[-1] == 1
if K.image_data_format() == "channels_first":
    cbam_feature3 = Permute((3, 1, 2))(cbam_feature3)
x3=multiply([cbam_feature13, cbam_feature3])
conv3 = Conv2D(filters=530, kernel_size=9, activation='relu')(x3)
conv3 = Dropout(0.5)(conv3)
flat3 = Flatten()(conv3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(128, activation='softsign')(merged)
x=Dropout(0.5)(dense1 )
x=Dense(64,activation='softsign')(x)
outputs = Dense(3, activation='sigmoid')(x)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# compile
model = multi_gpu_model(model, gpus=2)
model.compile(loss='Mloss2', optimizer=adam, metrics=['accuracy'])

n_epochs=50
n_batch_size=128
checkpointer = ModelCheckpoint(filepath='Ml2.h5', monitor='val_acc', verbose=1, save_best_only=True)
model.fit([x_train1,x_train2,x_train3],y_train,validation_data=([x_test1,x_test2,x_test3],y_test),
                batch_size=n_batch_size, epochs=n_epochs,  shuffle=True, callbacks=[checkpointer])
model = load_model('Ml2.h5')
score = model.evaluate([x_test1,x_test2,x_test3], y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
stime2=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(stime1+"------begin")
print(stime2+"------end")