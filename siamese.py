import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
import cv2
import os
import tensorflow.keras.backend as k
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
 
# 参数设置
# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100
 
# define the path to the base output directory
BASE_OUTPUT = "C:\\Users\\18446\\Desktop"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
MODEL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "Splot.png"])


def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
def get_imlist_2(path):
    ans=[]
    for f in os.listdir(path):
        if(f.endswith('.jpg')):
            ans.append(os.path.join(path,f))
    return ans
def get_pic_(path):
    c=get_imlist(path)
    siz=len(c)
    print(siz)
    datalist=[]
    os.makedirs(os.path.join('..','shoes_list'),exist_ok=True)
    for i in tqdm(range(siz)):
        img=cv2.imread(c[i],cv2.IMREAD_GRAYSCALE)#读取图片为灰度图
        img_ndarry=np.array(img,dtype='float64')#assary函数,转换为array
        data=cv2.resize(img_ndarry,(128,128))#大小重排
        data/=255
        plt.imshow(data)
        datalist.append(data) 
    return datalist
# 创建孪生网络模型的方法
def build_siamese_model(input_shape):
    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    #x = Dropout(0.3)(x)
    # prepare the final outputs
    x1=Flatten()(x)
    D_output =Dense(1024,activation="relu")(x1)
    D_output=Dense(256,activation='tanh')(D_output)
    out_puts = D_output
    # build the model
    my_model = Model(inputs, out_puts)
    # return the model to the calling function
    return my_model
 
 
# 创建图像对的方法
def make_pairs(images, labels):
    # 初始化两个空数组（图像对）和（标签，用来说明图像对是正向还是负向）
    pair_images = []
    pair_labels = []
    # 计算数据集中存在的类总数，然后为每个类标签建立索引列表，该列表为具有给定标签的所有示例提供索引
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]
 
    # 遍历所有图像
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        current_image = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idx_b = np.random.choice(idx[label])
        pos_image = images[idx_b]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        neg_idx = np.where(labels != label)[0]
        neg_image = images[np.random.choice(neg_idx)]
        # prepare a negative pair of images and update our lists
        pair_images.append([current_image, neg_image])
        pair_labels.append([0])
        # return a 2-tuple of our image pairs and labels
 
    return np.array(pair_images), np.array(pair_labels)
 
 
# 保存训练数据
def plot_training(h, plot_path):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)
 
 
# 计算欧氏距离
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsAA, featsBB) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = k.sum(k.square(featsAA - featsBB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

def reshapes(embed1):
    embed = tf.reshape(embed1, [-1, 256, 32, 1])

def paring_list(list_ans):
    res=[]
    lab=[]
    for i in range(len(list_ans)):
        if(i+5<len(list_ans)):
            res.append([list_ans[i],list_ans[i+5]])
            lab.append([0])
        res.append([list_ans[i],list_ans[i]]) 
        lab.append([1])
    return np.array(res),np.array(lab)
picture_path='D:\\TRAINING_csv\\2018.10.29'
list_a=[]
list_a=get_pic_(picture_path)
print(len(list_a))
print(tf.shape(list_a[0]))
plt.plot(list_a[0])
train_list=[]
train_label=[]
train_list,train_label=paring_list(list_a)
print(len(train_list))
print(len(train_label))
x_train=train_list[0:100]
y_lable=train_label[0:100]
x_test=train_list[100:]
y_test_lable=train_label[100:]

imgA = Input(IMG_SHAPE)
imgB = Input(IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
    
# 建立神经网络
distance = euclidean_distance([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
imgA = Input(IMG_SHAPE)
imgB = Input(IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
distance = euclidean_distance([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
imageA=tf.random.normal(shape=[128,128])
imageB=tf.random.normal(shape=[128,128])
imageA = np.expand_dims(imageA, axis=-1)
imageB = np.expand_dims(imageB, axis=-1)
    # add a batch dimension to both images
imageA = np.expand_dims(imageA, axis=0)
imageB = np.expand_dims(imageB, axis=0)
    # scale the pixel values to the range of [0, 1]
imageA = imageA / 255.0
imageB = imageB / 255.0
model.summary()
print(tf.shape([imageA,imageB]))
print(model.predict([imageA,imageB]))
model.compile(loss="binary_crossentropy", optimizer="adam",	metrics=["accuracy"])
#history = model.fit(x_train, y_lable, batch_size=2, epochs=10,validation_data=(x_test, y_test_lable))
print(tf.shape(x_train))
plot_model(model,to_file=MODEL_PLOT_PATH,show_shapes=True)
history = model.fit([x_train[:,0],x_train[:,1]], y_lable, validation_data=([x_test[:,0],x_test[:,1]], y_test_lable)
                    , batch_size=2, epochs=2
                   )
model.save(MODEL_PATH)
plot_training(history, PLOT_PATH)