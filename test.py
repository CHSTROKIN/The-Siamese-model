import time 
import tensorflow as tf
import matplotlib as plt 
import numpy as np
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
MODEL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "Splot.png"])
load_model = tf.keras.models.load_model(MODEL_PATH)
'''
picture_path='D:\\TRAINING_csv\\2018.10.29'
list_a=[]
list_a=get_pic_(picture_path)
'''
load_model.summary()
a=50
b=50
plt.subplot(1,2,1)
plt.imshow(list_a[a])
plt.subplot(1,2,2)
plt.imshow(list_a[b])
plt.show()
uw=[[list_a[a],list_a[b]]]
uw=np.array(uw)
t1=time.time()
print("similarity:",load_model.predict([uw[:,0],uw[:,1]]))
t2=time.time()
print("time to compare:",t2-t1)