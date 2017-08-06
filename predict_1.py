# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
TRAIN_FILE = 'training.csv'
TEST_FILE = 'test.csv'
SAVE_PATH = 'model'


VALIDATION_SIZE = 100    #验证集大小
EPOCHS = 100             #迭代次数
BATCH_SIZE = 64          #每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 10 #控制early stopping的参数

def rgb2gray(rgb):
     return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#根据给定的shape定义并初始化卷积核的权值变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#根据shape初始化bias变量
def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

 

#最后生成提交结果的时候要用到
keypoint_index = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29
}

 
def conv2d(x,W):
   return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 96, 96, 1])
xx = tf.placeholder(tf.float32, [None, 9216])
y_ = tf.placeholder("float", shape=[None, 30])
keep_prob = tf.placeholder("float")

def model():
   W_conv1 = weight_variable([3, 3, 1, 32])
   b_conv1 = bias_variable([32])
   x_image = tf.reshape(xx, [-1,96,96,1])
   h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
   h_pool1 = max_pool_2x2(h_conv1)
   
   W_conv2 = weight_variable([2, 2, 32, 64])
   b_conv2 = bias_variable([64])
   
   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
   h_pool2 = max_pool_2x2(h_conv2)
   
   W_conv3 = weight_variable([2, 2, 64, 128])
   b_conv3 = bias_variable([128])
   
   h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
   h_pool3 = max_pool_2x2(h_conv3)
   
   W_fc1 = weight_variable([11 * 11 * 128, 500])
   b_fc1 = bias_variable([500])
   
   h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
   h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
   
   W_fc2 = weight_variable([500, 500])
   b_fc2 = bias_variable([500])
   
   h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
   h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
   
   W_fc3 = weight_variable([500, 30])
   b_fc3 = bias_variable([30])
   
   y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
   rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
   return y_conv, rmse




def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (96, 96), (255)) #creates white canvas of 96x96 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((96.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((90,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((96 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (3, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((96.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,90), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((96 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 3)) #paste resized image on white canvas
    
    newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    #print(tva)
    return tva
    #print(tva)




def save_model(saver,sess,save_path):
        path = saver.save(sess, save_path)
        print 'model save in :{0}'.format(path)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

    #变量都要初始化 
    sess.run(tf.initialize_all_variables())
     
    best_validation_loss = 1000000.0
    current_epoch = 0
    

    saver = tf.train.Saver()
    saver.restore(sess,SAVE_PATH)

    X = imageprepare("011.png")


    y_pred = []

  
  
    y_batch =  y_conv.eval(feed_dict={xx: [X],keep_prob: 1.0}, session=sess)
    y_pred.extend(y_batch)
     
  
    print y_pred[0]
    print y_pred[0][1]
    lena = mpimg.imread('sample.png')
    
    for i in range(0, 30): 
       if(i % 2 == 0):
          plt.plot(y_pred[0][i]*96 ,y_pred[0][i+1]*96  ,"+")
       print (" %d ", i)
    plt.imshow(lena) 
    #plt.plot(1,11,"+")
    plt.show()

