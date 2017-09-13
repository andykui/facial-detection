# facial-detection

Tensorflow的tutorial里面有介绍用CNN(卷积神经网络)来识别手写数字，直接把那里的代码copy下来跑一遍也是可以的。但是那比较没有意思，kaggle上有一个人脸关键点识别的比赛，有数据集也比较有意思，就拿这个来练手了。
定义卷积神经网络

首先是定义网络结构，在这个例子里我用了3个卷积层，第一个卷积层用3∗3的卷积核，后面两个用2∗2的卷积核。每个卷积层后面都跟max_pool池化层，之后再跟3个全连接层（两个隐层一个输出层）。每个卷积层的feature_map分别用32、64、128。

产生权值的函数代码如下

#根据给定的shape定义并初始化卷积核的权值变量
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    #根据shape初始化bias变量
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


定义卷积运算的代码如下。对tf.nn.con2d()的参数还是要说明一下
1. x是输入的样本，在这里就是图像。x的shape=[batch, height, width, channels]。
- batch是输入样本的数量
- height, width是每张图像的高和宽
- channels是输入的通道，比如初始输入的图像是灰度图，那么channels=1，如果是rgb，那么channels=3。对于第二层卷积层，channels=32。
2. W表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]。
3. strides参数表示的是卷积核在输入x的各个维度下移动的步长。了解CNN的都知道，在宽和高方向stride的大小决定了卷积后图像的size。这里为什么有4个维度呢？因为strides对应的是输入x的维度，所以strides第一个参数表示在batch方向移动的步长，第四个参数表示在channels上移动的步长，这两个参数都设置为1就好。重点就是第二个，第三个参数的意义，也就是在height于width方向上的步长，这里也都设置为1。
4. padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，’VALID’的话卷积以后图像的高为Heightout=Height原图−Height卷积核+1StrideHeight， 宽也同理。

def conv2d(x,W):
    return tf.nn.cov2d(x,W,strides=[1,1,1,1],padding='VALID')


接着是定义池化层的代码，这里用2∗2的max_pool。参数ksize定义pool窗口的大小，每个维度的意义与之前的strides相同，所以实际上我们设置第二个，第三个维度就可以了。

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


定义好产生权重、卷积、池化的函数以后就要开始组装这个卷积神经网络了。定义之前再定义一下输入样本x与对应的目标值y_。这里用了tf.placeholder表示此时的x与y_是指定shape的站位符，之后在定义网络结构的时候并不需要真的输入了具体的样本，只要在求值的时候feed进去就可以了。激活函数用relu，api也就是tf.nn.relu。
keep_prob是最后dropout的参数，dropout的目的是为了抗过拟合。

rmse是损失函数，因为这里的目的是为了检测人脸关键点的位置，是回归问题，所以用root-mean-square-error。并且最后的输出层不需要套softmax，直接输出y值就可以了。

这样就组装好了一个卷积神经网络。后续的步骤就是根据输入样本来train这些参数啦。

    x = tf.placeholder("float", shape=[None, 96, 96, 1])
    y_ = tf.placeholder("float", shape=[None, 30])
    keep_prob = tf.placeholder("float")

    def model():
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
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


训练卷积神经网络
读取训练数据

定义好卷积神经网络的结构之后，就要开始训练。训练首先是要读取训练样本。下面的代码用于读取样本。

    import pandas as pd
    import numpy as np

    TRAIN_FILE = 'training.csv'
    TEST_FILE = 'test.csv'
    SAVE_PATH = 'model'


    VALIDATION_SIZE = 100    #验证集大小
    EPOCHS = 100             #迭代次数
    BATCH_SIZE = 64          #每个batch大小，稍微大一点的batch会更稳定
    EARLY_STOP_PATIENCE = 10 #控制early stopping的参数


    def input_data(test=False):
        file_name = TEST_FILE if test else TRAIN_FILE
        df = pd.read_csv(file_name)
        cols = df.columns[:-1]

        #dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
        df = df.dropna()    
        df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)

        X = np.vstack(df['Image'])
        X = X.reshape((-1,96,96,1))

        if test:
            y = None
        else:
            y = df[cols].values / 96.0       #将y值缩放到[0,1]区间

        return X, y

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
 
开始训练

执行训练的代码如下，save_model用于保存当前训练得到在验证集上loss最小的模型，方便以后直接拿来用。

tf.InteractiveSession()用来生成一个Session，(好像是废话…)。Session相当于一个引擎，TensorFlow框架要真正的进行计算，都要通过Session引擎来启动。

tf.train.AdamOptimizer是优化的算法，Adam的收敛速度会比较快,1e-3是learning rate,这里先简单的用固定的。minimize就是要最小化的目标，当然是最小化均方根误差了。

    def save_model(saver,sess,save_path):
        path = saver.save(sess, save_path)
        print 'model save in :{0}'.format(path)

    if __name__ == '__main__':
        sess = tf.InteractiveSession()
        y_conv, rmse = model()
        train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

        #变量都要初始化 
        sess.run(tf.initialize_all_variables())
        X,y = input_data()
        X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
        X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

        best_validation_loss = 1000000.0
        current_epoch = 0
        TRAIN_SIZE = X_train.shape[0]
        train_index = range(TRAIN_SIZE)
        random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]

        saver = tf.train.Saver()

        print 'begin training..., train dataset size:{0}'.format(TRAIN_SIZE)
        for i in xrange(EPOCHS):
            random.shuffle(train_index)  #每个epoch都shuffle一下效果更好
            X_train, y_train = X_train[train_index], y_train[train_index]

            for j in xrange(0,TRAIN_SIZE,BATCH_SIZE):
                print 'epoch {0}, train {1} samples done...'.format(i,j)

                train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE], 
                    y_:y_train[j:j+BATCH_SIZE], keep_prob:0.5})

            #电脑太渣，用所有训练样本计算train_loss居然死机，只好注释了。
            #train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
            validation_loss = rmse.eval(feed_dict={x:X_valid, y_:y_valid, keep_prob: 1.0})

            print 'epoch {0} done! validation loss:{1}'.format(i, validation_loss*96.0)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_epoch = i
                save_model(saver,sess,SAVE_PATH)   #即时保存最好的结果
            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print 'early stopping'
                break



在测试集上预测

下面的代码用于预测test.csv里面的人脸关键点，最后的y值要乘以96，因为之前缩放到[0,1]区间了。

    X,y = input_data(test=True)
    y_pred = []

    TEST_SIZE = X.shape[0]
    for j in xrange(0,TEST_SIZE,BATCH_SIZE):
        y_batch = y_conv.eval(feed_dict={x:X[j:j+BATCH_SIZE], keep_prob:1.0})
        y_pred.extend(y_batch)

    print 'predict test image done!'

    output_file = open('submit.csv','w')
    output_file.write('RowId,Location\n')

    IdLookupTable = open('IdLookupTable.csv')
    IdLookupTable.readline()

    for line in IdLookupTable:
        RowId,ImageId,FeatureName = line.rstrip().split(',')
        image_index = int(ImageId) - 1
        feature_index = keypoint_index[FeatureName]
        feature_location = y_pred[image_index][feature_index] * 96
        output_file.write('{0},{1}\n'.format(RowId,feature_location))

    output_file.close()
    IdLookupTable.close()
    
    如上说明来自：http://blog.csdn.net/thriving_fcl/article/details/50909109
    我这边添加了预测的代码，和整理了代码和优化了。但是由于test数据太少，效果不是很理想
