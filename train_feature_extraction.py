import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

nb_classes=43
batch_size=128
epochs=10

# TODO: Load traffic signs data.
with open('./train.p','rb') as f:
    data=pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'],data['labels'], test_size=0.33, random_state=0)
# TODO: Define placeholders and resize operation.
X=tf.placeholder(tf.float32,[None,32,32,3])
y=tf.placeholder(tf.int64,None)
resized=tf.image.resize_images(X,(227,227))
    

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
shape=(fc7.get_shape().as_list()[-1],nb_classes)
fc8w=tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b=tf.Variable(tf.zeros(nb_classes))
logits=tf.matmul(fc7,fc8w)+fc8b

# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss_op=tf.reduce_mean(cross_entropy)
accu_op=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),y),tf.float32))
train_op=tf.train.AdamOptimizer().minimize(loss_op)
init_op=tf.global_variables_initializer()

def eval_on_data(features, labels, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = features[offset:end]
        y_batch = labels[offset:end]

        loss, acc = sess.run([loss_op, accu_op], feed_dict={X: X_batch, y: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/features.shape[0], total_acc/features.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={X: X_train[offset:end], y: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")


# TODO: Train and evaluate the feature extraction model.
