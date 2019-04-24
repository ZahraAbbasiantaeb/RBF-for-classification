import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from data import x_train, x_test, x_validation, y_train, y_test, y_validation, cluster_count,\
    y_train_one_hot, y_validation_one_hot, num_of_classes, plot3D, y_test_one_hot, X_train_rbf,\
    X_test_rbf, X_validation_rbf, X_test_ebf, X_train_ebf, X_validation_ebf

x_train_phi = X_train_ebf

X_validation_phi = X_validation_ebf

X_test_phi = X_test_ebf

path = 'ebf_'+str(cluster_count)

epochs = 1000

hidden_neurons = cluster_count

outPut_neurons = num_of_classes

train_Y = y_train_one_hot

test_Y = y_test_one_hot

validation_Y = y_validation_one_hot


def init_weights(shape):

    return tf.Variable(tf.random_normal(shape, stddev=0.01))



def feedForward_NN(x, w1, b1):

    Z = tf.nn.sigmoid(tf.matmul(x,w1) + b1)

    return Z


tf.reset_default_graph()

tfX = tf.placeholder(tf.float32, [None, hidden_neurons])
tfY = tf.placeholder(tf.float32, [None, outPut_neurons])

w1 = init_weights([hidden_neurons, outPut_neurons])
b1 = init_weights([outPut_neurons])

validation_acc_plc = tf.placeholder(dtype=tf.float32)

network = feedForward_NN(tfX, w1, b1)

with tf.name_scope("training"):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=tfY,
                                                                  name='cost'))
    optimizer = tf.train.GradientDescentOptimizer(0.3, name='optimizer')

    train_op = optimizer.minimize(cost, name='train_op')


# train accuracy
train_acc = tf.equal(tf.argmax(network, 1), tf.argmax(train_Y, 1))
train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float32))

# validation accuracy
validation_acc = tf.equal(tf.argmax(network, 1), tf.argmax(validation_Y, 1))
validation_acc = tf.reduce_mean(tf.cast(validation_acc, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# add summaries
tf.summary.scalar('train_loss', cost)
tf.summary.scalar('train_accuracy', train_acc)
tf.summary.scalar('val_acc', validation_acc_plc)

merged_summary_op = tf.summary.merge_all()

validation_loss = []
train_loss = []
step = 0


def plot_loss(train_loss, validation_loss):

    plt.title('Loss curve')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(validation_loss, label = 'validation_loss')
    plt.legend(loc='upper left')
    plt.show()

    return


with tf.Session() as sess:

    sess.run(init)

    writer = tf.summary.FileWriter(path+"/graph")

    writer.add_graph(graph=tf.get_default_graph())

    for epoch in range(0, epochs):

        valid_acc = validation_acc.eval({tfX: X_validation_phi, tfY: validation_Y})

        _, validation_cost, s = sess.run([train_op, cost, merged_summary_op], feed_dict={tfX:x_train_phi, tfY:train_Y, validation_acc_plc: validation_acc.eval({tfX: X_validation_phi, tfY: validation_Y})}, )

        validation_loss.append(sess.run([cost], feed_dict={tfX:X_validation_phi, tfY:validation_Y}))

        train_loss.append(validation_cost)

        writer.add_summary(s, step)

        pred_train = sess.run(tf.argmax(network, 1), feed_dict={tfX:x_train_phi, tfY:train_Y})

        if(epoch%100 == 0):

            print('epoch is: '+ str(epoch))
            print(train_acc.eval({tfX: x_train_phi, tfY: train_Y}))
            print(validation_acc.eval({tfX: X_validation_phi, tfY: validation_Y}))

        step += 1

    save_path = saver.save(sess, path+"/model.ckpt")

    print("Model saved in path: %s" % save_path)

    plot3D(x_train, pred_train)


    print('test info:')
    pred_test = sess.run(tf.argmax(network, 1), feed_dict={tfX:X_test_phi, tfY:test_Y})
    print(accuracy_score(y_test, pred_test))
    print(confusion_matrix(y_test, pred_test))


    print('train info:')
    print(train_acc.eval({tfX:x_train_phi, tfY:train_Y}))
    print(confusion_matrix(y_train, pred_train))

    plot_loss(train_loss, validation_loss)

    plt.show()
