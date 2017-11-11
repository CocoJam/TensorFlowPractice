import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)
# one hot is good for multiple class classifications
# for example if  there is four class it can be 1,0,0,0 or 0,1,0,0 or 0,0,1,0 or 0,0,0,1

Nodes_for_hidden_1 = 500
Nodes_for_hidden_2 = 500
Nodes_for_hidden_3 = 500

Number_Of_Classes = 10
batch_size = 100

# martix is height by weight, hence 28 * 28 for image it could be a sting of values
Input_data = tf.placeholder('float', [None, 784])
Label_data = tf.placeholder('float')

def neural_network_modeling (data):
    # funtion (inputdata * weights) + biases
    # tf.summary.image('Image', Input_data)
    hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([784, Nodes_for_hidden_1])),
                      'biases' : tf.Variable(tf.random_normal([Nodes_for_hidden_1]))}
    hidden_layer_2 = {'weights' : tf.Variable(tf.random_normal([Nodes_for_hidden_1,Nodes_for_hidden_2])),
                      'biases' : tf.Variable(tf.random_normal([Nodes_for_hidden_2]))}
    hidden_layer_3 = {'weights' : tf.Variable(tf.random_normal([Nodes_for_hidden_2,Nodes_for_hidden_3])),
                      'biases' : tf.Variable(tf.random_normal([Nodes_for_hidden_3]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([Nodes_for_hidden_3,Number_Of_Classes])),
                      'biases' : tf.Variable(tf.random_normal([Number_Of_Classes]))}
    # linear function
    linear_function_layer1 = tf.add(tf.matmul(Input_data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    # activation function
    activation_function_layer1 = tf.nn.relu(linear_function_layer1)
    # linear function
    linear_function_layer2 = tf.add(tf.matmul(activation_function_layer1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    # activation function
    activation_function_layer2 = tf.nn.relu(linear_function_layer2)
    # linear function
    linear_function_layer3 = tf.add(tf.matmul(activation_function_layer2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    # activation function
    activation_function_layer3 = tf.nn.relu(linear_function_layer3)
    # At the end output layer
    output= tf.add(tf.matmul(activation_function_layer3, output_layer['weights']), output_layer['biases'])

    return output

def train_nn(x):
    predication = neural_network_modeling(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits= predication,labels= Label_data))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range (hm_epochs):
            epochs_cost = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict={Input_data: x, Label_data: y})
                epochs_cost += c
            print("epoch", i , "completed of", hm_epochs, "loss:", epochs_cost)
        correct = tf.equal(tf.argmax(predication,1), tf.argmax(Label_data,1))

        accuarcy = tf.reduce_mean(tf.cast(correct,"float"))

        print ("Accuracy", accuarcy.eval({Input_data: mnist.test.images, Label_data: mnist.test.labels}))
        write = tf.summary.FileWriter("Desktop/tmp/1")
        write.add_graph(sess.graph)
        saver = tf.train.Saver()
        save_path = saver.save(sess, "Desktop/tmp/model1.ckpt")
        print("Model saved in file: %s" % save_path)
        # tf.train.Saver().save(sess, "testing_model_1")

if __name__ == '__main__':
   train_nn(Input_data)