import tensorflow as tf
tf.reset_default_graph()

def setup():
    Nodes_for_hidden_1 = 500
    Nodes_for_hidden_2 = 500
    Nodes_for_hidden_3 = 500

    Number_Of_Classes = 10
    batch_size = 100

    # hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, Nodes_for_hidden_1])),
    #                   'biases': tf.Variable(tf.random_normal([Nodes_for_hidden_1]))}
    # hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([Nodes_for_hidden_1, Nodes_for_hidden_2])),
    #                   'biases': tf.Variable(tf.random_normal([Nodes_for_hidden_2]))}
    # hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([Nodes_for_hidden_2, Nodes_for_hidden_3])),
    #                   'biases': tf.Variable(tf.random_normal([Nodes_for_hidden_3]))}
    # output_layer = {'weights': tf.Variable(tf.random_normal([Nodes_for_hidden_3, Number_Of_Classes])),
    #                 'biases': tf.Variable(tf.random_normal([Number_Of_Classes]))}

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "Desktop/tmp/model1.ckpt")
        print("Finish loading model.")
        # print("Loading weights")
        # print(hidden_layer_1['weights'].eval())
        # print(hidden_layer_2['weights'].eval())
        # print(hidden_layer_3['weights'].eval())
        # print(output_layer['weights'].eval())
        # print("Loading biases")
        # print(hidden_layer_1['biases'].eval())
        # print(hidden_layer_2['biases'].eval())
        # print(hidden_layer_3['biases'].eval())
        # print(output_layer['biases'].eval())

if __name__ == '__main__':
    setup()