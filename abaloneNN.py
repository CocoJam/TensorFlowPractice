import numpy as np
import tensorflow as tf
# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
# Abalone Data Set based on regression for rings number
# provided it is not too good as using classification and one-hot seems to be better
class NN:

    def __init__(self,input_number, Hiddenlayer_number, Output_number, sample, testing):
        self.input_number = input_number
        self.Hiddenlayer_number = Hiddenlayer_number
        self.Output_number = Output_number
        self.sample_size = len(sample)
        self.sample = sample
        self.testing = testing

    def create_Network(self, batch_size):
        Input_data = tf.placeholder('float', [None, self.input_number])
        Label_data = tf.placeholder('float')
        number_of_layers = self.Hiddenlayer_number
        Number_Of_Classes = self.Output_number
        Layers = []
        next_layer = self.input_number
        input_layer = [{'weights': tf.Variable(tf.random_normal([self.input_number, next_layer])),
                        'biases' : tf.Variable(tf.random_normal([next_layer]))}]
        Layers += input_layer
        for i in range (1,number_of_layers):
            hidden_layer = [{'weights': tf.Variable(tf.random_normal([self.input_number, next_layer])),
                        'biases' : tf.Variable(tf.random_normal([next_layer]))}]
            Layers += hidden_layer
        output_layer = [{'weights' : tf.Variable(tf.random_normal([next_layer,1])),
                      'biases' : tf.Variable(tf.random_normal([1]))}]
        Layers += output_layer
        print(Number_Of_Classes)
        for i in Layers:
            print("Layers" , i)


        numbers_of_layers_length = len(Layers)


        function_aggr = tf.add(tf.matmul(Input_data, (Layers[0])['weights']), (Layers[0])['biases'])
        activation_function = tf.nn.sigmoid(function_aggr)
        # activation_function = function_aggr
        for i in range(1, numbers_of_layers_length-1):
            # agregation Process
            function_aggr = tf.add(tf.matmul(activation_function, (Layers[i])['weights']), (Layers[i])['biases'])
            activation_function = tf.nn.sigmoid(function_aggr)
            # activation_function = function_aggr
        output = tf.add(tf.matmul(activation_function, (Layers[numbers_of_layers_length-1])['weights']), (Layers[numbers_of_layers_length-1])['biases'])
        output = (Number_Of_Classes * tf.sigmoid(output))
        print("output", output)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Label_data))
        cost =tf.reduce_mean(tf.square(Label_data - output))
        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

        hm_epochs = 10000
        # needed editing
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for m in range(hm_epochs):
                epochs_cost = 0
                for i in range(int(self.sample_size / batch_size)):
                    x, y = self.batching(batch_size, i, self.sample)
                    _, c = sess.run([optimizer, cost], feed_dict={Input_data: x, Label_data: y})
                    epochs_cost += c
                print("epoch", (m+1), "completed of", hm_epochs, "loss:", epochs_cost)

            # needed to think about the correction eval()
            print(output)
            # correct = tf.equal(tf.argmax(output, 1), tf.argmax(Label_data, 1))
            correct = tf.square(output- Label_data)
            accuarcy = tf.reduce_mean(tf.cast(correct, "float"))
            x, y = self.batching(batch_size, i, self.testing)
            print("Accuracy", accuarcy.eval({Input_data: x, Label_data: y}))

            write = tf.summary.FileWriter("Desktop/tmp/2")
            write.add_graph(sess.graph)
            saver = tf.train.Saver()
            save_path = saver.save(sess, "Desktop/tmp/2/model2.ckpt")
            print("Model saved in file: %s" % save_path)
            prediction_X =np.array([[1.,0.71,0.555,0.195,1.9485,0.9455,0.3765,0.495],[3.,0.3,0.215,0.05,0.1185,0.048,0.0225,0.042]])
            # 12 and 4
            print(sess.run(output, feed_dict={Input_data: prediction_X}))
        return output


    def batching(self,batch_size, number_of_batches,sample):
        samples = sample
        if (number_of_batches == 0):
            x = samples[0:batch_size,:(self.input_number)]
            y = samples[0:batch_size, self.input_number]
            y = y[:,np.newaxis]
            return x,y
        x = samples[number_of_batches*batch_size:(number_of_batches+1)* batch_size,:(self.input_number)]
        y = samples[number_of_batches*batch_size:(number_of_batches+1)* batch_size, self.input_number]
        y = (y[:,np.newaxis])
        return x, y


def setupdata(name):
    data = open(name, 'r').readlines()
    length_of_data = len(data)
    training_data = (int) (length_of_data * 0.66)
    testing_data = (int) (length_of_data *0.34)
    print(length_of_data, training_data, testing_data)
    for i in range(0, len(data)):
        data[i]= data[i].replace("\n","").split(",")
    data = np.array(data)
    for i in range(0, length_of_data):
        if(data[i,0] == 'M'):
            data[i, 0] = 1
        elif (data[i,0] == 'F'):
            data[i,0] = 2
        else :
            data[i, 0] = 3
    data = np.float32(data)
    # print(np.any(data[:,0] == 3))
    print(data)
    return data[:training_data], data[training_data:], data[:,len(data[0,:])-1]



if __name__ == '__main__':
    data_training, data_testing, output_number = setupdata("testing1.txt")
    input_numbers = len(data_training[0,:-1])
    Hiddenlayers_number = 10
    output_number = len(set(output_number))
    NN= NN(input_numbers, Hiddenlayers_number, output_number, data_training, data_testing)
    NN.create_Network(10)

