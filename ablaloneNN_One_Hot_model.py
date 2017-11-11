import numpy as np
import tensorflow as tf


# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
# Abalone Data Set based on classification of number of rings
# This used the tech of one-Hot concept and the relu activation function. Instead of regression and sigmoid.
class NN:
    def __init__(self, input_number, Hiddenlayer_number, Output_number, sample, testing):
        self.input_number = input_number
        self.Hiddenlayer_number = Hiddenlayer_number+1
        self.Output_number = Output_number + 1
        self.sample_size = len(sample)
        self.sample = sample
        self.testing = testing

    def create_Network(self, batch_size):
        Input_data = tf.placeholder('float', [None, self.input_number])
        Label_data = tf.placeholder('float')
        number_of_layers = self.Hiddenlayer_number
        Number_Of_Classes = self.Output_number
        Layers = []
        # next_layer = (self.input_number)
        next_layer = (int) ((1/Hiddenlayers_number)*Number_Of_Classes)
        input_layer = [
            {'weights': tf.Variable(tf.random_normal([self.input_number, next_layer]), name="Weights_Input"),
             'biases': tf.Variable(tf.random_normal([next_layer]),name= "Biases_Input")}]
        tf.summary.histogram("Input_weights", ((input_layer[0])['weights']))
        tf.summary.histogram("Input_biases", ((input_layer[0])['biases']))
        Layers += input_layer

        for i in range(1, number_of_layers):
            next_layer1 = next_layer + (int) ((1/Hiddenlayers_number)*Number_Of_Classes)
            hidden_layer = [
                {'weights': tf.Variable(tf.random_normal([next_layer, next_layer1]), name="Weights_hidden_" + str(i)),
                 'biases': tf.Variable(tf.random_normal([next_layer1]),name= "Biases_hidden_" + str(i))}]
            next_layer = next_layer1
            tf.summary.histogram("hidden_weights_"+str(i), ((hidden_layer[0])['weights']))
            tf.summary.histogram("hidden_biases_"+str(i), ((hidden_layer[0])['biases']))
            Layers += hidden_layer
        output_layer = [
            {'weights': tf.Variable(tf.random_normal([next_layer, self.Output_number]), name="Weights_output"),
             'biases': tf.Variable(tf.random_normal([self.Output_number]), name="Biases_output")}]
        tf.summary.histogram("Output_weights", ((output_layer[0])['weights']))
        tf.summary.histogram("Output_biases", ((output_layer[0])['biases']))
        Layers += output_layer
        # print(Number_Of_Classes)
        # for i in Layers:
        #     print("Layers", i)
        numbers_of_layers_length = len(Layers)
        function_aggr = tf.nn.tanh(tf.add(tf.matmul(Input_data, (Layers[0])['weights']), (Layers[0])['biases']))

        for i in range(1, numbers_of_layers_length - 1):
            function_aggr = tf.nn.tanh(tf.add(tf.matmul(function_aggr, (Layers[i])['weights']), (Layers[i])['biases']))

        output = ((tf.add(tf.matmul(function_aggr, (Layers[numbers_of_layers_length - 1])['weights']),
                        (Layers[numbers_of_layers_length - 1])['biases'])))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Label_data))
        # cost =tf.square(output-Label_data)
        # tf.summary.scalar('cost', cost)
        # tf.summary.scalar('Accuracy', accuarcy)
        merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        hm_epochs = 100
        # needed editing
        with tf.Session() as sess:
            # summary = sess.run(merged, feed_dict={Input_data: x, Label_data: y})
            saver = tf.train.Saver()
            saver.restore(sess, "Desktop/tmp/3/model1.ckpt")
            write = tf.summary.FileWriter("Desktop/tmp/3")
            write.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            for m in range(hm_epochs):
                epochs_cost = 0
                for i in range(int(self.sample_size / batch_size)):
                    x, y = self.batching(batch_size, i, self.sample)
                    _, c , summary= sess.run([optimizer, cost, merged], feed_dict={Input_data: x, Label_data: y})
                    epochs_cost += c
                    write.add_summary(summary,m)
                print("epoch", (m + 1), "completed of", hm_epochs, "loss:", np.mean(epochs_cost))
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(Label_data, 1))

            accuarcy = tf.reduce_mean(tf.cast(correct, "float"))
            accuarcy_counter = 0
            accuarcy_accumlator = 0
            for i in range(int(len(self.testing)/batch_size)):
                x, y = self.batching(batch_size, i, self.testing)
                summary = sess.run(merged, feed_dict={Input_data: x, Label_data: y})
                write.add_summary(summary,i)
                out1 = sess.run(output,feed_dict={Input_data:x})
                # print(out1)
                print(sess.run(tf.argmax(out1,1)))
                print(sess.run(tf.argmax(y,1)))
                a=accuarcy.eval({Input_data: x, Label_data: y})
                accuarcy_counter += 1
                accuarcy_accumlator += a
                print("Accuracy", a)
            print("Accumlative", accuarcy_accumlator/accuarcy_counter)
            # print("Accuracy", accuarcy.eval({Input_data: x, Label_data: y}))


            # saver = tf.train.Saver()
            save_path = saver.save(sess, "Desktop/tmp/3/model1.ckpt")
            print("Model saved in file: %s" % save_path)


            # prediction_X =np.array([[1.,0.71,0.555,0.195,1.9485,0.9455,0.3765,0.495],[3.,0.3,0.215,0.05,0.1185,0.048,0.0225,0.042]])
            # 12 and 4
            # print(sess.run(output, feed_dict={Input_data: prediction_X}))
        return output

    def batching(self, batch_size, number_of_batches, samples):
        samples = samples
        ys = np.zeros([batch_size, self.Output_number])
        # print(samples)
        if (number_of_batches == 0):
            x = np.array(samples[0:batch_size, :(self.input_number)])
            x= x/ x.max(axis=0)
            y = samples[0:batch_size, self.input_number]
            for i in range(len(ys)):
                numbering = int(y[i] - 1)
                ys[i, numbering] = 1
            # print(ys)
            return x, ys
        x = np.array(samples[number_of_batches * batch_size:(number_of_batches + 1) * batch_size, :(self.input_number)])
        x = x / x.max(axis=0)
        y = samples[number_of_batches * batch_size:(number_of_batches + 1) * batch_size, self.input_number]
        for i in range(len(ys)):
            numbering = int(y[i] - 1)
            ys[i, numbering] = 1
        # print(ys)
        return x, ys


def setupdata(name):
    data = open(name, 'r').readlines()
    length_of_data = len(data)
    np.random.shuffle(data)
    training_data = (int)(length_of_data * 0.66)
    # testing_data = (int)(length_of_data * 0.34)
    # print(length_of_data, training_data, testing_data)
    for i in range(0, len(data)):
        data[i] = data[i].replace("\n", "").split(",")
    data = np.array(data)
    for i in range(0, length_of_data):
        if (data[i, 0] == 'M'):
            data[i, 0] = 1
        elif (data[i, 0] == 'F'):
            data[i, 0] = 2
        else:
            data[i, 0] = 3
    data = np.float32(data)
    # print(data)
    return data[:training_data], data[training_data:], data[:, len(data[0, :]) - 1]


if __name__ == '__main__':
    data_training, data_testing, output_number = setupdata("testing1.txt")
    input_numbers = len(data_training[0, :-1])
    Hiddenlayers_number = 4
    output_number = len(set(output_number))
    NN = NN(input_numbers, Hiddenlayers_number, output_number, data_training, data_testing)
    NN.create_Network(10)
