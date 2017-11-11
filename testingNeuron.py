import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
   Data_direction = '/tmp/data'
   number_step = 1000
   min_batch_size = 100

   data = input_data.read_data_sets(Data_direction, one_hot= True)
   x = tf.placeholder(tf.float32, [None,784])
   w = tf.Variable(tf.zeros([784,10]))

   y_dir = tf.placeholder(tf.float32, [None, 10])
   y_pre = tf.matmul(x,w)

   loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_pre, labels= y_dir))
   gradient_de = tf.train.AdamOptimizer(0.5).minimize(loss_function)

   correction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_dir, 1))
   accurate = tf.reduce_mean(tf.cast(correction, tf.float32))

   with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for x in range(number_step):
         xs, ys = data.train.next_batch(min_batch_size)
         sess.run(gradient_de, feed_dict={x: xs, y_dir: ys})
      acc = sess.run(accurate, feed_dict={x: data.test.images, y_dir: data.test.labels})
   print (acc)

if __name__ == '__main__':
   main()

