import time
import sys
import tensorflow as tf
import numpy as np


argv = sys.argv
num_iter = int(argv[2])
def us(t):
  return t * 1e6
BATCH_SIZE= int(argv[3])

if int(argv[1]):
  batch = tf.placeholder(shape=[BATCH_SIZE, 784], dtype=tf.float32)
else:
  batch_in = tf.get_variable("batch",
		      initializer=tf.truncated_normal([BATCH_SIZE, 784]))


sizes = [ 512,128,1]
if int(argv[1]):
  prev_layer = batch
else:
  prev_layer = batch_in

prev_size= 784
for size in sizes:
  w = tf.get_variable("w_" + str(size),
		      initializer=tf.truncated_normal([prev_size, size]))
  b = tf.get_variable("b_" + str(size),
		      initializer=tf.truncated_normal([size]))
  
  prev_layer = tf.matmul(prev_layer, w) + b
  prev_size = size

out = prev_layer
 
with tf.Session() as sess:
  
  batch_np = np.random.randn(BATCH_SIZE, 784)
  s1 = time.time()
  sess.run(tf.global_variables_initializer())
  s2 = time.time()
  if int(argv[1]):
    for i in range(num_iter): 
      sess.run(out.op, feed_dict={batch:batch_np})
  else:
    for i in range(num_iter): 
      sess.run(out.op)

  s3 = time.time()
  print("INIT:", us(s2-s1), "Forward Pass:", us(s3-s2)/num_iter, "Total:", us(s3-s1))
  

  
