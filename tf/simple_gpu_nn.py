import time
import sys
import tensorflow as tf
import numpy as np
import argparse
import json

def us(t):
  return t * 1e6

FLAGS = None
def main():


  num_iter = FLAGS.num_iter
  BATCH_SIZE= FLAGS.batch_size
  layers = [int(s) for s in FLAGS.layers.split(",")]
  feed_in = FLAGS.feed
  with tf.device("/gpu:0"):
   if feed_in:
      batch = tf.placeholder(shape=[BATCH_SIZE, layers[0]], dtype=tf.float32)
   else:
      batch_in = tf.get_variable("batch",
		      initializer=tf.truncated_normal([BATCH_SIZE, layers[0]]))


   if feed_in:
     prev_layer = batch
   else:
     prev_layer = batch_in

   prev_size= layers[0]
   for size in layers[1:]:
     w = tf.get_variable("w_" + str(size),
		      initializer=tf.truncated_normal([prev_size, size]))
     b = tf.get_variable("b_" + str(size),
		      initializer=tf.truncated_normal([size]))
  
     prev_layer = tf.matmul(prev_layer, w) + b
     prev_size = size

   out = prev_layer
 
  with tf.Session() as sess:
  
    batch_np = np.random.randn(BATCH_SIZE, layers[0])
    s1 = time.time()
    sess.run(tf.global_variables_initializer())
    s2 = time.time()
    if feed_in:
      for i in range(num_iter): 
        sess.run(out.op, feed_dict={batch:batch_np})
    else:
      for i in range(num_iter): 
        sess.run(out.op)

    s3 = time.time()
    print("INIT:", us(s2-s1), "Forward Pass:", us(s3-s2)/num_iter, "Total:", us(s3-s1))
    data = {}
    data["NUM_ITERATIONS"] = num_iter
    data["TIME"] = str(round(us(s3-s2)/num_iter,5))
    data["LAYERS"] = layers
    data["BATCH_SIZE"] = BATCH_SIZE
    data["FEED"] = feed_in
    with open(FLAGS.output, "w") as f:
      json.dump(data, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_layers", type=int)
  parser.add_argument("--batch_size", type=int)
  parser.add_argument("--layers", type=str)
  parser.add_argument("--feed", type=int)
  parser.add_argument("--output", type=str)

  parser.add_argument("--num_iter", type=int)
  FLAGS = parser.parse_args()
  main()
