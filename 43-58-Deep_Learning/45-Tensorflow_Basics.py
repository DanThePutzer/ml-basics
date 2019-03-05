import tensorflow as tf

# Tensorflow object
x1 = tf.constant(5)
x2 = tf.constant(6)

# Tensorflow performs multiplication
result = tf.multiply(x1,x2)

# Result will not be calculated until session is run -> Prints unevaluated tensor
print(result)

# Defining a session
sess = tf.Session()
# When session runs, output gets calculated and printed as a number
output = sess.run(result)
print(output)

# Closing session to save computing power
sess.close()