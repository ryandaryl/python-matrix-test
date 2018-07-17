import datetime, time
import tensorflow as tf
from tensorflow.python.framework import graph_util

def load_graph_from_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def create_graph(a,b,c):
    global sess
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        A = tf.Variable(initial_value=tf.random_normal([a, b]))
        B = tf.Variable(initial_value=tf.random_normal([b, c]))
        C = tf.matmul(A, B, name='output')
        sess.run(tf.global_variables_initializer())
        return g

def freeze_graph(g, filename):
    global sess
    output_graph = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output'])
    with tf.gfile.GFile('graph.pb', "wb") as f:
        f.write(output_graph.SerializeToString())

def calculate_flops(a,b,c):
    filename = './graph.pb'
    g = create_graph(a, b, c)
    print('Flops should be ~',2*a*b*c)
    freeze_graph(g, filename)
    g = load_graph_from_pb(filename)
    with g.as_default():
        start_time = datetime.datetime.now()
        flops = tf.profiler.profile(g, options = tf.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print('Flops:', flops)
        print('Seconds', elapsed_time)
        print('Gflops per second:', flops / elapsed_time / 10**9)

calculate_flops(25000, 15000, 9000)
