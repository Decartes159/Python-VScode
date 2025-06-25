import tensorflow as tf
import numpy as np
import cv2

def load_facenet_model(pb_model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return graph

def get_embedding(face, graph):
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('input:0')
            embeddings = graph.get_tensor_by_name('embeddings:0')
            phase_train = graph.get_tensor_by_name('phase_train:0')

            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')
            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            samples = np.expand_dims(face, axis=0)

            feed_dict = {input_tensor: samples, phase_train: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            return emb[0]
