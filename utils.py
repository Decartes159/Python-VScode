import tensorflow as tf
import numpy as np
import cv2

def load_facenet_model(pb_path):
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    return (face - mean) / std

def get_embedding(face, graph):
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('input:0')
        phase_train_tensor = graph.get_tensor_by_name('phase_train:0')
        embedding_tensor = graph.get_tensor_by_name('embeddings:0')

        face = preprocess_face(face)
        face = np.expand_dims(face, axis=0)

        feed_dict = {input_tensor: face, phase_train_tensor: False}
        embedding = sess.run(embedding_tensor, feed_dict=feed_dict)
        return embedding[0]
