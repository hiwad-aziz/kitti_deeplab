import os
import sys
import scipy
import numpy as np
import cv2
import tensorflow as tf
from helper import logits2image

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(sys.argv[1])
image_dir = sys.argv[2]

# DeepLabv3+ input and output tensors
image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')

# Create output directories in the image folder
if not os.path.exists(image_dir+'segmented_images/'):
    os.mkdir(image_dir+'segmented_images/')
if not os.path.exists(image_dir+'segmented_images_colored/'):
    os.mkdir(image_dir+'segmented_images_colored/') 

image_dir_segmented = image_dir+'segmented_images/'
image_dir_segmented_colored = image_dir+'segmented_images_colored/'

with tf.Session(graph=graph) as sess:
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png"):
            img = scipy.misc.imread(os.path.join(image_dir, fname)) 
            img = np.expand_dims(img, axis=0)
            probs = sess.run(softmax, {image_input: img})
            img = np.squeeze(probs)
            img_colored = logits2image(img)
            cv2.imwrite(image_dir_segmented+fname,img)
            cv2.imwrite(image_dir_segmented_colored+fname, img_colored)   
            print(fname)
