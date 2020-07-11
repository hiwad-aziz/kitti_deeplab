"""
Run deeplab on kitti sequences and produces 4 results. For example, if seq 05:
    - res/05/col: it holds the segmentation result with color
    - res/05/lab: it holds the segmentation result with class id in [0,18]
    - res/05/lab_class: it holds binary maps for each class. In class_6, each
      image has {0,1} values, 1 if the pixel belongs to class 6. It is stored
      as a [0,1] image.
    - res/05/prob: it holds probability maps for each class. Maps are stored in
      [0,255] so you have to divide it by 255.0 after loading.
"""

import os
import sys
import time

import argparse
import scipy
import numpy as np
import cv2
import tensorflow as tf
from helper import logits2image, width, height

NUM_CLASS = 19

# TODO: Set these parameters 
OUT_DIR = 'res'
WRITE = True # if false, display results. Else, save it to file.
SEQ_L = ['05', '03'] # sequences to segment
DATA_ROOT_DIR='/home/abenbihi/ws/datasets/kitti/' # directory with all sequences
DATA_ROOT_DIR='/home/gpu_user/assia/ws/tf/segnet/datasets/kitti/'
IMG_SUBDIR = 'image_2/' # sequence subdirectory to segment

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph('frozen_inference_graph.pb')

# DeepLabv3+ input and output tensors
image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')
probs_op = graph.get_tensor_by_name('prefix/ResizeBilinear_2:0')
probs_op = tf.nn.softmax(probs_op, axis=3) 


global_start_time = time.time()

with tf.Session(graph=graph) as sess:

    for seq in SEQ_L:
    
        image_dir = os.path.join(DATA_ROOT_DIR, seq, IMG_SUBDIR)

        # create output directories
        lab_out_dir = os.path.join(OUT_DIR, seq, 'lab') # label map
        lab_class_out_dir = os.path.join(OUT_DIR, seq, 'lab_class')
        col_out_dir = os.path.join(OUT_DIR, seq, 'col') # color result
        prob_out_dir = os.path.join(OUT_DIR, seq, 'prob') # probability maps
        if not os.path.exists(lab_out_dir):
            os.makedirs(lab_out_dir)
        if not os.path.exists(lab_class_out_dir):
            os.makedirs(lab_class_out_dir)
        if not os.path.exists(col_out_dir):
            os.makedirs(col_out_dir)
        if not os.path.exists(prob_out_dir):
            os.makedirs(prob_out_dir)
        for k in range(NUM_CLASS):
            if not os.path.exists(os.path.join(lab_class_out_dir, 'class_%d'%k)):
                os.makedirs(os.path.join(lab_class_out_dir, 'class_%d'%k))
            if not os.path.exists(os.path.join(prob_out_dir, 'class_%d'%k)):
                os.makedirs(os.path.join(prob_out_dir, 'class_%d'%k))
        

        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith(".png"):
                lab_fn = os.path.join(lab_out_dir, fname)
                if os.path.exists(lab_fn):
                    continue
                duration = time.time() - global_start_time
                print('%s - %s  %d:%02d'%(seq, fname, duration/60, duration%60))
                img = cv2.imread(os.path.join(image_dir, fname))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                img = img[:,:,::-1]

                img = np.expand_dims(img, axis=0)
                lab, probs = sess.run([softmax, probs_op], {image_input: img})

                lab = np.squeeze(lab)
                probs = np.squeeze(probs)
                col = logits2image(lab)
                probs = (probs*255).astype(np.uint8)

                # save labels
                if WRITE:
                    col_fn = os.path.join(col_out_dir, fname)
                    cv2.imwrite(col_fn, col)
               
                    for k in range(probs.shape[2]):
                        prob_k_fn = os.path.join(prob_out_dir, 'class_%d'%k, fname)
                        #print(prob_k_fn)
                        cv2.imwrite(prob_k_fn, probs[:,:,k])
                        #cv2.imwrite(prob_k_fn, (probs[:,:,k]*255).astype(np.uint8))
                    
                    for k in range(probs.shape[2]):
                        lab_k_fn = os.path.join(lab_class_out_dir, 'class_%d'%k, fname)
                        cv2.imwrite(lab_k_fn, (lab==k).astype(np.uint8))
                
                    lab_fn = os.path.join(lab_out_dir, fname)
                    cv2.imwrite(lab_fn, lab)
                else:
                    cv2.imshow('col', col)
                    cv2.imshow('lab', lab)
                    cv2.waitKey(0)
               
                    for k in range(probs.shape[2]):
                        prob_k_fn = os.path.join(prob_out_dir, 'class_%d'%k, fname)
                        cv2.imshow('prob_k', probs[:,:,k])
                        cv2.waitKey(0)
                        #cv2.imwrite(prob_k_fn, (probs[:,:,k]*255).astype(np.uint8))
                    
                    for k in range(probs.shape[2]):
                        lab_k_fn = os.path.join(lab_class_out_dir, 'class_%d'%k, fname)
                        cv2.imshow(lab_k_fn, (lab==k).astype(np.uint8))
                        cv2.waitKey(0)
                
                    
