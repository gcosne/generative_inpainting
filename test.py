import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

parser.add_argument('--image_dir', default='', type=str,
                    help='The directory of image to be completed')

parser.add_argument('--mask_dir', default='', type=str,
                    help='The directory of masks')
parser.add_argument('--output_dir', default='', type=str,
                    help='The directory of output, it must exists')



def input_imagef(white_im,mask_path):
    image = cv2.imread(white_im)
    mask  = cv2.imread(mask_path)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
   
    return(input_image)

if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()
    
    image_dir = args.image_dir
    mask_dir  = args.mask_dir
    output_dir = args.output_dir
    
    model = InpaintCAModel()
    
    list_img = os.listdir(image_dir)
    
    input_im = input_imagef(image_dir+list_img[0],mask_dir+list_img[0])

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        
        input_image =tf.placeholder(dtype=tf.float32,shape=input_im.shape,name='input_image')
        #         tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        
        print('Model loaded.')
        #         result = sess.run(output)

        for im_name in list_img:
            print('processing im:',im_name)
            input_im = input_imagef(image_dir+im_name,mask_dir+im_name)
            output_path = output_dir + im_name
            feed_dict = {input_image: input_im}
            result = sess.run(output, feed_dict=feed_dict)
            cv2.imwrite(output_path, result[0][:, :, ::-1])
