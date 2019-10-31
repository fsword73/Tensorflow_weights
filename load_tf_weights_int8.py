import json
import logging
import math
import os
import sys
from io import open

import numpy as np
np.set_printoptions(threshold=np.inf)

logger = logging.getLogger(__name__)

def compute_array_histogram_int8(array_data, tile_size):
    if tile_size < 1:
        return

    shape= array_data.shape
    ndim = array_data.ndim

    if ndim != 2:
        return
    if shape[0] < tile_size:
        return
    if shape[1] < tile_size:
        return
    #get max value 
    #max value = 127 
    #quantize to 8 bits 
    
    temp_array = array_data.__abs__()
    max_value = temp_array.max()
    
    temp_array = (temp_array*128/max_value).astype(np.int32)     
    
    total_tiles = shape[0]/tile_size * shape[1] * tile_size
    total_tiles = int(total_tiles)
    
    
    
    tiles_zero = 0
    #Statics of Tile
    for i in range(0, shape[0], tile_size): 
        for j in range(0, shape[1], tile_size): 
            temp = temp_array[i:i+tile_size, j:j+tile_size]
            sum_val = temp.sum()
            if sum_val == 0:
                tiles_zero = tiles_zero+1
    print ("%s, %sx%s, %s, %s \n" %("tiles", tile_size, tile_size, total_tiles, tiles_zero))    
    
    
        

def compute_array_histogram(array_data, tile_size):
    if tile_size < 1:
        return
    shape= array_data.shape
    ndim = array_data.ndim
    histogram_value = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-13, 1-15, 1e-25] )
    
    #temp_array= array_data.__abs__()
    temp_array = array_data
    histogram = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
    
    if ndim == 1:        
        for i in range(0, shape[0], 1):            
                temp_val = temp_array[i]
                #print(temp_val)
                for k in range(0, 14, 1):
                    if  temp_val > histogram_value[k]:                        
                        index = k
                        break
                
                histogram[index] = histogram[index]+1
    else:
        for i in range(0, shape[0], 1):            
            for j in range(0, shape[1], 1):
                index = 14
                temp_val = temp_array[i][j]
                for k in range(0, 14, 1):
                    if  temp_val > histogram_value[k]:
                        index = k
                        break
                histogram[index] = histogram[index]+1

    print(histogram) 
    
    
def load_tf_weights_in_bert(tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        #import numpy as np
        import tensorflow as tf
    except ImportError:
        #logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        #    "https://www.tensorflow.org/install/ for installation instructions.")        
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    #logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            #logger.info("Skipping {}".format("/".join(name)))
            continue
        #pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                l = l
                #pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                #pointer = getattr(pointer, 'bias')
                l = l
            elif l[0] == 'output_weights':
                l = l
                #pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                l = l
                #pointer = getattr(pointer, 'classifier')
            else:
                try:
                   # pointer = getattr(pointer, l[0])
                   l = l
                except AttributeError:
                    #logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                #pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            l = l
            #pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            array.shape == array.shape
            #assert pointer.shape == array.shape            
        except AssertionError as e:
            #e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        #pointer.data = torch.from_numpy(array)
        
        
        print(m_name)
        print(":{\n")
        print(array.shape)
        for tile_size in ( 1, 2, 4, 8,16,32):
            compute_array_histogram_int8(array, tile_size)    
            #compute_array_histogram(array,1)
        print("\n}\n");
        
if __name__ == '__main__':
    #docker mapping home to /data
    load_tf_weights_in_bert(sys.argv[1])    
    

