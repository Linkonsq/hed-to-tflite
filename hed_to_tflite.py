import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Define paths
DESKTOP_PATH = "D:\\Projects\\Python\\hed_to_tflite" # path of your model file and prototxt
CAFFE_MODEL = os.path.join(DESKTOP_PATH, "hed_pretrained_bsds.caffemodel")
PROTOTXT = os.path.join(DESKTOP_PATH, "deploy.prototxt")
OUTPUT_TFLITE = os.path.join(DESKTOP_PATH, "hed.tflite")

def caffe_to_tf(caffemodel_path, prototxt_path, output_path):
    # Verify files exist
    if not all([os.path.exists(caffemodel_path), os.path.exists(prototxt_path)]):
        raise FileNotFoundError("Missing Caffe model or prototxt file")
    
    # Load Caffe model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    # Build HED architecture
    input_layer = layers.Input(shape=(None, None, 3), name='input')
    x = layers.Lambda(lambda x: (x - [104.00698793, 116.66876762, 122.67891434]) / 255.0)(input_layer)
    
    # Store layers in a dictionary for weight assignment
    layer_dict = {}
    
    # VGG-style encoder
    x, layer_dict = _conv_block(x, net, 'conv1', 64, 2, layer_dict)
    x, layer_dict = _conv_block(x, net, 'conv2', 128, 2, layer_dict)
    x, layer_dict = _conv_block(x, net, 'conv3', 256, 3, layer_dict)
    x, layer_dict = _conv_block(x, net, 'conv4', 512, 3, layer_dict)
    x, layer_dict = _conv_block(x, net, 'conv5', 512, 3, layer_dict)
    
    # Side outputs
    side_outputs = []
    for i in range(1, 6):
        side, layer_dict = _side_output(x, net, f'score-dsn{i}', f'upscore-dsn{i}', layer_dict)
        side_outputs.append(side)
    
    # Fusion layer
    fusion = _fusion_layer(side_outputs, net, layer_dict)
    
    # Create model
    tf_model = Model(inputs=input_layer, outputs=fusion)
    
    # Assign weights after model is built
    _assign_weights(tf_model, layer_dict, net)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")

def _conv_block(x, net, prefix, filters, num_convs, layer_dict):
    for i in range(1, num_convs + 1):
        layer_name = f'{prefix}_{i}'
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        conv_layer = layers.Conv2D(filters, (3, 3), activation='relu', name=layer_name)
        x = conv_layer(x)
        layer_dict[layer_name] = conv_layer
    
    if prefix != 'conv5':
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'pool{prefix[-1]}')(x)
    
    return x, layer_dict

def _side_output(x, net, score_name, upsample_name, layer_dict):
    score_layer = layers.Conv2D(1, (1, 1), name=score_name)
    score = score_layer(x)
    layer_dict[score_name] = score_layer
    
    upsample = layers.UpSampling2D(size=(1, 1), name=upsample_name)(score)
    return upsample, layer_dict

def _fusion_layer(side_outputs, net, layer_dict):
    concat = layers.Concatenate()(side_outputs)
    fusion_layer = layers.Conv2D(1, (1, 1), name='score-fuse')
    fusion = fusion_layer(concat)
    layer_dict['score-fuse'] = fusion_layer
    return fusion

def _assign_weights(model, layer_dict, net):
    for layer_name, layer in layer_dict.items():
        try:
            blobs = net.getLayer(layer_name).blobs
            weights = np.array(blobs[0].data).transpose((2, 3, 1, 0))
            biases = np.array(blobs[1].data)
            layer.set_weights([weights, biases])
            print(f"Successfully assigned weights for {layer_name}")
        except Exception as e:
            print(f"Failed to assign weights for {layer_name}: {str(e)}")

# Run conversion
try:
    caffe_to_tf(
        caffemodel_path=CAFFE_MODEL,
        prototxt_path=PROTOTXT,
        output_path=OUTPUT_TFLITE
    )
except Exception as e:
    print(f"Conversion failed: {str(e)}")