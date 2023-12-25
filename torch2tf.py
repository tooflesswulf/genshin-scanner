# from onnx_tf.backend import prepare
import torch
import sys

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import tensorflow as tf
import tensorflowjs as tfjs

from training import SimpleNetwork, LModule


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} /path/to/checkpoint.ckpt')
        exit(0)

    model_path = sys.argv[1]
    print(f'Converting {model_path} from torch to TF.')
    m = LModule(SimpleNetwork(im_size=200))
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    m.load_state_dict(ckpt['state_dict'])
    m.eval()

    dummy_input = torch.zeros((1, 3, 200, 200))
    print(m.model(dummy_input))

    keras_model = nobuco.pytorch_to_keras(
        m.model,
        args=[dummy_input], kwargs=None,
        inputs_channel_order=ChannelOrder.TENSORFLOW,
        outputs_channel_order=ChannelOrder.TENSORFLOW
    )
    print(keras_model(tf.zeros((1, 200, 200, 3))))

    tfjs.converters.save_keras_model(keras_model, 'tfjs')

