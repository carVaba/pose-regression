from keras.models import Model
from keras.utils.data_utils import get_file
from keras.utils import print_summary
from keras.utils import plot_model
import tensorflow as tf
import posereg
from posereg import pa16j


win_size = (256, 256)
input_shape = win_size + (3,)

weights_file = 'reception_mpii_weights_tf_ch_last_v1.h5'
TF_WEIGHTS_PATH = \
    'https://github.com/dluvizon/pose-regression/releases/download/0.1.1/' \
    + weights_file
md5_hash = '0f41d21e6c049ca590b520367f950f7f'
cache_subdir = 'models'


# Build the model and load the pre-trained weights on MPII
model = posereg.build(input_shape, pa16j.num_joints, export_heatmaps=True)
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
                        cache_subdir=cache_subdir)
model.load_weights(weights_path)
##model.ddsummay()
#print_summary(model)
#plot_model(model , to_file="model2.png" ,show_shapes=True, show_layer_names=True , rankdir='TB')
model.save("dluvison.h5")
import tensorflow as tf
from posereg.activations import channel_softmax_2d
_channel_softmax_2d = channel_softmax_2d()
converter = tf.lite.TFLiteConverter.from_keras_model_file("dluvison.h5" , custom_objects = {
    '_channel_softmax_2d': _channel_softmax_2d,
    'tf': tf
})
tflite_model = converter.convert()
