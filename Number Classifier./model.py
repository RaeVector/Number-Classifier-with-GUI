from builtins import print
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import tensorflow_datasets as tfds


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

#here we turn the image tnesor into a 32 bit float and divide the values by 255 (turns each value into a 1 or a 0)
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).cache()
ds_train = ds_train.batch(128).shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.cache().batch(128).map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

basic_cnn_model = keras.models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
            ])

basic_cnn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

basic_cnn_model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

basic_cnn_model.save('basic_cnn.keras')