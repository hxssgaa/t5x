import functools

import seqio
import t5.data
import json
import jax
import tensorflow as tf
from t5.evaluation import metrics
from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

@seqio.map_over_dataset
def _process1(x):
    return {'inputs': x['inputs'].values[0], 'targets': x['targets'].values[0]}


# ================================ Wikipedia ===================================
TaskRegistry.add(
    "tulu_v2",
    source=seqio.TFExampleDataSource({'train': 'gs://hxtpu_bucket/tulu_tf_datasets/flan_v2-train.tfrecord-00000-of-00001',
                                      'validation': 'gs://hxtpu_bucket/tulu_tf_datasets/flan_v2-validation.tfrecord-00000-of-00001'}, feature_description={
        'inputs': tf.io.VarLenFeature(tf.string),
        'targets': tf.io.VarLenFeature(tf.string),
    }),
    preprocessors=[
        _process1,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu])
