import functools

import seqio
import t5.data
import json
import jax
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
    import pdb; pdb.set_trace()
    return json.loads(x)

# ================================ Wikipedia ===================================
TaskRegistry.add(
    "tulu_v2",
    source=seqio.TextLineDataSource({'train': 'gs://hxtpu_bucket/inst_tuning/t5_tulu_v2_data.jsonl'}), #"wikipedia/20230601.en:1.0.0"),
    preprocessors=[
        _process1,
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": "inputs",
                "targets": "targets"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder()
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])
