import functools

import seqio
import t5.data
from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

# ================================ Wikipedia ===================================
TaskRegistry.add(
    "sea_corpus",
    source=seqio.TextLineDataSource({'train': 'gs://hxtpu_bucket/sea_corpus/train_massive_filter.txt'}), #"wikipedia/20230601.en:1.0.0"),
    preprocessors=[
        # functools.partial(
        #     preprocessors.rekey, key_map={
        #         "inputs": None,
        #         "targets": None
        #     }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])
