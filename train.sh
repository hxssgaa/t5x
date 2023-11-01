PROJECT_DIR=${HOME}"/t5x/dir1/user_dir"
T5X_DIR="/home/hxssgaa/t5x"  # directory where the t5x is cloned.
TFDS_DATA_DIR="/home/hxssgaa/data"
MODEL_DIR="gs://hxtpu_bucket/t5_large_sea_ada"
ACTIVATION_DTYPE=bfloat16
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="pretrain_large_sea_corpus.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.ACTIVATION_DTYPE=\"${ACTIVATION_DTYPE}\" \
  --gin.network.T5Config.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.utils.RestoreCheckpointConfig.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_large/checkpoint_1100000\" \
  --gin.DROPOUT_RATE=0.0