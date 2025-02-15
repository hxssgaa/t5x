PROJECT_DIR=${HOME}"/t5x/dir1/user_dir"
T5X_DIR="/home/hxssgaa/t5x"  # directory where the t5x is cloned.
TFDS_DATA_DIR="/home/hxssgaa/data"
MODEL_DIR="gs://hxtpu_bucket/flan_t5_mix_xl_sea"
ACTIVATION_DTYPE=bfloat16
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="finetune_xl_tulu2.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"sea_flan\" \
  --gin.ACTIVATION_DTYPE=\"${ACTIVATION_DTYPE}\" \
  --gin.network.T5Config.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.utils.RestoreCheckpointConfig.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000\" \
  --gin.DROPOUT_RATE=0.05

# gs://hxtpu_bucket/t5_large_sea_ada/checkpoint_1180001