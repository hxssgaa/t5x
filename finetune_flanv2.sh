MODEL_DIR="gs://hxtpu_bucket/flan_t5_large_sea_ada"
PROJECT_DIR=${HOME}"/t5x/dir1/user_dir"
# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="/home/hxssgaa/tensorflow_datasets"
T5X_DIR="/home/hxssgaa/t5x"  # directory where the T5X repo is cloned.
ACTIVATION_DTYPE=bfloat16
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="finetune_large_flan2.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"flan2022_submix\" \
  --gin.ACTIVATION_DTYPE=\"${ACTIVATION_DTYPE}\" \
  --gin.network.T5Config.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.utils.RestoreCheckpointConfig.dtype=\"${ACTIVATION_DTYPE}\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://hxtpu_bucket/t5_large_sea_ada/checkpoint_1180001\" \
  --gin.DROPOUT_RATE=0.05