# PROJECT_DIR=${HOME}"/t5x/dir1/user_dir"
# T5X_DIR="/home/hxssgaa/t5x"  # directory where the t5x is cloned.
# TFDS_DATA_DIR="/home/hxssgaa/data"
# MODEL_DIR="/home/hxssgaa/models"  # "/home/hxssgaa/models"
# ACTIVATION_DTYPE=bfloat16
# export PYTHONPATH=${PROJECT_DIR}

# python3 ${T5X_DIR}/t5x/train.py \
#   --gin_search_paths=${PROJECT_DIR} \
#   --gin_file="finetune_large_squad1.gin" \
#   --gin.MODEL_DIR=\"${MODEL_DIR}\" \
#   --gin.ACTIVATION_DTYPE=\"${ACTIVATION_DTYPE}\" \
#   --gin.network.T5Config.dtype=\"${ACTIVATION_DTYPE}\" \
#   --gin.utils.RestoreCheckpointConfig.dtype=\"${ACTIVATION_DTYPE}\" \
#   --gin.INIT_CHECKPOINT=\"gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000\" \
#   --gin.DROPOUT_RATE=0.1


MODEL_DIR="/home/hxssgaa/models"

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="/home/hxssgaa/tensorflow_datasets"
T5X_DIR="/home/hxssgaa/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt19_ende_train.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"squad_v010_allanswers\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5.1.1.base/model.ckpt-1000000\" \
  --tfds_data_dir=${TFDS_DATA_DIR}