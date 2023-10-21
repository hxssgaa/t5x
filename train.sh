PROJECT_DIR=${HOME}"/t5x/dir1/user_dir"
T5X_DIR="/home/hxssgaa/t5x"  # directory where the t5x is cloned.
TFDS_DATA_DIR="/home/hxssgaa/data"
MODEL_DIR="gs://hxtpu_bucket/t5"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="small_pretrain_dummy_wikipedia.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\"