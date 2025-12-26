export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python wiyh_h5.py \
    --src-paths /mnt/data/data/zyp/wiyh/wiyh-sdk/data/tmp \
    --output-path wiyh_lerobot \
    --executor local \
    --tasks-per-job 3 \
    --workers 10
