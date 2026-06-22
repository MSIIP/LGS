PROJECT_DIR=/home/shiym/projects/LGS
MODEL_PATH="${PROJECT_DIR}/work_dirs/sft-luna16-cls"

START=1          # 起始 index
END=20           # 结束 index（总轮数）
N=4              # 步长（例如40 → checkpoint-40, checkpoint-80...）
GPU_ID=0         # GPU ID

for (( i=${START}; i<=${END}; i++ ))
do
    CKPT=$((i * N))
    CKPT_DIR="${MODEL_PATH}/checkpoint-${CKPT}"
    RESULT_PATH="${CKPT_DIR}/eval/output.jsonl"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    swift infer \
        --model ${CKPT_DIR} \
        --val_dataset ${PROJECT_DIR}/work_dirs/dataset/luna16_cls_valid.json \
        --result_path ${RESULT_PATH} \
        --max_new_tokens 1024 \
        --num_beams 1 \
        --temperature 0 \
        --infer_backend vllm \
        --vllm_gpu_memory_utilization 0.9 \
        --stream True
done

for (( i=${START}; i<=${END}; i++ ))
do
    CKPT=$((i * N))
    CKPT_DIR="${MODEL_PATH}/checkpoint-${CKPT}"
    echo "evaluating for ${CKPT_DIR} (${i})"

    python ${PROJECT_DIR}/src/train_and_eval/eval_luna16_cls.py \
        --input_path ${CKPT_DIR}/eval/output.jsonl
done
