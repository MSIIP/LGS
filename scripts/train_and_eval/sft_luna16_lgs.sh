PROJECT_DIR=/home/shiym/projects/LGS
MODEL_PATH="${PROJECT_DIR}/work_dirs/sft-luna16-lgs"
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29501 \
swift sft \
    --deepspeed zero2 \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset ${PROJECT_DIR}/work_dirs/dataset/luna16_lgs_train.json ${PROJECT_DIR}/work_dirs/dataset/luna16_det_train.json \
    --split_dataset_ratio 0 \
    --output_dir $MODEL_PATH \
    --add_version False \
    --create_checkpoint_symlink True \
    --train_type full \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy epoch \
    --save_steps 1
