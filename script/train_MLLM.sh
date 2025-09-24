# export CUDA_HOME=xxx

CUDA_VISIBLE_DEVICES=0 python3 src/train_new.py \
    --model_name_or_path ../path/to/Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type MLLM \
    --mode both \
    --data_dir ./data/train/ \
    --train_filename train.jsonl \
    --bf16 True \
    --output_dir ./outputs/Qwen2.5-VL-7B-Instruct-both-qkv-only \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to tensorboard \
