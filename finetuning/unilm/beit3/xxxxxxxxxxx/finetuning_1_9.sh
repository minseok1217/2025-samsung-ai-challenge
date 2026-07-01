# finetuning.sh
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29501 run_beit3_finetuning.py \
    --model beit3_large_patch16_480 \
    --input_size 480 \
    --task vqav2 \
    --nb_classes 4 \
    --batch_size 4 \
    --layer_decay 1.0 \
    --lr 1e-5 \
    --update_freq 1 \
    --epochs 30 \
    --warmup_epochs 1 \
    --drop_path 0.15 \
    --sentencepiece_model /data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm \
    --finetune /data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3_large_indomain_patch16_224.pth \
    --data_path /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/VMCBench_as_Official_VQAv2_9_1/ \
    --output_dir /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/output_9_1_batch_4/ \
    --log_dir /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/log/ \
    --weight_decay 0.01 \
    --seed 42 \
    --save_ckpt_freq 5 \
    --task_head_lr_weight 20 \
    --opt_betas 0.9 0.98 \
    --clip_grad 1.0
    # --checkpoint_activations