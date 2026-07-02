python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/beit3.spm \
        --finetune /data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/beit3_large_patch16_384_coco_retrieval.pth \
        --data_path /data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/coco_retrieval \
        --eval \
        --dist_eval