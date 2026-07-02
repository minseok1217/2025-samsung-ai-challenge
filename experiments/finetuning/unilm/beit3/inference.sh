python -m torch.distributed.run --nproc_per_node=1 run_beit3_inference.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 1 \
        --sentencepiece_model beit3.spm \
        --finetune beit3_large_patch16_480_coco_captioning.pth \
        --data_path . \
        --eval \
        --dist_eval