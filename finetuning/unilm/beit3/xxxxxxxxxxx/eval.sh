# 추론 시 사용할 데이터 경로를 새로 생성된 경로로 변경합니다.
# 이전 스크립트에서 `--data_path`가 학습 데이터 경로였는데, 이제 테스트 데이터 경로로 바꿉니다.

# finetuning_inference.sh (수정 예시)
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 29503 run_beit3_finetuning.py \
    --model beit3_large_patch16_480 \
    --input_size 480 \
    --task vqav2 \
    --nb_classes 4 \
    --batch_size 16 \
    --sentencepiece_model /data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm \
    --finetune /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/output_batch_16/checkpoint-best.pth \
    --data_path /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/VMCBench_as_Official_VQAv2/ \
    --output_dir /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/your_vmcbench_test_predictions/ \
    --eval \
    --dist_eval \
    --log_dir /data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/log_eval/ \
    --seed 42 \