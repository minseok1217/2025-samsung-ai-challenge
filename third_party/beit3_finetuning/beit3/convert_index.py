from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/data_vqaformat_yesno",
    tokenizer=tokenizer,
    annotation_data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/data_vqaformat_yesno/vqa",
)