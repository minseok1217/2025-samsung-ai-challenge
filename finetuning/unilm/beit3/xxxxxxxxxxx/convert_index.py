from datasets import RetrievalDataset, CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm")

# RetrievalDataset.make_flickr30k_dataset_index(
#     data_path="",
#     tokenizer=tokenizer,
#     karpathy_path="/path/to/your_data",
# )


CaptioningDataset.make_nocaps_captioning_dataset_index(
    data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w_2/nocap",
    # tokenizer=tokenizer,
)