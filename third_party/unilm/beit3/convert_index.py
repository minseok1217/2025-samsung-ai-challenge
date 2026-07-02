# from datasets import RetrievalDataset
# from transformers import XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm")

# RetrievalDataset.make_coco_dataset_index(
#     data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/coco_retrieval",
#     tokenizer=tokenizer,
# )

# from datasets import RetrievalDataset
# from transformers import XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm")

# RetrievalDataset.make_flickr30k_dataset_index(
#     data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/flickr30k",
#     tokenizer=tokenizer,
#     karpathy_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/flickr30k",
# )


from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm")

CaptioningDataset.make_coco_captioning_dataset_index(
    data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w",
    tokenizer=tokenizer,
)