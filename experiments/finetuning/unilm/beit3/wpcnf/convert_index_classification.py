from datasets import RetrievalDataset, BinaryClassificationDataset, CrossEntorpyDataset, LanguageModelMultiChoiceDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm")

LanguageModelMultiChoiceDataset.make_dataset_index(
    data_path="/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w/language_model_multichoice/",
    tokenizer=tokenizer,
    task="language_model_multichoice",
)