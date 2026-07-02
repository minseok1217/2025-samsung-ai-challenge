# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

import utils
from glossary import normalize_word
from randaug import RandomAugment

import pandas as pd
from PIL import Image
import re

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body

def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))

def _make_retrieval_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                for sent in item["sentences"]:
                    tokens = tokenizer.tokenize(sent["raw"])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": image_path, 
                            "text_segment": token_ids, 
                            "image_id": len(image_counter), 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_retrieval.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass

def _make_captioning_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                if item["split"] in ["train", "restval"]:
                    for sent in item["sentences"]:
                        tokens = tokenizer.tokenize(sent["raw"])
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        items.append({
                                "image_path": image_path, 
                                "text_segment": token_ids, 
                                "image_id": item["cocoid"], 
                        })
                else:
                    items.append({
                                "image_path": image_path, 
                                "text_segment": None, 
                                "image_id": item["cocoid"], 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_captioning.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass

def _make_nocaps_dataset_index(
        data_path,  
        split="val", 
):
    if split == "val":
        json_file = "nocaps_val_4500_captions.json"
    elif split == "test":
        json_file = "nocaps_test_image_info.json"
    nocaps_split_json_file = os.path.join(data_path, json_file)
    items = []
    image_counter = set()
    print("read %s" % nocaps_split_json_file)
    with open(nocaps_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            image_path = os.path.join(split, item["file_name"])
            items.append({
                "image_path": image_path, 
                "text_segment": None, 
                "image_id": item["id"], 
            })

            if image_path not in image_counter:
                image_counter.add(image_path)

    print("Find %d images and %d image-text pairs for nocaps dataset %s split !" % \
        (len(image_counter), len(items), split))
    index_file = os.path.join(data_path, "nocaps.%s.jsonl" % split)
    _write_data_into_jsonl(items, index_file)

class NLVR2Dataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("nlvr2.train.index.jsonl", )
        elif split == "val":
            return ("nlvr2.dev.index.jsonl", )
        elif split == "test":
            return ("nlvr2.test-P.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_json(preifx, json_file, tokenizer, index_file):
        items = []
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(preifx, str(data["directory"])) if "directory" in data else preifx
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                tokens = tokenizer.tokenize(data["sentence"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "text_segment": token_ids,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, nlvr_repo_path):
        cls.__preprocess_json(
            preifx="images/train", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/train.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("train")[0]), 
        )
        cls.__preprocess_json(
            preifx="dev", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/dev.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("val")[0]), 
        )
        cls.__preprocess_json(
            preifx="test1", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/test1.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("test")[0]), 
        )

class ImageNetDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("imagenet.train.index.jsonl", )
        elif split == "val":
            return ("imagenet.val.index.jsonl", )
        elif split == "test":
            return ("imagenet.val.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["label"] = item["label"]
        return data
    
    @staticmethod
    def _find_classes(dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _make_imagenet_index(data_path, index_path, data_path_prefix, class_to_idx, split):
        items = []
        index_file = os.path.join(index_path, f"imagenet.{split}.index.jsonl")
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(data_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    path = path.replace(data_path_prefix, "")
                    items.append({
                        "image_path": path,
                        "label": class_index,
                    })

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, train_data_path, val_data_path, index_path):
        data_path_prefix = train_data_path[:[x[0]==x[1] for x in zip(train_data_path, val_data_path)].index(0)]
        classes, class_to_idx = cls._find_classes(train_data_path)
        cls._make_imagenet_index(
             data_path=train_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="train",
        )
        cls._make_imagenet_index(
             data_path=val_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="val",
        )

class VQAv2Dataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)
        
        self.ans2label = ans2label
        self.label2ans = label2ans

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("vqa.train.jsonl", "vqa.trainable_val.jsonl")
        elif split == "val":
            return ("vqa.rest_val.jsonl", )
        elif split == "test":
            return ("vqa.test.jsonl", )
        elif split == "test-dev":
            return ("vqa.test-dev.jsonl", )            
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
            labels = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
                labels[l] = s
            data["labels"] = torch.FloatTensor(labels)
        else:
            data["qid"] = self.items[index]["qid"]
        return data

    @staticmethod
    def get_score(occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    # @classmethod
    # def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
    #     with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
    #         questions_train2014 = json.load(fp)["questions"]
    #     with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
    #         questions_val2014 = json.load(fp)["questions"]
    #     # with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp: # ms
    #     #     questions_test2015 = json.load(fp)["questions"]
    #     # with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
    #     #     questions_test_dev2015 = json.load(fp)["questions"]

    #     with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
    #         annotations_train2014 = json.load(fp)["annotations"]
    #     with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
    #         annotations_val2014 = json.load(fp)["annotations"]

    #     annotations = dict()

    #     for split, questions in zip(
    #         # ["train", "val", "test", "test-dev"], # ms
    #         # [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
    #         ["train", "val"],
    #         [questions_train2014, questions_val2014],
    #     ):
    #         _annot = defaultdict(dict)
    #         for q in questions:
    #             question_text = q["question"]
    #             tokens = tokenizer.tokenize(question_text)
    #             token_ids = tokenizer.convert_tokens_to_ids(tokens)

    #             assert q["question_id"] not in _annot[q["image_id"]]
    #             _annot[q["image_id"]][q["question_id"]] = {
    #                 "question": question_text, 
    #                 "token_ids": token_ids, 
    #             }

    #         annotations[split] = _annot

    #     all_major_answers = list()

    #     for split, annots in zip(
    #         ["train", "val"], [annotations_train2014, annotations_val2014],
    #     ):
    #         # _annot = annotations[split]
    #         for q in annots:
    #             all_major_answers.append(q["multiple_choice_answer"])

    #     all_major_answers = [normalize_word(word) for word in all_major_answers]
    #     counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    #     ans2label = {k: i for i, k in enumerate(counter.keys())}
    #     label2ans = list(counter.keys())

    #     for split, annots in zip(
    #         ["train", "val"], [annotations_train2014, annotations_val2014],
    #     ):
    #         _annot = annotations[split]
    #         for q in annots:
    #             answers = q["answers"]
    #             answer_count = {}
    #             for answer in answers:
    #                 answer_ = answer["answer"]
    #                 answer_count[answer_] = answer_count.get(answer_, 0) + 1

    #             labels = []
    #             scores = []
    #             for answer in answer_count:
    #                 if answer not in ans2label:
    #                     continue
    #                 labels.append(ans2label[answer])
    #                 score = cls.get_score(answer_count[answer])
    #                 scores.append(score)

    #             assert "labels" not in _annot[q["image_id"]][q["question_id"]]
    #             assert "question" in _annot[q["image_id"]][q["question_id"]]
    #             _annot[q["image_id"]][q["question_id"]]["labels"] = labels
    #             _annot[q["image_id"]][q["question_id"]]["scores"] = scores

    #     for split in ["train", "val"]:
    #         filtered_annot = dict()
    #         for ik, iv in annotations[split].items():
    #             new_q = dict()
    #             for qk, qv in iv.items():
    #                 if len(qv["labels"]) != 0:
    #                     new_q[qk] = qv
    #             if len(new_q) != 0:
    #                 filtered_annot[ik] = new_q
    #         annotations[split] = filtered_annot

    #     split2items = {}
    #     # for split in ["train", "val", "test", "test-dev"]: # ms
    #     for split in ["train", "val"]:
    #         annot = annotations[split]
    #         split_name = {
    #             "train": "train2014",
    #             "val": "val2014",
    #             # "test": "test2015", # ms
    #             # "test-dev": "test2015",
    #         }[split]
    #         paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
    #         random.shuffle(paths)
    #         annot_paths = [path for path in paths \
    #             if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

    #         if len(paths) == len(annot_paths):
    #             print("all images have caption annotations")
    #         else:
    #             print("not all images have caption annotations")
    #         print(len(paths), len(annot_paths), len(annot))

    #         items = []
    #         for path in annot_paths:
    #             iid = int(path.split("/")[-1].split("_")[-1][:-4])
    #             _annot = annotations[split][iid]
    #             for qid in _annot:
    #                 q = _annot[qid]
    #                 if split in ["train", "val"]:
    #                     labels = q["labels"]
    #                     scores = q["scores"]
    #                 else:
    #                     labels, scores = [], []

    #                 items.append({
    #                     "image_path": os.path.join(split_name, path.split('/')[-1]), 
    #                     "text_segment": q["token_ids"], 
    #                     "labels": labels, 
    #                     "scores": scores, 
    #                     "qid": qid, 
    #                 })
    #         split2items[split] = items

    #         _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

    #     # Following ViLT, we use 1000 images of the original val set as the final val set        
    #     val_image2items = defaultdict(list)
    #     for item in split2items["val"]:
    #         val_image2items[item["image_path"]].append(item)
        
    #     print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

    #     val_images = list(val_image2items.keys())
    #     random.shuffle(val_images)
    #     trainable_val = []
    #     rest_val = []
    #     for i, image_id in enumerate(val_images):
    #         if i < 1000:
    #             rest_val += val_image2items[image_id]
    #         else:
    #             trainable_val += val_image2items[image_id]
        
    #     _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
    #     _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

    #     with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
    #         for ans in ans2label:
    #             to_json = {
    #                 "answer": ans, 
    #                 "label": ans2label[ans]
    #             }
    #             writer.write("%s\n" % json.dumps(to_json))

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
            ["train", "val"],
            [questions_train2014, questions_val2014],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                # VMCBench의 question_id와 image_id가 VQAv2와 다를 수 있으므로, assert는 조심
                # VMCBench 변환 스크립트에서 image_id와 question_id를 VMCBench index로 사용했으므로 괜찮을 것
                assert q["question_id"] not in _annot[q["image_id"]] 
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text, 
                    "token_ids": token_ids, 
                }

            annotations[split] = _annot

        # --- VMCBench를 위한 ans2label/label2ans 정의 (수정된 부분 시작) ---
        # 4지선다형이므로, 0, 1, 2, 3이 각각 'A', 'B', 'C', 'D'의 인덱스라고 가정
        # 실제 레이블은 'answer_label' 필드에서 직접 가져올 것이므로, ans2label은 4개 고정
        ans2label = {'A': 0, 'B': 1, 'C': 2, 'D': 3} # 또는 실제 답변 텍스트로 매핑해도 되지만,
                                                    # 어차피 answer_label을 쓸 것이므로 의미 없을 수 있음
        label2ans = ['A_choice', 'B_choice', 'C_choice', 'D_choice'] # 더미 또는 실제 선택지 내용으로 채움
                                                                     # 이 부분은 VQA 평가에 사용되는 것이므로,
                                                                     # VMCBench 평가 방식에 따라 조정 필요
                                                                     # 여기서는 '0 items' 해결에 집중
        # --- VMCBench를 위한 ans2label/label2ans 정의 (수정된 부분 끝) ---


        for split, annots_list in zip( # annots_list로 변수명 변경 (confusion 피하기 위해)
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q_annot in annots_list: # q_annot으로 변수명 변경
                # 여기서 VMCBench의 answer_label을 직접 사용
                # VMCBench 변환 스크립트에서 'answer_label' 필드를 추가했으므로 사용 가능
                if "answer_label" not in q_annot:
                    # 경고 메시지 또는 에러 처리 (만약 answer_label이 없으면 문제가 됨)
                    print(f"Warning: 'answer_label' not found in annotation for qid {q_annot['question_id']}. Skipping.")
                    continue

                # VQAv2의 여러 답변 및 점수 계산 로직 대신, VMCBench의 answer_label 사용
                # VMCBench는 4지선다형이므로, labels와 scores는 단일 값으로 구성됨
                labels = [q_annot["answer_label"]] # answer_label은 이미 0, 1, 2, 3 중 하나
                
                # scores는 VQA의 채점 방식이므로, VMCBench의 경우 1.0으로 고정하거나 0으로 설정
                # 혹은 모델 예측 시 사용될 더미 점수
                scores = [1.0] # 4지선다형은 정답 하나이므로 점수 1.0

                # 이 부분의 assert는 VMCBench의 question_id와 image_id가
                # questions JSON에서 가져온 것과 매칭되는지 확인하는 것이므로 유지
                assert "labels" not in _annot[q_annot["image_id"]][q_annot["question_id"]]
                assert "question" in _annot[q_annot["image_id"]][q_annot["question_id"]]

                _annot[q_annot["image_id"]][q_annot["question_id"]]["labels"] = labels
                _annot[q_annot["image_id"]][q_annot["question_id"]]["scores"] = scores

        # 이 필터링 로직은 그대로 유지 (labels가 비어있는 샘플은 제외)
        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0: # 이제 labels가 비는 경우가 없어질 것임 (answer_label 덕분에)
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
            }[split]
            
            # `glob`을 사용하려면 `import glob`을 파일 상단에 추가해야 합니다.
            # 또한, `data_path`가 이미지 파일들의 최상위 폴더여야 합니다.
            # 당신의 `VMCBench_as_Official_VQAv2` 폴더 아래에 `train2014`가 직접 있으므로,
            # `data_path`를 `VMCBench_as_Official_VQAv2`로 설정해야 합니다.
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.shuffle(paths)
            
            # VMCBench 변환 스크립트에서 image_id를 VMCBench의 index를 사용했으므로
            # path.split("/")[-1].split("_")[-1][:-4]로 올바르게 파싱될 것
            annot_paths = [path for path in paths \
                if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have corresponding question/annotation pairs") # 메시지 수정
            else:
                print(f"Warning: Not all images ({len(paths)}) have corresponding question/annotation pairs ({len(annot_paths)}).")
            print(f"DEBUG: Found {len(paths)} images, {len(annot_paths)} images with annotations, {len(annot)} unique image_ids in annotations for {split} split.") # 디버깅 메시지 추가

            items = []
            for path in annot_paths:
                iid = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    # VMCBench의 labels와 scores는 이미 위에서 설정했으므로, if split in ["train", "val"] 조건이 필요 없음
                    labels = q["labels"]
                    scores = q["scores"]

                    items.append({
                        "image_path": os.path.join(split_name, path.split('/')[-1]), 
                        "text_segment": q["token_ids"], 
                        "labels": labels, # [answer_label] 형태
                        "scores": scores, # [1.0] 형태
                        "qid": qid, 
                    })
            split2items[split] = items

            # _write_data_into_jsonl 함수가 정의되어 있어야 합니다.
            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))
            print(f"Write {os.path.join(data_path, 'vqa.%s.jsonl' % split)} with {len(items)} items !")

        # Following ViLT, we use 1000 images of the original val set as the final val set 
        # 이 부분은 VMCBench의 'val' 세트를 다시 'trainable_val'과 'rest_val'로 나누는 로직
        # VMCBench에 그대로 적용할지 결정해야 합니다. (일반적으로는 train, val만 있으면 됨)
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_path in enumerate(val_images): # image_id 대신 image_path 사용
            if i < 1000: # 1000개 이미지는 rest_val (평가용)
                rest_val.extend(val_image2items[image_path]) # extend로 리스트 합치기
            else: # 나머지 trainable_val (훈련용 검증)
                trainable_val.extend(val_image2items[image_path])
        
        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))
        print(f"Write {os.path.join(data_path, 'vqa.trainable_val.jsonl')} with {len(trainable_val)} items !")
        print(f"Write {os.path.join(data_path, 'vqa.rest_val.jsonl')} with {len(rest_val)} items !")

        # answer2label.txt 파일 생성
        # 이 파일은 VQAv2 평가 스크립트가 필요로 하는 답변 사전입니다.
        # VMCBench의 경우, A, B, C, D에 해당하는 실제 답변 텍스트로 사전을 구성하는 것이 좋을 수 있습니다.
        # 예를 들어, {'질문1_A_답변':0, '질문1_B_답변':1, ...} -> 이렇게 하면 사전 크기가 너무 커짐.
        # 대신 4지선다형이므로, 0, 1, 2, 3 레이블을 직접 answer2label에 매핑하는 것이 합리적입니다.
        # 예를 들어, VQAv2처럼 normalize_word가 아니라 'A', 'B', 'C', 'D'를 키로 사용할 수도 있습니다.
        # 여기서는 VMCBench의 'A', 'B', 'C', 'D' 매핑을 사용합니다.
        vmcbench_ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans_char, label_idx in vmcbench_ans_map.items():
                to_json = {
                    "answer": ans_char, # 실제 답변 텍스트 대신 'A', 'B', 'C', 'D'를 키로 사용
                    "label": label_idx
                }
                writer.write("%s\n" % json.dumps(to_json))

class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

    @staticmethod
    def make_flickr30k_dataset_index(data_path, tokenizer, karpathy_path):

        with open(os.path.join(karpathy_path, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        split2items = defaultdict(list)
        split2images = defaultdict(set)

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])
            split = each_item["split"]

            for text_segment in each_item["sentences"]:
                tokens = tokenizer.tokenize(text_segment["raw"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                split2items[split].append({
                    "image_path": image_path, 
                    "text_segment": token_ids, 
                    "image_id": len(split2images[split]), 
                })

            assert each_item["filename"] not in split2images[split]
            split2images[split].add(each_item["filename"])

        for split in split2items:
            print("%d images and %d image-text pairs!" % (len(split2images[split]), len(split2items[split])))
            _write_data_into_jsonl(split2items[split], os.path.join(data_path, "flickr30k.%s.jsonl" % split))

    @staticmethod
    def make_coco_dataset_index(data_path, tokenizer):
        # _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        # _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

class CaptioningDataset(BaseDataset):

    def __init__(self, data_path, split, transform, 
                tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task, 
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob
    
    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")

class BinaryClassificationDataset(BaseDataset):
    def __init__(self, data_path, split, transform, tokenizer, num_max_bpe_tokens, task):
        super().__init__(
            data_path=data_path, split=split, transform=transform, 
            tokenizer=tokenizer, num_max_bpe_tokens=num_max_bpe_tokens, task=task
        )

    @staticmethod
    def get_index_files(split, task="binary_classification"):
        return (f"{task}.{split}.jsonl", )

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        # 이제 인덱스 파일에 "label"과 "image_id"가 모두 있으므로 정상적으로 작동합니다.
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_csv(data_path, csv_filename, tokenizer, index_file):
        items = []
        csv_path = os.path.join(data_path, csv_filename)
        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.lower() for col in df.columns]
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found. Skipping.")
            return

        for index, row in df.iterrows():
            text = row.get('text', '').strip()
            image_path = row.get('image_path', '')
            if not text or not image_path:
                continue

            last_word = text.split(' ')[-1]
            label = 1 if last_word == "true" else 0
            
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # ▼▼▼ 핵심 수정 부분 ▼▼▼
            # "identifier"를 "image_id"로 변경합니다.
            items.append({
                "image_path": image_path,
                "text_segment": token_ids,
                "label": label,
                # "identifier"가 아니라 "image_id"로 저장되어야 합니다.
                "image_id": row.get("id", f"item_{index}"), 
            })
            # ▲▲▲ 수정 완료 ▲▲▲
            
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, task="binary_classification"):
        print(f"🚀 Creating index files for '{task}'...")
        for split_name in ["train", "val", "test"]:
            cls.__preprocess_csv(
                data_path=data_path,
                csv_filename=f"{split_name}.csv",
                tokenizer=tokenizer, 
                index_file=os.path.join(data_path, cls.get_index_files(split_name, task)[0])
            )
        print("✅ Index file creation complete.")

class CrossEntorpyDataset(BaseDataset):
    def __init__(self, data_path, split, transform, tokenizer, num_max_bpe_tokens, task):
        super().__init__(
            data_path=data_path, 
            split=split, 
            transform=transform, 
            tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, 
            task=task
        )

    @staticmethod
    def get_index_files(split, task="cross_entropy"):
        return (f"{task}.{split}.jsonl", )

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_csv(data_path, csv_filename, tokenizer, index_file):
        items = []
        csv_path = os.path.join(data_path, csv_filename)
        try:
            df = pd.read_csv(csv_path)
            # 표준화를 위해 컬럼 이름을 소문자로 통일합니다.
            df.columns = [col.lower() for col in df.columns]
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found. Skipping.")
            return

        # 정답 문자를 숫자 레이블로 변환하기 위한 맵
        answer_to_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

        for index, row in df.iterrows():
            try:
                # ▼▼▼ 핵심 수정 부분 ▼▼▼
                # 1. 각 행에서 'text'와 'answer' 컬럼의 값을 명확하게 가져옵니다.
                text = row.get('text', '')
                answer_letter = str(row.get('answer', '')).upper()
                
                # 텍스트가 비어있으면 건너뜁니다.
                if not text:
                    continue
                
                # 2. 정답 문자를 숫자 레이블로 변환합니다.
                label = answer_to_label_map.get(answer_letter, -1)
                
                # answer가 A,B,C,D 중 하나가 아니면 해당 행은 건너뜁니다.
                if label == -1:
                    continue
                # ▲▲▲ 수정 완료 ▲▲▲
                
                # 텍스트를 토큰 ID로 변환
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                items.append({
                    "image_path": row["img_path"],
                    "text_segment": token_ids, # 이제 각 행마다 다른 token_ids가 저장됩니다.
                    "label": label,
                    "identifier": row.get("id", f"item_{index}"),
                })
            except KeyError as e:
                print(f"Warning: Missing column {e} in {csv_filename}. Skipping row {index}.")
                continue
                
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, task="cross_entropy"):
        """
        train.csv, val.csv, test.csv를 읽어 인덱스 파일을 생성합니다.
        """
        print(f"🚀 Creating index files for '{task}'...")
        
        for split_name in ["train", "val", "test"]:
            cls.__preprocess_csv(
                data_path=data_path,
                csv_filename=f"{split_name}.csv",
                tokenizer=tokenizer, 
                index_file=os.path.join(data_path, cls.get_index_files(split_name, task)[0])
            )
        print("✅ Index file creation complete.")

class LanguageModelTFDataset(BaseDataset):
    """
    [수정됨] 텍스트를 전처리하여 실제 내용의 마지막 토큰을 예측하는 데이터셋 클래스.
    """
    def __init__(self, data_path, split, transform, 
                 tokenizer, num_max_bpe_tokens, task="language_modeling"):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task,
        )
        self.mask_token_id = tokenizer.mask_token_id
        
        index_file = self.get_index_files(split, task)[0]
        index_file_path = os.path.join(data_path, index_file)
        
        if not os.path.exists(index_file_path):
            raise FileNotFoundError(f"Index file not found: {index_file_path}.")
            
        self.items = []
        with open(index_file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_index_files(split, task=None):
        # 이 파일 이름은 convert 스크립트에서 생성한 파일 이름과 일치해야 합니다.
        return (f"{task}.{split}.jsonl", )

    def _load_image(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        entry = self.items[index]
        
        # --- 1. 이미지 처리 ---
        image = self._load_image(os.path.join(self.data_path, entry["image_path"]))

        # --- 2. 텍스트 처리 ---
        raw_text = entry.get("raw")
        if raw_text is None:
            raw_text = self.tokenizer.decode(entry["caption"])

        # 'this statement is' 또는 'The answer is'를 기준으로 텍스트를 분리
        parts = re.split(r'\s*(?:this statement is|The answer is)\s*', raw_text, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            # 분리 실패 시, 이 데이터는 건너뛰도록 빈 딕셔너리 반환 (DataLoader에서 처리)
            return {} 

        main_content = parts[0].strip()
        label_word = parts[1].strip().lower()

        # --- 3. MLM 프롬프트 생성 ---
        # `main_content` 뒤에 고정된 프롬프트와 [MASK] 토큰을 추가합니다.
        prompt_text = f"{main_content} this statement is [MASK]"
        
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        text_ids = torch.full((self.num_max_bpe_tokens,), self.tokenizer.pad_token_id, dtype=torch.long)
        prompt_len = min(len(prompt_tokens), self.num_max_bpe_tokens)
        text_ids[:prompt_len] = torch.tensor(prompt_tokens[:prompt_len], dtype=torch.long)
        padding_mask = (text_ids == self.tokenizer.pad_token_id)
        
        # --- 4. 선택지 및 정답 인덱스 구성 ---
        # 선택지는 항상 'true'와 'false'로 고정됩니다.
        choices = ["true", "false"]
        choice_token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
        
        # 실제 레이블 단어가 'true'이면 정답 인덱스는 0, 'false'이면 1이 됩니다.
        true_answer_index = 0 if 'true' in label_word else 1
        
        # --- 5. 핸들러가 요구하는 형식으로 반환 ---
        return {
            "image": image,
            "text_ids": text_ids,
            "padding_mask": padding_mask,
            "choice_token_ids": torch.tensor(choice_token_ids, dtype=torch.long),
            "true_answer_index": torch.tensor(true_answer_index, dtype=torch.long),
            "image_id": entry.get("image_id"),
        }

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, task="language_modeling"):
        """
        [수정됨] COCO 캡션 데이터셋을 위한 인덱스 파일을 생성하는 메인 함수.
        train과 restval 데이터를 합쳐서 한 번에 저장합니다.
        """
        print(f"🚀 Creating index files for '{task}'...")
        original_json_file = "dataset_coco.json"

        # ▼▼▼ 핵심 수정 부분 ▼▼▼

        # 1. train + restval 데이터를 담을 빈 리스트를 생성합니다.
        train_items = []

        # 2. train과 restval을 처리하고 결과를 리스트에 추가합니다.
        for split_name in ["train", "restval"]:
            print(f"Processing '{split_name}' split and collecting items...")
            # __preprocess_captions_json이 아이템 리스트를 반환하도록 수정했다고 가정합니다.
            # (아래 __preprocess_captions_json 수정본 참고)
            items = cls.__preprocess_captions_json(
                data_path=data_path,
                original_json_file=original_json_file,
                split_name=split_name,
                tokenizer=tokenizer,
            )
            train_items.extend(items)
        
        # 3. 수집된 모든 train 데이터를 파일에 한 번만 씁니다.
        train_index_file_path = os.path.join(data_path, cls.get_index_files("train", task)[0])
        _write_data_into_jsonl(train_items, train_index_file_path)
        print(f"Write {train_index_file_path} with {len(train_items)} items !")

        # 4. val과 test는 개별적으로 처리합니다.
        for split_name in ["val", "test"]:
            print(f"Processing '{split_name}' split...")
            items = cls.__preprocess_captions_json(
                data_path=data_path,
                original_json_file=original_json_file,
                split_name=split_name,
                tokenizer=tokenizer,
            )
            index_file_path = os.path.join(data_path, cls.get_index_files(split_name, task)[0])
            _write_data_into_jsonl(items, index_file_path)
            print(f"Write {index_file_path} with {len(items)} items !")
        
        # ▲▲▲ 수정 완료 ▲▲▲
        
        print("✅ Index file creation complete.")

    @staticmethod
    def __preprocess_captions_json(data_path, original_json_file, split_name, tokenizer):
        """
        [수정됨] 데이터를 파일에 쓰는 대신, 처리된 아이템 리스트를 반환합니다.
        """
        with open(os.path.join(data_path, original_json_file), 'r') as f:
            full_data = json.load(f)

        if split_name == "train":
            img_path = "train2014"
        elif split_name == "val":
            img_path = "val2014"
        elif split_name == "test":
            img_path = "test_2014"

        items = []
        for item in full_data['images']:
            if item['split'] == split_name:
                for sentence in item['sentences']:
                    tokens = tokenizer.tokenize(sentence['raw'])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                        "image_path": os.path.join(img_path, item['filename']),
                        "image_id": item['cocoid'],
                        "caption": token_ids,
                        "raw": sentence['raw'],
                    })
        # 파일에 쓰는 대신, 결과 리스트를 반환
        return items

class LanguageModelMultiChoiceDataset(BaseDataset):
    def __init__(self, data_path, split, transform, 
                 tokenizer, num_max_bpe_tokens, task="language_model_multichoice"):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.task = task
        
        index_file = self.get_index_files(split, task)[0]
        index_file_path = os.path.join(data_path, index_file)
        
        if not os.path.exists(index_file_path):
            raise FileNotFoundError(f"Index file not found: {index_file_path}.")
            
        self.items = []
        with open(index_file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_index_files(split, task=None):
        # 이 파일 이름은 convert 스크립트에서 생성한 파일 이름과 일치해야 합니다.
        return (f"{task}.{split}.jsonl", )
    
    def set_index_files_epoch(self, epoch):
        self.index_files = self.get_index_files_epoch(self.split, self.task, epoch)

    def get_index_files_epoch(self, split, task=None, epoch=0):
        return (f"{task}.{split}.epoch{epoch}.jsonl", )
    
    def get_index_files_epoch_2(self, split, task=None, epoch=0):
        print(f"DEBUG: get_index_files_epoch_2 called with split={split}, task={task}, epoch={epoch}")
        return f"{task}.{split}.epoch{epoch}.jsonl"

    def _load_image(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        entry = self.items[index]
        
        # --- 1. 이미지 처리 ---
        image = self._load_image(os.path.join(self.data_path, entry["image_path"]))

        # --- 2. 텍스트 처리 ---
        raw_text = entry.get("raw")
        if raw_text is None:
            raw_text = self.tokenizer.decode(entry["caption"])

        parts = re.split(r'\s*(?:this statement is|The answer is|Answer:)\s*', raw_text, flags=re.IGNORECASE)

        # print(f"DEBUG: split parts = {parts}")

        if len(parts) < 2:
            return {} 

        main_content = parts[0].strip()
        label_info = parts[1].strip().upper() # label_word를 대문자로 변경

        # --- 3. MLM 프롬프트 생성 ---
        # 이제 프롬프트는 항상 질문+답변 형식이 됩니다.
        prompt_text = f"{main_content} Answer: [MASK]."
        
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        text_ids = torch.full((self.num_max_bpe_tokens,), self.tokenizer.pad_token_id, dtype=torch.long)
        prompt_len = min(len(prompt_tokens), self.num_max_bpe_tokens)
        text_ids[:prompt_len] = torch.tensor(prompt_tokens[:prompt_len], dtype=torch.long)
        padding_mask = (text_ids == self.tokenizer.pad_token_id)
        
        # ▼▼▼ 핵심 수정 부분 ▼▼▼
        # --- 4. 선택지 및 정답 인덱스 구성 ---
        choices = ["A", "B", "C", "D"]
        choice_token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
        
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # label_info (예: "A", "B", "TRUE", "FALSE")에서 정답 문자를 찾습니다.
        # 정답이 A,B,C,D 중 하나가 아니면 이 데이터는 건너뜁니다.
        true_answer_letter = None
        for letter in answer_map.keys():
            if letter in label_info:
                true_answer_letter = letter
                break
                
        if true_answer_letter is None:
            return {} # A,B,C,D 정답이 없으면 학습에서 제외

        true_answer_index = answer_map[true_answer_letter]
        # ▲▲▲ 수정 완료 ▲▲▲
        
        # --- 5. 핸들러가 요구하는 형식으로 반환 ---
        return {
            "image": image,
            "text_ids": text_ids,
            "padding_mask": padding_mask,
            "choice_token_ids": torch.tensor(choice_token_ids, dtype=torch.long),
            "true_answer_index": torch.tensor(true_answer_index, dtype=torch.long),
            "image_id": entry.get("image_id"),
        }

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, task="language_modeling", random_=False):
        if random_:
            for i in range(10):
                print(f"🚀 Creating index files for '{task}'... i : {i}")
                original_json_file = f"dataset_coco_{i}.json"

                # ▼▼▼ 핵심 수정 부분 ▼▼▼

                # 1. train + restval 데이터를 담을 빈 리스트를 생성합니다.
                train_items = []

                # 2. train과 restval을 처리하고 결과를 리스트에 추가합니다.
                for split_name in ["train", "restval"]:
                    print(f"Processing '{split_name}' split and collecting items...")
                    # __preprocess_captions_json이 아이템 리스트를 반환하도록 수정했다고 가정합니다.
                    # (아래 __preprocess_captions_json 수정본 참고)
                    items = cls.__preprocess_captions_json(
                        data_path=data_path,
                        original_json_file=original_json_file,
                        split_name=split_name,
                        tokenizer=tokenizer,
                    )
                    train_items.extend(items)
                
                # 3. 수집된 모든 train 데이터를 파일에 한 번만 씁니다.
                print(f"Writing file name ")
                print(cls.get_index_files(cls, split = "train", task = task, epoch = i))
                train_index_file_path = os.path.join(data_path, cls.get_index_files(cls, split = "train", task = task, epoch = i))
                _write_data_into_jsonl(train_items, train_index_file_path)
                print(f"Write {train_index_file_path} with {len(train_items)} items !")

            # 4. val과 test는 개별적으로 처리합니다.
            for split_name in ["val", "test"]:
                print(f"Processing '{split_name}' split...")
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                index_file_path = os.path.join(data_path, cls.get_index_files(split_name, task)[0])
                _write_data_into_jsonl(items, index_file_path)
                print(f"Write {index_file_path} with {len(items)} items !")
            
            # ▲▲▲ 수정 완료 ▲▲▲
            print("✅ Index file creation complete.")

        else:
            print(f"🚀 Creating index files for '{task}'...")
            original_json_file = "dataset_coco.json"

            # ▼▼▼ 핵심 수정 부분 ▼▼▼

            # 1. train + restval 데이터를 담을 빈 리스트를 생성합니다.
            train_items = []

            # 2. train과 restval을 처리하고 결과를 리스트에 추가합니다.
            for split_name in ["train", "restval"]:
                print(f"Processing '{split_name}' split and collecting items...")
                # __preprocess_captions_json이 아이템 리스트를 반환하도록 수정했다고 가정합니다.
                # (아래 __preprocess_captions_json 수정본 참고)
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                train_items.extend(items)
            
            # 3. 수집된 모든 train 데이터를 파일에 한 번만 씁니다.
            train_index_file_path = os.path.join(data_path, cls.get_index_files("train", task)[0])
            _write_data_into_jsonl(train_items, train_index_file_path)
            print(f"Write {train_index_file_path} with {len(train_items)} items !")

            # 4. val과 test는 개별적으로 처리합니다.
            for split_name in ["val", "test"]:
                print(f"Processing '{split_name}' split...")
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                index_file_path = os.path.join(data_path, cls.get_index_files(split_name, task)[0])
                _write_data_into_jsonl(items, index_file_path)
                print(f"Write {index_file_path} with {len(items)} items !")
            
            # ▲▲▲ 수정 완료 ▲▲▲
            
            print("✅ Index file creation complete.")

    @staticmethod
    def __preprocess_captions_json(data_path, original_json_file, split_name, tokenizer):
        """
        [수정됨] 데이터를 파일에 쓰는 대신, 처리된 아이템 리스트를 반환합니다.
        """
        with open(os.path.join(data_path, original_json_file), 'r') as f:
            full_data = json.load(f)

        if split_name == "train":
            img_path = "train2014"
        elif split_name == "val":
            img_path = "val2014"
        elif split_name == "test":
            img_path = "test_2014"

        items = []
        for item in full_data['images']:
            if item['split'] == split_name:
                for sentence in item['sentences']:
                    tokens = tokenizer.tokenize(sentence['raw'])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                        "image_path": os.path.join(img_path, item['filename']),
                        "image_id": item['cocoid'],
                        "caption": token_ids,
                        "raw": sentence['raw'],
                    })
        # 파일에 쓰는 대신, 결과 리스트를 반환
        return items

class LanguageModelMultiChoicePrefixDataset(BaseDataset):
    def __init__(self, data_path, split, transform, 
                 tokenizer, num_max_bpe_tokens, task="language_model_multichoice"):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.task = task
        
        index_file = self.get_index_files(split, task)[0]
        index_file_path = os.path.join(data_path, index_file)
        
        if not os.path.exists(index_file_path):
            raise FileNotFoundError(f"Index file not found: {index_file_path}.")
            
        self.items = []
        with open(index_file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_index_files(split, task=None):
        # 이 파일 이름은 convert 스크립트에서 생성한 파일 이름과 일치해야 합니다.
        return (f"{task}.{split}.jsonl", )
    
    def set_index_files_epoch(self, epoch):
        self.index_files = self.get_index_files_epoch(self.split, self.task, epoch)

    def get_index_files_epoch(self, split, task=None, epoch=0):
        return (f"{task}.{split}.epoch{epoch}.jsonl", )
    
    def get_index_files_epoch_2(self, split, task=None, epoch=0):
        print(f"DEBUG: get_index_files_epoch_2 called with split={split}, task={task}, epoch={epoch}")
        return f"{task}.{split}.epoch{epoch}.jsonl"

    def _load_image(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        entry = self.items[index]
        
        # --- 1. 이미지 처리 ---
        image = self._load_image(os.path.join(self.data_path, entry["image_path"]))

        # --- 2. 텍스트 처리 ---
        raw_text = entry.get("raw")
        if raw_text is None:
            raw_text = self.tokenizer.decode(entry["caption"])

        parts = re.split(r'\s*(?:this statement is|The answer is | Answer:)\s*', raw_text, flags=re.IGNORECASE)

        # print(f"DEBUG: split parts = {parts}")

        if len(parts) < 2:
            return {} 

        main_content = parts[0].strip()
        label_info = parts[1].strip().upper() # label_word를 대문자로 변경

        # --- 3. MLM 프롬프트 생성 ---
        # 이제 프롬프트는 항상 질문+답변 형식이 됩니다.
        prompt_text = f"{main_content} \n Answer: [MASK]."
        
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        text_ids = torch.full((self.num_max_bpe_tokens,), self.tokenizer.pad_token_id, dtype=torch.long)
        prompt_len = min(len(prompt_tokens), self.num_max_bpe_tokens)
        text_ids[:prompt_len] = torch.tensor(prompt_tokens[:prompt_len], dtype=torch.long)
        padding_mask = (text_ids == self.tokenizer.pad_token_id)
        
        # ▼▼▼ 핵심 수정 부분 ▼▼▼
        # --- 4. 선택지 및 정답 인덱스 구성 ---
        choices = ["A", "B", "C", "D"]
        choice_token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
        
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # label_info (예: "A", "B", "TRUE", "FALSE")에서 정답 문자를 찾습니다.
        # 정답이 A,B,C,D 중 하나가 아니면 이 데이터는 건너뜁니다.
        true_answer_letter = None
        for letter in answer_map.keys():
            if letter in label_info:
                true_answer_letter = letter
                break
                
        if true_answer_letter is None:
            return {} # A,B,C,D 정답이 없으면 학습에서 제외

        true_answer_index = answer_map[true_answer_letter]
        # ▲▲▲ 수정 완료 ▲▲▲
        
        # --- 5. 핸들러가 요구하는 형식으로 반환 ---
        return {
            "image": image,
            "text_ids": text_ids,
            "padding_mask": padding_mask,
            "choice_token_ids": torch.tensor(choice_token_ids, dtype=torch.long),
            "true_answer_index": torch.tensor(true_answer_index, dtype=torch.long),
            "image_id": entry.get("image_id"),
        }

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, task="language_modeling", random_=False):
        if random_:
            for i in range(10):
                print(f"🚀 Creating index files for '{task}'... i : {i}")
                original_json_file = f"dataset_coco_{i}.json"

                # ▼▼▼ 핵심 수정 부분 ▼▼▼

                # 1. train + restval 데이터를 담을 빈 리스트를 생성합니다.
                train_items = []

                # 2. train과 restval을 처리하고 결과를 리스트에 추가합니다.
                for split_name in ["train", "restval"]:
                    print(f"Processing '{split_name}' split and collecting items...")
                    # __preprocess_captions_json이 아이템 리스트를 반환하도록 수정했다고 가정합니다.
                    # (아래 __preprocess_captions_json 수정본 참고)
                    items = cls.__preprocess_captions_json(
                        data_path=data_path,
                        original_json_file=original_json_file,
                        split_name=split_name,
                        tokenizer=tokenizer,
                    )
                    train_items.extend(items)
                
                # 3. 수집된 모든 train 데이터를 파일에 한 번만 씁니다.
                print(f"Writing file name ")
                print(cls.get_index_files_epoch_2(cls, split = "train", task = task, epoch = i))
                train_index_file_path = os.path.join(data_path, cls.get_index_files_epoch_2(cls, split = "train", task = task, epoch = i))
                _write_data_into_jsonl(train_items, train_index_file_path)
                print(f"Write {train_index_file_path} with {len(train_items)} items !")

            # 4. val과 test는 개별적으로 처리합니다.
            for split_name in ["val", "test"]:
                print(f"Processing '{split_name}' split...")
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                index_file_path = os.path.join(data_path, cls.get_index_files(split_name, task)[0])
                _write_data_into_jsonl(items, index_file_path)
                print(f"Write {index_file_path} with {len(items)} items !")
            
            # ▲▲▲ 수정 완료 ▲▲▲
            print("✅ Index file creation complete.")

        else:
            print(f"🚀 Creating index files for '{task}'...")
            original_json_file = "dataset_coco.json"

            # ▼▼▼ 핵심 수정 부분 ▼▼▼

            # 1. train + restval 데이터를 담을 빈 리스트를 생성합니다.
            train_items = []

            # 2. train과 restval을 처리하고 결과를 리스트에 추가합니다.
            for split_name in ["train", "restval"]:
                print(f"Processing '{split_name}' split and collecting items...")
                # __preprocess_captions_json이 아이템 리스트를 반환하도록 수정했다고 가정합니다.
                # (아래 __preprocess_captions_json 수정본 참고)
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                train_items.extend(items)
            
            # 3. 수집된 모든 train 데이터를 파일에 한 번만 씁니다.
            train_index_file_path = os.path.join(data_path, cls.get_index_files("train", task)[0])
            _write_data_into_jsonl(train_items, train_index_file_path)
            print(f"Write {train_index_file_path} with {len(train_items)} items !")

            # 4. val과 test는 개별적으로 처리합니다.
            for split_name in ["val", "test"]:
                print(f"Processing '{split_name}' split...")
                items = cls.__preprocess_captions_json(
                    data_path=data_path,
                    original_json_file=original_json_file,
                    split_name=split_name,
                    tokenizer=tokenizer,
                )
                index_file_path = os.path.join(data_path, cls.get_index_files(split_name, task)[0])
                _write_data_into_jsonl(items, index_file_path)
                print(f"Write {index_file_path} with {len(items)} items !")
            
            # ▲▲▲ 수정 완료 ▲▲▲
            
            print("✅ Index file creation complete.")

    @staticmethod
    def __preprocess_captions_json(data_path, original_json_file, split_name, tokenizer):
        """
        [수정됨] 데이터를 파일에 쓰는 대신, 처리된 아이템 리스트를 반환합니다.
        """
        with open(os.path.join(data_path, original_json_file), 'r') as f:
            full_data = json.load(f)

        if split_name == "train":
            img_path = "train2014"
        elif split_name == "val":
            img_path = "val2014"
        elif split_name == "test":
            img_path = "test_2014"

        items = []
        for item in full_data['images']:
            if item['split'] == split_name:
                for sentence in item['sentences']:
                    tokens = tokenizer.tokenize(sentence['raw'])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                        "image_path": os.path.join(img_path, item['filename']),
                        "image_id": item['cocoid'],
                        "caption": token_ids,
                        "raw": sentence['raw'],
                    })
        # 파일에 쓰는 대신, 결과 리스트를 반환
        return items

task2dataset = {
    "nlvr2": NLVR2Dataset, 
    "vqav2": VQAv2Dataset, 
    "flickr30k": RetrievalDataset, 
    "coco_retrieval": RetrievalDataset,  
    "coco_captioning": CaptioningDataset,
    "nocaps": CaptioningDataset,
    "imagenet": ImageNetDataset,
    "binary_classification": BinaryClassificationDataset,
    "cross_entropy": CrossEntorpyDataset,
    "language_model_tf": LanguageModelTFDataset,
    "language_model_multichoice": LanguageModelMultiChoiceDataset,
    "language_model_multichoice_prefix" : LanguageModelMultiChoicePrefixDataset,
}


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def build_imagenet_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataset_by_split(args, split, is_train=True, epoch=None):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, tokenizer=tokenizer, 
        num_max_bpe_tokens=args.num_max_bpe_tokens, 
        task=args.task, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    if epoch is not None:
        dataset.set_index_files_epoch(epoch)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=True)

def create_downstream_dataset_only_train(args, epoch=0):
    return create_dataset_by_split(args, split="train", is_train=True, epoch=epoch)