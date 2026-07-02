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
from randaug import RandomAugment # randaug.py가 같은 디렉토리에 있다고 가정

import logging
logger = logging.getLogger(__name__)


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
            
            # --- START OF MODIFICATION FOR JSON/JSONL LOADING ---
            try:
                # 먼저 JSONL 형식으로 읽어보려 시도
                with open(index_file, mode="r", encoding="utf-8") as reader:
                    for line in reader:
                        data = json.loads(line)
                        items.append(data)
                print(f"Loaded {len(items) - offset} image-text pairs from {index_file} (JSONL format). ")
            except json.JSONDecodeError:
                # JSONL이 아니면 전체 파일을 하나의 JSON 객체로 읽으려 시도
                print(f"Attempting to load {index_file} as a single JSON object (not JSONL).")
                with open(index_file, mode="r", encoding="utf-8") as reader:
                    full_data = json.load(reader) # 전체 파일을 하나의 JSON 객체로 로드
                    if isinstance(full_data, list):
                        items.extend(full_data) # 리스트라면 extend
                    else:
                        items.append(full_data) # 리스트가 아니라면 단일 객체로 추가
                print(f"Loaded {len(items) - offset} image-text pairs from {index_file} (single JSON object). ")
            # --- END OF MODIFICATION ---

            offset = len(items)
            
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split, task=None):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        # image_path는 self.data_path 기준으로 상대 경로일 수 있습니다.
        # VQA v2의 경우 이미지 경로가 'train2014/COCO_train2014_000000xxxxxx.jpg' 형태이므로,
        # self.data_path를 /data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/VQA_v2/images 로 설정한다고 가정합니다.
        
        # 이미지 경로를 올바르게 조합
        # self.data_path (ex: /data/VQA_v2/images)
        # + image_path (ex: train2014/COCO_train2014_000000xxxxxx.jpg)
        full_image_path = os.path.join(self.data_path, image_path)
        
        image = self.loader(full_image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str): # 'question_text'는 여기로 들어옴
            # --- START OF FIX ---
            # 문자열을 토큰 문자열 리스트로 변환
            tokens_str = self.tokenizer.tokenize(text_segment) 
            # 토큰 문자열 리스트를 토큰 ID 리스트로 변환 (핵심 수정)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
            # --- END OF FIX ---
        else:
            # text_segment가 이미 토큰 ID 리스트인 경우 (기존 VQAv2 데이터 등)
            tokens = text_segment[:] 
            # 여기서는 tokens가 이미 int 리스트라고 가정.
            # 만약 여기도 str이라면 문제가 생길 수 있지만, 현재 VQAv2FourChoiceDataset은 str만 처리.

        if len(tokens) == 0:
            raise RuntimeError("The text segment should contain at least one token!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2: # [CLS] ... [SEP] 토큰을 위한 공간 확보
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens


    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"] # image_path는 'train2014/COCO_train2014_xxxxxx.jpg' 같은 형식이어야 함
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"] # BaseDataset은 'text_segment'를 사용
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
        split=("train", "restval", "val"), 
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
                if item["split"] in ["train", "restval", "val"]:
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

def _make_captioning_visual7w_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval", "val"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_visual_v7w.json")
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
        # VQAv2는 정답이 3129개 클래스로 분류되므로, answer2label.txt 파일이 필요합니다.
        # 4지선다형 VQA에서는 이 파일이 필요하지 않을 수 있습니다.
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
            data["labels"] = torch.FloatTensor(labels) # One-hot-like soft labels
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

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
            ["train", "val", "test", "test-dev"],
            [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text, 
                    "token_ids": token_ids, 
                }

            annotations[split] = _annot

        all_major_answers = list()

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            # _annot = annotations[split]
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"]
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = cls.get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0:
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val", "test", "test-dev"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
                "test": "test2015",
                "test-dev": "test2015",
            }[split]
            # Assumes images are in a subdirectory like 'train2014' or 'val2014' inside data_path
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have caption annotations")
            else:
                print("not all images have caption annotations")
            print(len(paths), len(annot_paths), len(annot))

            items = []
            for path in annot_paths:
                iid = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    if split in ["train", "val"]:
                        labels = q["labels"]
                        scores = q["scores"]
                    else:
                        labels, scores = [], []

                    items.append({
                        "image_path": os.path.join(split_name, path.split('/')[-1]), 
                        "text_segment": q["token_ids"], # VQAv2 기존 데이터는 질문이 토큰 ID 리스트
                        "labels": labels, 
                        "scores": scores, 
                        "qid": qid, 
                    })
            split2items[split] = items

            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

        # Following ViLT, we use 1000 images of the original val set as the final val set         
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]
        
        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))


# --- START OF NEW CODE FOR 4-CHOICE VQA DATASET ---
class VQAv2FourChoiceDataset(BaseDataset):
    """
    Custom Dataset for VQA-v2 with 4-choice answers,
    using JSONL files generated by the augmentation script.
    """
    def __init__(self, data_path, split, tokenizer, num_max_bpe_tokens, **kwargs):
        # BaseDataset의 __init__은 index_files를 self.get_index_files()로 호출합니다.
        # 여기서는 self.data_path가 직접 jsonl 파일이 있는 디렉토리를 가리킨다고 가정합니다.
        # 예를 들어, /data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/VQA_v2/generated_vqa_data_abcd
        
        super().__init__(
            data_path=data_path, 
            split=split, 
            tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, 
            transform=kwargs.pop('transform'), # transform은 BaseDataset이 직접 받으므로 kwargs에서 추출하여 전달
            task='vqav2_4choice' # task 인자를 명시적으로 전달하여 get_index_files에서 사용
        )
        logger.info(f"VQAv2FourChoiceDataset initialized for split: {split}")

    @staticmethod
    def get_index_files(split, task=None):
        # augmentation.py가 생성한 파일명에 맞게 설정합니다.
        # 이 파일명은 data_path의 하위 디렉토리가 아니라, data_path 자체에 있을 것입니다.
        # 현재 augmentation.py는 `.json` 확장자로 파일을 저장하고 있습니다.
        if split == "train":
            return ("vqa_v2_train_4choice_abcd_generated.json", ) # .json 확장자 유지
        elif split == "val":
            return ("vqa_v2_val_4choice_abcd_generated.json", )   # .json 확장자 유지
        else:
            raise RuntimeError(f"split {split} is not found for VQAv2FourChoiceDataset!")

    def _get_image(self, image_id: int):
        # augmentation.py에서 image_id만 제공하므로, 원본 VQA 이미지 경로를 직접 구성해야 합니다.
        # VQA_v2 데이터셋의 이미지 경로 규칙을 따릅니다.
        # ex: COCO_train2014_000000xxxxxx.jpg
        # BaseDataset의 self.data_path는 jsonl 파일이 있는 디렉토리이므로,
        # 이미지의 루트 디렉토리를 별도로 지정하거나, 이 함수 내에서 구성해야 합니다.
        # 편의상 self.data_path의 상위 경로로 가정합니다.
        
        # 이미지의 실제 루트 디렉토리 (VQA_v2/images)를 명시적으로 설정합니다.
        # 이 경로는 환경에 따라 다를 수 있으므로, 정확하게 설정해야 합니다.
        # 예를 들어, VQA_DATA_DIR = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/VQA_v2"
        image_root_dir = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/VQA_v2/images"

        if self.split == "train":
            image_subdir = "train2014"
        elif self.split == "val":
            image_subdir = "val2014"
        else:
            # 다른 split이 들어올 경우 (test 등), 해당 디렉토리 처리 로직 추가 필요
            image_subdir = "test2015" # VQA v2 test set의 일반적인 디렉토리 이름
            logger.warning(f"Using default image subdirectory '{image_subdir}' for unknown split '{self.split}'")


        image_filename = f"COCO_{image_subdir}_{str(image_id).zfill(12)}.jpg"
        full_image_path = os.path.join(image_root_dir, image_subdir, image_filename)
        
        if not os.path.exists(full_image_path):
            logger.error(f"Image not found: {full_image_path}")
            # 이미지가 없을 경우 학습 중단 또는 더미 이미지 반환 등 처리 필요
            # 현재 상태로는 FileNotFoundError를 발생시켜 학습 중단 (올바른 접근)
            raise FileNotFoundError(f"Image not found at {full_image_path}")

        image = self.loader(full_image_path)
        return self.transform(image)

    def __getitem__(self, index: int):
        item = self.items[index]

        question_text = item['question'] 
        image_id = item['image_id']
        correct_answer_idx = item['answer_index'] # 0, 1, 2, 3

        # 이미지 로딩 (BaseDataset의 _get_image를 오버라이드하여 image_id 기반으로 로드)
        img = self._get_image(image_id) 

        # 질문 토큰화 및 패딩 (BaseDataset의 _get_text_segment 사용)
        language_tokens, padding_mask, _ = self._get_text_segment(question_text)

        # 모델 입력 형식에 맞춰 데이터 구성
        # BEiT3ForVisualQuestionAnsweringFourChoice의 forward 시그니처:
        # forward(self, image, question, padding_mask, **kwargs)
        
        # Fairseq의 collate_fn (utils.merge_batch_tensors_by_dict_key)이
        # 'net_input' 딕셔너리를 받아서 배치로 묶을 것이므로, 그에 맞춰 구조를 만듭니다.
        data = {
            'id': index, # 샘플 ID
            'net_input': {
                'image': img, # PIL Image는 transforms에 의해 Tensor로 변환됨
                'question': torch.tensor(language_tokens, dtype=torch.long),
                'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            },
            'target': torch.tensor(correct_answer_idx, dtype=torch.long), # 정답 인덱스 (0, 1, 2, 3)
            'question_id': item['question_id'], # 평가용으로 필요할 수 있음
            'original_correct_answer': item['original_correct_answer'], # 디버깅용
            'choices': item['choices'], # 디버깅용
        }
        return data
# --- END OF NEW CODE FOR 4-CHOICE VQA DATASET ---


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
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")


class CaptioningDataset(BaseDataset):

    def __init__(self, data_path, split, transform, 
                 tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, 
            task=task, # BaseDataset에 task 인자 전달
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl", )
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

    # def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob): # ms
    #     bool_masked_pos = [0] * len(tokens)
    #     to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
    #     to_mask = max(to_mask, 1)
    #     num_masked_tokens = 0
    #     while num_masked_tokens < to_mask:
    #         i = random.randint(1, num_tokens - 1)
    #         if bool_masked_pos[i] == 0:
    #             bool_masked_pos[i] = 1
    #             tokens[i] = self._get_mask_token(tokens[i])
    #             num_masked_tokens += 1

    #     return tokens, bool_masked_pos

    def _masking_on_text_tokens(self, tokens, num_tokens):
        """
        Masks only the last non-special token of the sentence.
        """
        # Create a boolean mask initialized to all False (0)
        bool_masked_pos = [0] * len(tokens)
        
        # The last actual token is at index num_tokens - 2 
        # (before the end-of-sentence token)
        last_token_idx = num_tokens - 2
        
        # We only proceed if there is a token to mask (i.e., not an empty sentence)
        if last_token_idx >= 1:
            # Set the position of the last token to True (1)
            bool_masked_pos[last_token_idx] = 1
            # Replace the token at that position with the [MASK] token
            tokens[last_token_idx] = self.mask_token_id
            
        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            masked_tokens = language_tokens[:]
            # masked_tokens, language_masked_pos = \
            #     self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob) # ms
            masked_tokens, language_masked_pos = \
                self._masking_on_text_tokens(masked_tokens, num_tokens)
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        # _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

    @staticmethod
    def make_visual7w_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="train")
        _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="val")
        # _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")

class BinaryDataset(BaseDataset):

    def __init__(self, data_path, split, transform, 
                 tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, 
            task=task, # BaseDataset에 task 인자 전달
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl", )
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

    def _masking_on_text_tokens(self, tokens, num_tokens):
        """
        Masks only the last non-special token of the sentence.
        """
        # Create a boolean mask initialized to all False (0)
        bool_masked_pos = [0] * len(tokens)
        
        # The last actual token is at index num_tokens - 2 
        # (before the end-of-sentence token)
        last_token_idx = num_tokens - 2
        
        # We only proceed if there is a token to mask (i.e., not an empty sentence)
        if last_token_idx >= 1:
            # Set the position of the last token to True (1)
            bool_masked_pos[last_token_idx] = 1
            # Replace the token at that position with the [MASK] token
            tokens[last_token_idx] = self.mask_token_id
            
        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            masked_tokens = language_tokens[:]
            # masked_tokens, language_masked_pos = \
            #     self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob) # ms
            masked_tokens, language_masked_pos = \
                self._masking_on_text_tokens(masked_tokens, num_tokens)
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        # _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

    @staticmethod
    def make_visual7w_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="train")
        _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="val")
        # _make_captioning_visual7w_karpathy_dataset_index(data_path, tokenizer, split="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")


# --- UPDATED task2dataset MAPPING ---
task2dataset = {
    "nlvr2": NLVR2Dataset, 
    "vqav2": VQAv2Dataset, 
    "flickr30k": RetrievalDataset, 
    "coco_retrieval": RetrievalDataset,    
    "coco_captioning": CaptioningDataset,
    "nocaps": CaptioningDataset,
    "imagenet": ImageNetDataset,
    "vqav2_4choice": VQAv2FourChoiceDataset, # 새로운 4지선다형 VQA 데이터셋 추가
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
        collate_fn=utils.merge_batch_tensors_by_dict_key, # 이 collate_fn이 VQAv2FourChoiceDataset의 출력을 처리할 수 있어야 함
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
            transforms.Resize(size, interpolation=3),   # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task] # 여기서 args.task에 따라 올바른 데이터셋 클래스가 선택됩니다.
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {'transform': transform} # transform을 VQAv2FourChoiceDataset의 **kwargs로 전달
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        # transform은 이제 opt_kwargs를 통해 전달됩니다.
        tokenizer=tokenizer, 
        num_max_bpe_tokens=args.num_max_bpe_tokens, 
        task=args.task, # get_index_files에서 사용될 수 있도록 task도 전달
        **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="val", is_train=False) # 보통 val이 eval에 사용됨
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=False) # val set은 학습 시에는 is_train=False로 사용