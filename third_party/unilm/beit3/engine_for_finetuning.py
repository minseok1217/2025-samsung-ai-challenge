# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# timm 임포트 재확인
from timm.utils import ModelEma, accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy # BCEWithLogitsLoss는 직접 import 필요

# datasets 모듈의 함수 임포트
from datasets import get_sentencepiece_model_for_beit3

import utils # utils.py 임포트


class BaseTaskHandler:
    """Base class for all task handlers."""
    def __init__(self, args):
        self.args = args
        # MetricLogger는 train_one_epoch, evaluate 함수에서 생성되므로,
        # 각 step 함수에서는 직접 받아서 사용하거나, 필요한 경우 handler에 할당합니다.
        self.metric_logger = None 
        self.split = None

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        raise NotImplementedError
    
    @torch.no_grad() # 평가 단계는 기울기 계산 필요 없음
    def valid_step(self, model, samples):
        raise NotImplementedError
    
    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        """
        Helper method to compute loss and metrics for logging/evaluation.
        This might be called by train_step or valid_step, or directly by evaluate/train_one_epoch.
        """
        raise NotImplementedError

    def before_eval(self, metric_logger, data_loader, **kwargs):
        """Setup before evaluation loop."""
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        """Cleanup and final metric computation after evaluation loop."""
        # 기본적으로 metric_logger에서 평균 값을 가져오고, 리턴 키를 지정합니다.
        # 하위 클래스에서 오버라이드하여 태스크별 최종 평가 로직을 구현합니다.
        raise NotImplementedError("after_eval must be implemented by concrete handlers.")


# 기존 TaskHandler 클래스는 BaseTaskHandler와 중복되거나 통합될 수 있습니다.
# BEiT3의 원본 코드가 BaseTaskHandler와 TaskHandler 두 가지 스타일을 섞어 쓴다면,
# 이를 명확히 해야 합니다. 여기서는 BaseTaskHandler가 메인이며,
# 기존 TaskHandler의 기능들을 BaseTaskHandler 또는 그 하위 클래스에 통합합니다.
# 만약 `TaskHandler`가 다른 곳에서 필수적으로 사용된다면, 이름 충돌을 피하기 위해
# `BaseTaskHandler`의 이름을 바꾸거나, `TaskHandler`를 더 추상적인 역할로 변경해야 합니다.
# 주어진 코드 조각으로 보아 `BaseTaskHandler`가 새로운 추상 클래스 역할을 하는 것으로 보입니다.

# --- NLVR2Handler 구현 ---
class NLVR2Handler(BaseTaskHandler):
    def __init__(self, args):
        super().__init__(args)
        # NLVR2는 이진 분류 (True/False), 따라서 CrossEntropyLoss 사용
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        # NLVR2 데이터셋은 samples['net_input']에 'image', 'image2', 'text_description', 'padding_mask'를,
        # samples['label']에 'label'을 가집니다.
        # model(**samples["net_input"])은 BEiT3ForVisualReasoning의 forward를 호출
        logits = model(
            image_a=samples['net_input']['image'], 
            image_b=samples['net_input']['image2'], 
            text_description=samples['net_input']['language_tokens'], # NLVR2 데이터셋의 텍스트 키는 language_tokens
            padding_mask=samples['net_input']['padding_mask']
        )
        labels = samples['label'] # NLVR2 레이블

        # 손실 계산
        loss = self.loss_fn(input=logits, target=labels)
        
        # 정확도 계산
        acc = (logits.max(-1)[-1] == labels).float().mean() * 100.0

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        return {'loss': loss.item(), 'acc': acc.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        logits = model(
            image_a=samples['net_input']['image'], 
            image_b=samples['net_input']['image2'], 
            text_description=samples['net_input']['language_tokens'],
            padding_mask=samples['net_input']['padding_mask']
        )
        labels = samples['label']

        loss = self.loss_fn(input=logits, target=labels)
        
        # 정확도 계산 (top-1)
        correct_preds = (logits.max(-1)[-1] == labels).float().sum()
        total_preds = labels.size(0)

        return {
            'loss': loss.item(),
            'correct_predictions': correct_preds.item(),
            'total_predictions': total_preds,
            'accuracy': (correct_preds / total_preds * 100.0).item() if total_preds > 0 else 0.0,
        }

    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        # NLVR2는 이진 분류이므로 accuracy만 필요
        pred_labels = torch.argmax(logits, dim=-1)
        correct_count = (pred_labels == targets).sum().item()
        total_count = targets.size(0)
        return {'acc': (correct_count / total_count) * 100.0}
        
    def after_eval(self, **kwargs):
        # metric_logger에서 평균 acc를 가져와서 반환
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.acc))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc"
        

# --- VQAv2Handler 구현 ---
class VQAv2Handler(BaseTaskHandler):
    def __init__(self, args):
        super().__init__(args)
        # VQAv2는 일반적으로 Multi-label soft target classification으로 처리
        # BCEWithLogitsLoss와 유사한 Loss를 사용합니다.
        # VQA의 labels는 N_answers 차원의 soft target (e.g., 0.3, 0.6, 1.0)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        self.label2ans = None # datasets.py에서 로드된 answer2label 정보

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        # samples['net_input']에 'image', 'question', 'padding_mask'
        # samples['labels']에 soft targets
        
        logits = model(
            image=samples['net_input']['image'], 
            question=samples['net_input']['question'], 
            padding_mask=samples['net_input']['padding_mask']
        ) # [batch_size, num_classes (3129)]
        
        labels = samples['labels'] # [batch_size, num_classes (3129)] - float tensor

        # BCEWithLogitsLoss는 logits와 target을 float로 기대합니다.
        loss = self.loss_fn(input=logits.float(), target=labels.float()) * labels.shape[1] # Multi-label Loss

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        return {'loss': loss.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        logits = model(
            image=samples['net_input']['image'], 
            question=samples['net_input']['question'], 
            padding_mask=samples['net_input']['padding_mask']
        )
        
        loss = None
        if 'labels' in samples: # validation set에 labels가 있는 경우 (trainable_val, rest_val)
            labels = samples['labels']
            loss = self.loss_fn(input=logits.float(), target=labels.float()) * labels.shape[1]

            # VQA score 계산 (정식 VQA 평가 스크립트의 점수 계산 로직을 간략화)
            # utils.VQAScore()가 있다면 그것을 사용
            if hasattr(utils, 'VQAScore'):
                vqa_score = utils.VQAScore()(logits, labels).item() * 100.0 # VQA 점수
            else:
                # VQAScore 함수가 없는 경우, 단순 accuracy (임시)
                # VQA는 정확도 100%가 아닐 수 있으므로 VQA score가 더 적합
                vqa_score = 0.0 # Placeholder if VQAScore is not available
                pred_labels = torch.argmax(logits, dim=-1) # 가장 높은 로짓 선택
                # 이 방식은 soft target VQA에서는 정확하지 않음 (정확한 VQA 스코어는 별도 계산 필요)

            return {
                'loss': loss.item() if loss is not None else 0.0,
                'vqa_score': vqa_score,
                'qid': samples['qid'].cpu().numpy(), # 평가 후 JSON 덤프를 위해 필요
                'pred_logits': logits.cpu().numpy(), # 평가 후 JSON 덤프를 위해 필요
            }
        else: # test set (labels가 없는 경우, qid만 있음)
            _, preds = logits.max(-1) # 가장 높은 로짓의 인덱스 선택
            predictions_list = []
            if 'qid' in samples and self.label2ans:
                for qid_item, pred_idx in zip(samples['qid'], preds):
                    predictions_list.append({
                        "question_id": qid_item.item(),
                        "answer": self.label2ans[pred_idx.item()] # 인덱스를 실제 답변으로 변환
                    })
            return {'predictions': predictions_list}

    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        # VQAv2는 복잡한 평가 로직이 있으므로, 여기서 정확도 대신 VQA Score를 반환
        if hasattr(utils, 'VQAScore'):
            score = utils.VQAScore()(logits, targets).item() * 100.0
            return {'vqa_score': score}
        else:
            return {'vqa_score': 0.0} # VQAScore 함수가 없다면 0으로 임시 반환

    def before_eval(self, metric_logger, data_loader, **kwargs):
        super().before_eval(metric_logger, data_loader, **kwargs)
        self.predictions = [] # for test split
        # data_loader.dataset이 VQAv2Dataset인지 확인하고 label2ans를 가져옴
        if hasattr(data_loader.dataset, 'label2ans'):
            self.label2ans = data_loader.dataset.label2ans
        else:
            logger.warning("label2ans not found in dataset. VQA prediction output might be incomplete.")

    def after_eval(self, **kwargs):
        # 평가 후 최종 결과 반환
        # test split의 경우 self.predictions를, val split의 경우 score를 반환
        if len(self.predictions) > 0:
            return self.predictions, "prediction" # prediction 덤프용
        else:
            # Score (val set)
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"


# --- VQAv2FourChoiceHandler는 이미 존재한다고 가정 ---
class VQAv2FourChoiceHandler(BaseTaskHandler):
    def __init__(self, args):
        super().__init__(args)
        # 4지선다형 VQA는 분류 문제이므로 CrossEntropyLoss를 사용합니다.
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        logits = model(**samples["net_input"]) # [batch_size, 4]
        targets = samples["target"] # [batch_size]

        loss = self.loss_fn(logits, targets)

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        return {'loss': loss.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        logits = model(**samples["net_input"]) # [batch_size, 4]
        targets = samples["target"] # [batch_size]
        
        loss = self.loss_fn(logits, targets)

        pred_labels = torch.argmax(logits, dim=-1)
        correct = (pred_labels == targets).sum().item()
        total = targets.size(0)

        return {
            'loss': loss.item(), 
            'correct_predictions': correct, 
            'total_predictions': total,
            'accuracy': (correct / total) * 100.0 if total > 0 else 0.0,
        }
        
    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        pred_labels = torch.argmax(logits, dim=-1)
        correct_count = (pred_labels == targets).sum().item()
        total_count = targets.size(0)
        return {
            'accuracy': (correct_count / total_count) * 100.0,
            'correct_predictions': correct_count,
            'total_predictions': total_count,
        }
    
    def after_eval(self, **kwargs):
        # 4지선다형은 정확도를 주 지표로 사용
        # evaluate 함수에서 취합된 정확도 평균을 반환하도록 합니다.
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.accuracy)) # metric_logger에 accuracy 미터가 있다고 가정
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "accuracy"


# --- START OF OTHER HANDLERS (PLACEHOLDER) ---
# ImageNetHandler
class ImageNetHandler(BaseTaskHandler):
    def __init__(self, args) -> None:
        super().__init__(args)
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            self.criterion = SoftTargetCrossEntropy()
        elif args.label_smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        image = samples['image']
        label = samples['label']
        if mixup_fn is not None:
            image, label = mixup_fn(image, label) # mixup이 여기서 적용된다고 가정
        
        logits = model(image=image)
        loss = self.criterion(logits, label)
        
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        return {"loss": loss.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        image = samples['image']
        label = samples['label']
        logits = model(image=image)
        loss = self.criterion(logits, label)
        
        acc1, acc5 = accuracy(logits, label, topk=(1, 5))
        return {'loss': loss.item(), 'acc1': acc1.item(), 'acc5': acc5.item()}
    
    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        # ImageNet은 top-1, top-5 정확도를 사용
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        return {'acc1': acc1.item(), 'acc5': acc5.item()}

    def after_eval(self, **kwargs):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"


class RetrievalHandler(BaseTaskHandler):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        # Retrieval 모델은 일반적으로 손실을 직접 반환
        loss, vision_cls, language_cls = model(
            image=samples['net_input']['image'], 
            text_description=samples['net_input']['language_tokens'], 
            padding_mask=samples['net_input']['padding_mask']
        )
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()
        torch.cuda.synchronize()
        return {"loss": loss.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        # eval_batch 로직을 valid_step에 통합
        vision_cls, _ = model(image=samples['net_input']['image'], only_infer=True)
        _, language_cls = model(
            text_description=samples['net_input']['language_tokens'], 
            padding_mask=samples['net_input']['padding_mask'], 
            only_infer=True
        )
        # 평가 단계에서 특징을 수집하여 after_eval에서 최종 계산
        self.image_feats.append(vision_cls.clone().cpu()) # CPU로 옮겨서 메모리 절약
        self.text_feats.append(language_cls.clone().cpu())
        self.image_ids.append(samples['image_id'].clone().cpu())
        return {} # 손실이나 즉각적인 메트릭 없음

    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        return {} # Retrieval은 after_eval에서 모든 메트릭을 계산

    def before_eval(self, metric_logger, data_loader, **kwargs):
        super().before_eval(metric_logger, data_loader, **kwargs)
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        
    def after_eval(self, **kwargs):
        # Retrieval 평가 로직 (복잡)
        # 이전에 제공된 RetrievalHandler의 after_eval 로직과 동일하게 구현
        image_feats_cat = torch.cat(self.image_feats, dim=0)
        text_feats_cat = torch.cat(self.text_feats, dim=0)
        image_ids_cat = torch.cat(self.image_ids, dim=0)

        # 이미지별 고유 특징 취합 (기존 RetrievalHandler 로직)
        image_feats = {}
        for feats, ids in zip(image_feats_cat, image_ids_cat):
            if ids.item() not in image_feats:
                image_feats[ids.item()] = feats
        
        tiids = image_ids_cat # 모든 텍스트-이미지 쌍의 이미지 ID
        iids = torch.LongTensor(list(image_feats.keys())) # 고유 이미지 ID

        sorted_tensors = [image_feats[key].view(1, -1) for key in sorted(image_feats.keys())]
        image_cls_feats = torch.cat(sorted_tensors, dim=0).to(text_feats_cat.device) # GPU로 다시 옮김
        text_cls_feats = text_feats_cat.to(text_feats_cat.device)
        iids = iids.to(text_feats_cat.device)
        tiids = tiids.to(text_feats_cat.device) # GPU로 다시 옮김

        scores = image_cls_feats @ text_cls_feats.t()
        
        # Recall@k 계산 로직 (기존 RetrievalHandler와 동일)
        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]

        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]

        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

        eval_result = {
            "tr_r10": tr_r10.item() * 100.0, 
            "tr_r5": tr_r5.item() * 100.0, 
            "tr_r1": tr_r1.item() * 100.0, 
            "ir_r10": ir_r10.item() * 100.0, 
            "ir_r5": ir_r5.item() * 100.0, 
            "ir_r1": ir_r1.item() * 100.0, 
            "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"


class CaptioningHandler(BaseTaskHandler):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.predictions = []
        self.criterion = utils.BertCaptioningLoss(args.label_smoothing, args.drop_worst_ratio, args.drop_worst_after)
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        self.num_beams = args.num_beams
        self.max_len = args.num_max_bpe_tokens
        self.length_penalty = args.length_penalty
        self.vocab_size = args.vocab_size

    def train_step(self, model, samples, optimizer, loss_scaler, mixup_fn):
        # samples['net_input']에 'image', 'text_ids', 'padding_mask', 'language_masked_pos'
        # samples['image_id'], 'global_step'
        
        logits, _ = model(
            image=samples['net_input']['image'], 
            text_ids=samples['net_input']['language_tokens'], # Text ID (masked)
            padding_mask=samples['net_input']['padding_mask'], 
            language_masked_pos=samples['net_input']['language_masked_pos'] if 'language_masked_pos' in samples['net_input'] else None, 
            image_id=samples['image_id'] if 'image_id' in samples else None # 이미지 ID는 필수가 아닐 수 있음
        )
        masked_labels = samples['net_input']['language_tokens'][samples['net_input']['language_masked_pos'].bool()] # 원본 텍스트 ID
        
        loss = self.criterion(logits, masked_labels, samples['global_step'])
        
        score = torch.max(logits, -1)[1].data == masked_labels
        acc = torch.sum(score.float()) / torch.sum(samples['net_input']['language_masked_pos'])

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=self.args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(optimizer is not None))
        else:
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        return {"loss": loss.item(), "acc": acc.item()}

    @torch.no_grad()
    def valid_step(self, model, samples):
        # Beam search decoding logic (complex)
        # samples['net_input']['image']만 사용
        image = samples['net_input']['image']
        image_id = samples['image_id'] if 'image_id' in samples else None

        cur_len = 2
        num_keep_best = 1
        TOPN_PER_BEAM = 3

        batch_size = image.size(0)
        mask_id = self.tokenizer.mask_token_id
        cls_id = self.tokenizer.cls_token_id
        pad_id = self.tokenizer.pad_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_token_ids = [sep_id]

        cls_ids = torch.full((batch_size, 1), cls_id, dtype=torch.long, device=image.device)
        mask_ids = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=image.device)
        cur_input_ids = torch.cat([cls_ids, mask_ids], dim=1)
        tmp_ids = torch.full((batch_size, self.max_len-1), mask_id, dtype=torch.long, device=image.device)
        decoding_results = torch.cat([cls_ids, tmp_ids], dim=1)
        
        # Expand input to num beams
        cur_input_ids = cur_input_ids.unsqueeze(1).expand(batch_size, self.num_beams, cur_len)
        cur_input_ids = cur_input_ids.contiguous().view(batch_size * self.num_beams, cur_len)
        decoding_results = decoding_results.unsqueeze(1).expand(batch_size, self.num_beams, self.max_len)
        decoding_results = decoding_results.contiguous().view(batch_size * self.num_beams, self.max_len)
        image = image.unsqueeze(1).expand(batch_size, self.num_beams, image.size(-3), image.size(-2), image.size(-1))
        image = image.contiguous().view(batch_size * self.num_beams, image.size(-3), image.size(-2), image.size(-1))

        generated_hyps = [
            utils.BeamHypotheses(
                num_keep_best, self.max_len, length_penalty=self.length_penalty, early_stopping=False
            ) for _ in range(batch_size)
        ]
        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=cur_input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # done sentences
        done = [False for _ in range(batch_size)]
        incremental_state = {}

        while cur_len <= self.max_len:
            next_token_idx = 1
            padding_masks = torch.full(
                cur_input_ids.shape, 0, dtype=torch.long, device=image.device
            )
            input_image = image
            if cur_len != 2:
                input_image = None

            outputs, incremental_state_next = model(
                image=input_image, text_ids=cur_input_ids, language_masked_pos=None,
                padding_mask=padding_masks, text_len=cur_len, incremental_state=incremental_state)
            incremental_state = incremental_state_next

            scores = outputs[:, next_token_idx, :]
            scores = F.log_softmax(scores, dim=-1)
            assert scores.size() == (batch_size * self.num_beams, self.vocab_size)
            _scores = scores + beam_scores[:, None].expand_as(scores)
            _scores = _scores.view(batch_size, self.num_beams * self.vocab_size)
            next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * self.num_beams, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * self.num_beams)

            next_batch_beam = []
            for batch_ex in range(batch_size):
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_id, 0)] * self.num_beams)
                    continue

                next_sent_beam = []
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    beam_id = idx // self.vocab_size
                    word_id = idx % self.vocab_size
                    if (word_id.item() in eos_token_ids and cur_len + 1 <= self.max_len) or (cur_len + 1 == self.max_len):
                        generated_hyps[batch_ex].add(
                            decoding_results[batch_ex * self.num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * self.num_beams + beam_id))
                    if len(next_sent_beam) == self.num_beams:
                        break

                if cur_len + 1 == self.max_len:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == self.num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_id, 0)] * self.num_beams
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == self.num_beams * (batch_ex + 1)
            
            assert len(next_batch_beam) == batch_size * self.num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = cur_input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = cur_input_ids.new([x[2] for x in next_batch_beam])

            cur_input_ids = cur_input_ids[beam_idx, :]
            decoding_results = decoding_results[beam_idx, :]
            for module in incremental_state:
                for key in incremental_state[module]:
                    result = incremental_state[module][key].index_select(0, beam_idx)
                    incremental_state[module][key] = result[:,:,:-1,:]
            
            next_ids = torch.full(
                (batch_size * self.num_beams, 1), mask_id, dtype=torch.long, device=image.device
            )
            cur_input_ids = torch.cat([beam_words.unsqueeze(1), next_ids], dim=1)
            decoding_results[:, cur_len-1] = beam_words
            cur_len = cur_len + 1
            if all(done):
                break
        
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                            dtype=torch.float).fill_(-1e5).to(cur_input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores, min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1
            all_best.append(best)
        
        decoded = cur_input_ids.new(batch_size, num_keep_best, self.max_len).fill_(pad_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        
        captions = self.tokenizer.batch_decode(decoded.squeeze(1), skip_special_tokens=True)
        
        predictions_list = []
        if image_id is not None:
            for qid, pred in zip(image_id, captions):
                predictions_list.append({
                    "image_id": qid.item(), 
                    "caption": pred, 
                })
        return {'predictions': predictions_list}


    def _get_loss_and_metrics(self, logits, targets, **kwargs):
        # 캡셔닝은 loss와 acc를 반환
        return {'loss': kwargs.get('loss', 0.0), 'acc': kwargs.get('acc', 0.0)}

    def before_eval(self, metric_logger, data_loader, **kwargs):
        super().before_eval(metric_logger, data_loader, **kwargs)
        self.predictions = []

    def after_eval(self, **kwargs):
        return self.predictions, "prediction"


# --- get_handler 함수 수정 ---
def get_handler(args):
    # 451번째 라인이 이 부분일 것입니다.
    if args.task == "nlvr2":
        return NLVR2Handler(args)
    elif args.task == "vqav2":
        # 기존 VQAv2 핸들러 (3129 클래스 분류)
        return VQAv2Handler(args) 
    elif args.task == "vqav2_4choice": # 여기에 새로운 VQAv2FourChoiceHandler를 추가
        return VQAv2FourChoiceHandler(args)
    elif args.task == "imagenet":
        return ImageNetHandler(args) 
    elif args.task in ["coco_captioning", "nocaps"]:
        return CaptioningHandler(args)
    elif args.task in ["flickr30k", "coco_retrieval"]:
        return RetrievalHandler(args)
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


# def train_one_epoch(
#         model: torch.nn.Module, data_loader: Iterable, 
#         optimizer: torch.optim.Optimizer, device: torch.device, 
#         handler: BaseTaskHandler, # BaseTaskHandler 타입 힌트
#         epoch: int, start_steps: int, 
#         lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
#         update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
#         log_writer: Optional[utils.TensorboardLogger] = None, 
#         task = None, mixup_fn=None, # <- 여기에 task가 있지만, args는 없음
# ):
#     model.train(True)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     if task in ["coco_captioning", "nocaps"]: # <--- 여기서 args.task를 사용해야 하는데, task만 있음
#         data["global_step"] = global_step

#     if loss_scaler is None:
#         model.zero_grad()
#         model.micro_steps = 0
#     else:
#         optimizer.zero_grad()

#     for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         step = data_iter_step // update_freq
#         global_step = start_steps + step  # global training iteration
        
#         # Update LR & WD for the first acc
#         if lr_schedule_values is not None and data_iter_step % update_freq == 0:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        
#         # put input data into cuda
#         # samples['net_input']의 각 항목을 GPU로 이동
#         for key in data['net_input'].keys():
#             if isinstance(data['net_input'][key], torch.Tensor):
#                 data['net_input'][key] = data['net_input'][key].to(device, non_blocking=True)
#                 if loss_scaler is None and key.startswith("image"):
#                     data['net_input'][key] = data['net_input'][key].half() # image는 FP16
        
#         # samples['target'] 또는 samples['labels'] 같은 다른 데이터도 GPU로 이동
#         if 'target' in data and isinstance(data['target'], torch.Tensor):
#             data['target'] = data['target'].to(device, non_blocking=True)
#         if 'labels' in data and isinstance(data['labels'], torch.Tensor):
#             data['labels'] = data['labels'].to(device, non_blocking=True)
#         if 'label' in data and isinstance(data['label'], torch.Tensor): # NLVR2, ImageNet용
#             data['label'] = data['label'].to(device, non_blocking=True)
#         if 'qid' in data and isinstance(data['qid'], torch.Tensor): # VQAv2용
#             data['qid'] = data['qid'].to(device, non_blocking=True)
#         if 'image_id' in data and isinstance(data['image_id'], torch.Tensor): # Captioning용
#             data['image_id'] = data['image_id'].to(device, non_blocking=True)
        
#         # mixup for imagenet finetuning
#         if mixup_fn is not None and args.task == "imagenet": # mixup은 ImageNet에만 적용
#             data["net_input"]["image"], data["label"] = mixup_fn(data["net_input"]["image"], data["label"]) # ImageNet은 data["image"], data["label"]

#         if args.task in ["coco_captioning", "nocaps"]:
#             data["global_step"] = global_step # 캡셔닝 태스크에만 global_step 전달

#         # handler.train_step 호출
#         # train_step은 손실 스케일링 및 옵티마이저 스텝을 내부적으로 처리
#         results = handler.train_step(model, data, optimizer, loss_scaler, mixup_fn)

#         loss_value = results.pop("loss")

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         # loss_scaler는 handler.train_step 내부에서 처리되므로 여기서는 필요 없음.
#         # DeepSpeed 사용 시에는 model.backward, model.step, optimizer.zero_grad는 DeepSpeed가 자동으로 처리
#         # non-DeepSpeed (loss_scaler is None) 시에는 handler.train_step 내부에서 처리
        
#         if loss_scaler is None: # DeepSpeed 미사용 시
#             grad_norm = None # DeepSpeed가 아니면 grad_norm은 여기서 직접 계산되지 않을 수 있음
#             loss_scale_value = 1.0 # 스케일링 없음
#         else: # DeepSpeed 사용 시
#             # DeepSpeed는 내부적으로 grad_norm, loss_scale_value를 관리
#             # 여기서 직접 가져오기 어려울 수 있음. DeepSpeed 엔진에서 직접 접근 필요
#             grad_norm = model.gradient_norms[0] if hasattr(model, 'gradient_norms') and len(model.gradient_norms) > 0 else None
#             loss_scale_value = loss_scaler.state_dict()["scale"] if loss_scaler is not None else 1.0
        
#         # EMA 업데이트
#         if model_ema is not None and (data_iter_step + 1) % update_freq == 0:
#             model_ema.update(model)

#         torch.cuda.synchronize()

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(loss_scale=loss_scale_value)
#         min_lr = 10.
#         max_lr = 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])

#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         if grad_norm is not None:
#              metric_logger.update(grad_norm=grad_norm)

#         if log_writer is not None:
#             kwargs = {"loss": loss_value}
#             for key in results: # train_step에서 반환된 다른 메트릭들 (예: acc)
#                 kwargs[key] = results[key]
#             log_writer.update(head="train", **kwargs)

#             kwargs = {
#                 "loss_scale": loss_scale_value, 
#                 "lr": max_lr, 
#                 "min_lr": min_lr, 
#                 "weight_decay": weight_decay_value, 
#             }
#             if grad_norm is not None:
#                 kwargs["grad_norm"] = grad_norm
#             log_writer.update(head="opt", **kwargs)
#             log_writer.set_step()

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: BaseTaskHandler,
        epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        args=None, # 이전에 추가한 args
        mixup_fn=None, # <--- 여기에 mixup_fn 인자가 있습니다.
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        
        # put input data into cuda
        for key in data['net_input'].keys():
            if isinstance(data['net_input'][key], torch.Tensor):
                data['net_input'][key] = data['net_input'][key].to(device, non_blocking=True)
                # Image를 Half로 변환하는 로직: args.task를 사용하여 조건부 실행
                # if loss_scaler is None and key.startswith("image"): # <-- 이 부분도 args.task에 따라
                # if args.task in ["vqav2_4choice", "vqav2", "nlvr2", "imagenet", "flickr30k", "coco_retrieval", "coco_captioning", "nocaps"] \
                if key == 'image' and loss_scaler is None and \
                   (args.task in ["nlvr2", "vqav2", "vqav2_4choice", "imagenet", "flickr30k", "coco_retrieval", "coco_captioning", "nocaps"]): # 이미지 데이터만 half로
                   data['net_input'][key] = data['net_input'][key].half()
                # print(f"key: {key}, type: {type(data['net_input'][key])}, dtype: {data['net_input'][key].dtype if isinstance(data['net_input'][key], torch.Tensor) else 'N/A'}")
            else: # net_input 안에 텐서가 아닌 리스트/dict가 있다면
                # 예를 들어, NLVR2의 image2 같은 경우도 처리될 수 있음.
                # 명시적으로 이미지 텐서만 .half()를 적용
                if key == 'image2' and isinstance(data['net_input'][key], torch.Tensor) and loss_scaler is None:
                     data['net_input'][key] = data['net_input'][key].to(device, non_blocking=True).half()
                     # print(f"key: {key} (image2), type: {type(data['net_input'][key])}, dtype: {data['net_input'][key].dtype if isinstance(data['net_input'][key], torch.Tensor) else 'N/A'}")

        # samples['target'] 또는 samples['labels'] 같은 다른 데이터도 GPU로 이동
        # 그리고 FP16 학습 시 타겟은 보통 LongTensor (fp16 변환X)
        for target_key in ['target', 'labels', 'label', 'qid', 'image_id']:
            if target_key in data and isinstance(data[target_key], torch.Tensor):
                data[target_key] = data[target_key].to(device, non_blocking=True)

        # mixup for imagenet finetuning
        if mixup_fn is not None and args.task == "imagenet": # args.task 사용
            data["net_input"]["image"], data["label"] = mixup_fn(data["net_input"]["image"], data["label"])

        if args.task in ["coco_captioning", "nocaps"]: # args.task 사용
            data["global_step"] = global_step

        results = handler.train_step(model, data, optimizer, loss_scaler, mixup_fn)

        loss_value = results.pop("loss")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            # DeepSpeed 미사용 시 loss.backward() 및 optimizer.step()은 handler.train_step에서 직접 처리됨
            # 따라서 여기서는 특별히 할 일 없음.
            grad_norm = None
            loss_scale_value = 1.0 # 스케일링 없음
        else: # DeepSpeed 사용 시 (model.backward, model.step은 DeepSpeed가 처리)
            grad_norm = model.gradient_norms[0] if hasattr(model, 'gradient_norms') and len(model.gradient_norms) > 0 else None
            loss_scale_value = loss_scaler.state_dict()["scale"] if loss_scaler is not None else 1.0
        
        # EMA 업데이트
        if model_ema is not None and (data_iter_step + 1) % update_freq == 0:
            model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if grad_norm is not None:
             metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs_log = {"loss": loss_value} # 'kwargs' 변수명 충돌 피하기 위해 'kwargs_log'로 변경
            for key in results: 
                kwargs_log[key] = results[key]
            log_writer.update(head="train", **kwargs_log)

            kwargs_log_opt = { # 'kwargs' 변수명 충돌 피하기 위해 'kwargs_log_opt'로 변경
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
            }
            if grad_norm is not None:
                kwargs_log_opt["grad_norm"] = grad_norm
            log_writer.update(head="opt", **kwargs_log_opt)
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# def evaluate(data_loader, model, device, handler: BaseTaskHandler): # handler 타입 힌트
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()
#     handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

#     total_correct_predictions = 0
#     total_samples = 0
    
#     for data in metric_logger.log_every(data_loader, 10, header):
#         # samples['net_input']의 각 항목을 GPU로 이동
#         for key in data['net_input'].keys():
#             if isinstance(data['net_input'][key], torch.Tensor):
#                 data['net_input'][key] = data['net_input'][key].to(device, non_blocking=True)
        
#         # samples['target'] 또는 samples['labels'] 같은 다른 데이터도 GPU로 이동
#         if 'target' in data and isinstance(data['target'], torch.Tensor):
#             data['target'] = data['target'].to(device, non_blocking=True)
#         if 'labels' in data and isinstance(data['labels'], torch.Tensor):
#             data['labels'] = data['labels'].to(device, non_blocking=True)
#         if 'label' in data and isinstance(data['label'], torch.Tensor):
#             data['label'] = data['label'].to(device, non_blocking=True)
#         if 'qid' in data and isinstance(data['qid'], torch.Tensor):
#             data['qid'] = data['qid'].to(device, non_blocking=True)
#         if 'image_id' in data and isinstance(data['image_id'], torch.Tensor):
#             data['image_id'] = data['image_id'].to(device, non_blocking=True)
            
#         with torch.cuda.amp.autocast():
#             # handler.valid_step은 dict를 반환 (loss, correct_predictions, total_predictions, accuracy 등)
#             results = handler.valid_step(model=model, samples=data)
            
#             # VQAv2FourChoiceHandler, NLVR2Handler, ImageNetHandler의 경우
#             if 'correct_predictions' in results and 'total_predictions' in results:
#                 total_correct_predictions += results['correct_predictions']
#                 total_samples += results['total_predictions']
#                 metric_logger.update(loss=results.get('loss', 0.0))
#                 # 미터 이름을 각 핸들러의 after_eval에서 기대하는 이름으로 설정
#                 # 예를 들어, VQAv2FourChoiceHandler는 'accuracy', NLVR2Handler는 'acc', ImageNetHandler는 'acc1'
#                 if handler.__class__.__name__ == 'VQAv2FourChoiceHandler':
#                     metric_logger.meters['accuracy'].update(results['accuracy'], n=results['total_predictions'])
#                 elif handler.__class__.__name__ == 'NLVR2Handler':
#                     metric_logger.meters['acc'].update(results['accuracy'], n=results['total_predictions'])
#                 elif handler.__class__.__name__ == 'ImageNetHandler':
#                     metric_logger.meters['acc1'].update(results['acc1'], n=results['total_predictions'])
#                     metric_logger.meters['acc5'].update(results['acc5'], n=results['total_predictions'])

#             # VQAv2Handler의 경우 (prediction 덤프 또는 vqa_score)
#             elif 'predictions' in results:
#                 # Test set (labels 없음) -> predictions 리스트에 추가
#                 handler.predictions.extend(results['predictions'])
#             elif 'vqa_score' in results:
#                 # Val set (labels 있음) -> VQA score 업데이트
#                 metric_logger.meters['vqa_score'].update(results['vqa_score'], n=results['qid'].shape[0] if 'qid' in results else 1)
            
#             # CaptioningHandler의 경우 (predictions 덤프)
#             elif 'predictions' in results and isinstance(handler, CaptioningHandler):
#                 handler.predictions.extend(results['predictions'])

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()

#     # after_eval을 호출하여 최종 결과 반환
#     return handler.after_eval(metric_logger=metric_logger) # ms

@torch.no_grad()
def evaluate(data_loader, model, device, handler: BaseTaskHandler, args=None): # <-- 여기에 args 인자를 추가합니다!
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    # total_correct_predictions, total_samples은 이제 handler.valid_step 내부에서 처리하고,
    # metric_logger를 통해 업데이트되도록 합니다.
    # 여기서는 단순히 handler.valid_step의 결과를 집계합니다.
    
    for data in metric_logger.log_every(data_loader, 10, header):
        # input data to cuda
        for key in data['net_input'].keys():
            if isinstance(data['net_input'][key], torch.Tensor):
                data['net_input'][key] = data['net_input'][key].to(device, non_blocking=True)
        for target_key in ['target', 'labels', 'label', 'qid', 'image_id']:
            if target_key in data and isinstance(data[target_key], torch.Tensor):
                data[target_key] = data[target_key].to(device, non_blocking=True)
            
        with torch.cuda.amp.autocast():
            results = handler.valid_step(model=model, samples=data)
            
            # results 딕셔너리에서 필요한 메트릭을 추출하여 metric_logger에 업데이트합니다.
            # 각 핸들러의 valid_step이 반환하는 딕셔너리 키에 따라 다름.
            if 'loss' in results:
                metric_logger.update(loss=results['loss'])
            if 'accuracy' in results: # VQAv2FourChoiceHandler, NLVR2Handler
                metric_logger.meters['accuracy'].update(results['accuracy'], n=results['total_predictions'])
            if 'acc' in results: # NLVR2Handler (for NLVR2, if it uses 'acc' key)
                metric_logger.meters['acc'].update(results['acc'], n=results['total_predictions'])
            if 'acc1' in results: # ImageNetHandler
                metric_logger.meters['acc1'].update(results['acc1'], n=results['total_predictions'])
            if 'acc5' in results: # ImageNetHandler
                metric_logger.meters['acc5'].update(results['acc5'], n=results['total_predictions'])
            if 'vqa_score' in results: # VQAv2Handler
                metric_logger.meters['vqa_score'].update(results['vqa_score'], n=results['qid'].shape[0] if 'qid' in results else 1)
            
            # VQAv2Handler의 test set 예측 또는 Captioning Handler의 예측 저장
            if 'predictions' in results and isinstance(results['predictions'], list):
                handler.predictions.extend(results['predictions'])

    metric_logger.synchronize_between_processes()

    return handler.after_eval(metric_logger=metric_logger)