# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

from engine_for_finetuning import train_one_epoch, get_handler, evaluate
from datasets import create_downstream_dataset, get_sentencepiece_model_for_beit3 # get_sentencepiece_model_for_beit3 추가
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune # modeling_finetune은 이미 임포트되어 있음

# --- START OF NEW IMPORTS FOR VQAv2_4choice TASK ---
# Fairseq 관련 구조를 직접 임포트하지 않고,
# 필요한 클래스들을 직접 정의하거나, utils.py에서 정의된 것을 사용하는 방식인 것으로 보입니다.
# `engine_for_finetuning.py`에 `get_handler`가 이미 태스크별 로직을 분기하고 있으므로,
# 새로운 태스크 핸들러를 정의하고 그에 맞는 evaluate/train_one_epoch 로직을 사용해야 합니다.
# 여기서는 VQAv2_4choice가 일반 분류와 유사하게 처리된다고 가정하고 `get_handler`를 조정합니다.

# Fairseq의 Task/Criterion/Dataset 구조를 직접 따르지 않으므로,
# 여기서는 `get_handler` (engine_for_finetuning.py에 있음)가 `vqav2_4choice`를 처리하도록
# `engine_for_finetuning.py`를 수정해야 합니다.
# 현재 이 파일은 태스크 객체를 직접 생성하는 것이 아니라,
# `create_model`을 통해 모델을 만들고, `create_downstream_dataset`을 통해 데이터셋을 만들며,
# `get_handler`를 통해 태스크별 로직을 호출하는 방식입니다.
# --- END OF NEW IMPORTS ---


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet', 'vqav2_4choice'], # 'vqav2_4choice' 추가
                        help='Name of task to fine-tuning')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=None, 
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str, required=True, 
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # augmentation parameters for imagenet finetuning
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed) # This line is commented in original code, keeping it commented.

    cudnn.benchmark = True

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # DataLoader 생성
    # VQAv2FourChoiceDataset은 args.task가 'vqav2_4choice'일 때 create_downstream_dataset 내부에서 선택됩니다.
    data_loader_train, data_loader_val = create_downstream_dataset(args)

    # --- START OF MODEL CONFIG LOGIC MODIFICATION ---
    # `modeling_finetune.py`에 추가한 4지선다형 모델을 로드하기 위한 로직
    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        elif args.task == "vqav2_4choice": # 새로운 4지선다형 VQA 태스크
            model_config = "%s_vqav2_4choice" % args.model # modeling_finetune.py에 등록된 새 모델 이름
        else: # 기존 VQAv2 등 다른 태스크
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)
    # --- END OF MODEL CONFIG LOGIC MODIFICATION ---

    model = create_model(
        model_config, # 여기에 수정된 model_config가 사용됩니다.
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    if args.finetune:
        utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train.dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lrs)
    elif args.task_head_lr_weight > 1:
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # --- START OF TASK HANDLER AND EVALUATION LOGIC MODIFICATION ---
    # VQAv2_4choice는 일반 분류처럼 정확도 평가를 따르도록 합니다.
    # get_handler는 engine_for_finetuning.py에 정의되어 있으며,
    # 여기에서 VQAv2_4choice 태스크를 위한 특정 로직을 추가해야 합니다.
    task_handler = get_handler(args) 

    # mixup for imagenet
    mixup_fn = None
    if args.task in ["imagenet", "in1k"]:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.nb_classes)

    if args.eval:
        data_loader_test = create_downstream_dataset(args, is_eval=True)
        if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet", "vqav2_4choice"]:
            ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler, args) # <-- args 인자 추가
            print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
            exit(0)
        elif args.task == "vqav2":
            result, _ = evaluate(data_loader_test, model, device, task_handler, args) # <-- args 인자 추가
            utils.dump_predictions(args, result, "vqav2_test")
            exit(0)
        elif args.task in ["coco_captioning", "nocaps"]:
            predictions, _ = evaluate(data_loader_test, model, device, task_handler, args)
            prediction_file = utils.dump_predictions(args, predictions, "{}_test".format(args.task))
            if utils.is_main_process() and args.task == "coco_captioning":
                captioning_result = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_test".format(args.task))
                result_file = os.path.join(args.output_dir, f"{args.task}_result.json")
                print(json.dumps(captioning_result))
                utils.write_result_to_jsonl(captioning_result, result_file)
            exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, task_handler, epoch, 
            epoch * num_training_steps_per_epoch, lr_schedule_values, loss_scaler, 
            args.clip_grad, args.update_freq, model_ema, log_writer, args, # args는 이미 추가됨
            mixup_fn=mixup_fn, # <--- 여기에 mixup_fn 인자를 추가합니다!
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            # VQAv2_4choice는 일반 분류 평가 로직을 따르도록 합니다.
            if args.task not in ["coco_captioning", "nocaps", "vqav2"]:
                test_stats, task_key = evaluate(data_loader_val, model, device, task_handler, args) # <-- args 인자 추가
            elif args.task == "vqav2":
                result, _ = evaluate(data_loader_val, model, device, task_handler, args) # <-- args 인자 추가
                # VQAv2 평가 로직은 복잡하므로, 여기서는 prediction 덤프만 수행하고
                # 실제 VQA 평가는 `evaluate` 함수 내부나 별도 스크립트에서 이루어지는지 확인해야 합니다.
                # 임시로 dump_predictions를 사용합니다.
                _ = utils.dump_predictions(args, result, f"{args.task}_val_e{epoch}")
                # 정확도 계산을 위해 task_key를 "accuracy" 등으로 설정 (engine_for_finetuning에서 반환하는 값에 따라 다름)
                # 여기서는 VQAv2의 복잡한 평가를 단순화하여, VQAv2FourChoice와 유사하게 accuracy를 사용한다고 가정합니다.
                task_key = "accuracy" # 또는 실제 VQAv2 평가 지표 (VQA score)에 따라 변경
                if utils.is_main_process():
                    # VQA score 계산 로직 (기존 VQAv2Dataset 및 eval 함수에 따라 다름)
                    # 여기서는 test_stats가 이미 VQA score를 포함한다고 가정합니다.
                    test_stats = result # evaluate 함수가 직접 결과를 반환한다고 가정
                    print(f"VQA score for VQAv2: {test_stats.get('overall_accuracy', 0):.2f}%")
                torch.distributed.barrier()
                if not utils.is_main_process():
                    # 다른 프로세스에서 결과 읽기 (필요 시)
                    pass # test_stats = utils.read_result_from_jsonl(result_file)

            else: # Captioning
                predictions, _ = evaluate(data_loader_val, model, device, task_handler, args)
                prediction_file = utils.dump_predictions(args, predictions, f"{args.task}_val_e{epoch}")
                result_file = os.path.join(args.output_dir, f"{args.task}_result_val_e{epoch}.json")
                task_key = "CIDEr"
                if utils.is_main_process():
                    test_stats = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_val".format(args.task))
                    utils.write_result_to_jsonl(test_stats, result_file)
                torch.distributed.barrier()
                if not utils.is_main_process():
                    test_stats = utils.read_result_from_jsonl(result_file)

            print(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
            if max_accuracy < test_stats[task_key]:
                max_accuracy = test_stats[task_key]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max performance: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(acc=test_stats[task_key], head="perf", step=epoch)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else: # If no validation loader
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                          'epoch': epoch,
                          'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)