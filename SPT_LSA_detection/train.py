from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy, IoULoss, FocalLoss
import os, gc
from utils.sampler import RASampler, list_collate
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload, collate_fn
from models.create_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=Warning)
import utils.config as config

from matplotlib.pyplot import sca
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from utils.losses import DetectionLoss
best_acc1 = 0
MODELS = ['vit', 'swin', 't2t']


def init_parser():
    parser = argparse.ArgumentParser(description='COCO quick training script')
    # Data args
    parser.add_argument('--data_path', default='../dataset', type=str, help='dataset path')
    parser.add_argument('--coco_path', default= '/vit/dataset/coco', type=str, help='coco path')    
    parser.add_argument('--dataset', default='COCO', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'SVHN', 'COCO'], type=str, help='Image Net dataset path')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')
    # Optimization hyperparams  
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')    
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')    
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--model', type=str, default='swin', choices=MODELS)
    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')
    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--ls', action='store_false', help='label smoothiping')
    parser.add_argument('--channel', type=int, help='disable cuda')
    parser.add_argument('--heads', type=int, help='disable cuda')
    parser.add_argument('--depth', type=int, help='disable cuda')
    parser.add_argument('--tag', type=str, help='tag', default='')
    parser.add_argument('--seed', type=int, default=0, help='seed')    
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')    
    parser.add_argument('--resume', default=False, help='Version')       
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--cm',action='store_false' , help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu',action='store_false' , help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    parser.add_argument('--is_LSA', action='store_false', help='Locality Self-Attention')    
    parser.add_argument('--is_SPT', action='store_false', help='Shifted Patch Tokenization')

    return parser




torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader)
    losses = []
    
    for batch_idx, (x, y) in enumerate(loop):
        # print(f'x: {x}, y:{y}')
        x = x.cuda()
        y0, y1, y2 = (
            y[0].cuda(),
            y[1].cuda(),
            y[2].cuda()
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            print(out.shape)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main(args):
    global best_acc1
    torch.cuda.set_device(args.gpu)
    data_info = datainfo(logger, args)

    model = create_model(data_info['img_size'], data_info['n_classes'], args)
    
    model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = DetectionLoss()
    scaler = torch.cuda.amp.GradScaler()

    # dataloader.py COCODataset loader
    # train_loader, test_loader, train_eval_loader = get_loaders(
    #     args, train_json_path=os.path.join(args.coco_path, 'annotations', 'instances_train2017.json'),
    #     test_json_path=None,
    #     val_json_path=os.path.join(args.coco_path, 'annotations', 'instances_val2017.json'),
    #     data_info=data_info
    # )

    # coco_loader.py CocoDataset loader
    train_loader, test_loader, train_eval_loader = get_loaders(args)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE,
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)


        if epoch % 10 == 0 and epoch > 0:
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer)

            print("On Test Loader:")
            check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                train_eval_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,   
            )
            print(f"mAP: {mapval.item()}")

    save_checkpoint(model, optimizer)
    print(f"mAP: {mapval.item()}")
    print("Finish the Training")
    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    gc.collect()
    torch.cuda.empty_cache()
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model

    if not args.is_SPT:
        model_name += "-Base"
    else:
        model_name += "-SPT"
 
    if args.is_LSA:
        model_name += "-LSA"
        
    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    
    main(args)
