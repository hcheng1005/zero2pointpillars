
import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars
from loss import Loss


def main(args):
    setup_seed()
    
    # 加载训练数据集
    train_dataset = Kitti(data_root=args.data_root, split='train')
    
    # 加载验证数据集
    val_dataset = Kitti(data_root=args.data_root, split='val')