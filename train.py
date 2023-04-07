import os
import logging
import random
import warnings
import numpy as np
from dataset import create_dataset
from config.config import parse_commandline_args, save_config_yaml
from rda import RDA
from net import resnet50
from mindspore import nn
import mindspore as ms

config_file = "./config/rda.yaml"

def main(args):

    # ms.set_context(device_target="GPU")
    # ms.set_context(device_id=args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    lb_dst, ulb_dst, val_dst = create_dataset(args.dataset, args.num_labels, args.num_classes, args.batch_size)
    step_size_train = ulb_dst.get_dataset_size()
    step_size_val = val_dst.get_dataset_size()

    # dataloader
    lb_loader = lb_dst.create_tuple_iterator(num_epochs=args.num_train_iter)
    ulb_loader = ulb_dst.create_tuple_iterator(num_epochs=args.num_train_iter)
    val_loader = val_dst.create_tuple_iterator()

    train_model = resnet50(num_classes=10, pretrained=False)
    eval_model = resnet50(num_classes=10, pretrained=False)

    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.03, total_step=args.num_train_iter,
                       step_per_epoch=1024, decay_epoch=args.epoch)

    model = RDA(args.num_classes, args.ema_m, args.gpu, train_model, eval_model,
                     args.epoch, args.T, args.p_cutoff, 1.0,
                 args.min_clust_K, args.max_clust_K, 1.0, args.clust_cutoff, args.clust_cont_temp,
                     lb_loader, ulb_loader, val_loader, lr, num_train_iter=args.num_train_iter)
    model.train()

if __name__ == "__main__":
    my_args = parse_commandline_args(filepath=config_file)

    main(my_args)