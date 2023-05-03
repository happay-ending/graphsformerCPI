import os
import math
import tempfile
import argparse
from glob import glob

import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from data_process import split_classified_data, move_file
from graphsformerCPI import graphsformerCPI
from my_dataset import Classified_DataSet
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import classified_train_one_epoch, classified_evaluate, \
    classified_evaluate_skt_version, \
    classified_train_one_epoch_skt_version

'''
## 多GPU启动指令
- 如果要使用```classified_multi_gpu.py```脚本，使用以下指令启动
- python -m torch.distributed.launch --nproc_per_node=2 --use_env classified_multi_gpu.py
- 其中```nproc_per_node```为并行GPU的数量
- 如果要指定使用某几块GPU可使用如下指令，例如使用第1块和第4块GPU进行训练：
- CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env classified_multi_gpu.py
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25641 --use_env classified_multi_gpu.py
'''


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_sample_ids, val_sample_ids, cpd_max_len, prt_max_len = split_classified_data(
        root=args.data_path, rank=0, train_sample_ratio=0.8, cpd_threshold=args.cpd_threshold,
        prt_threshold=args.prt_threshold)

    # 实例化训练数据集
    train_data_set = Classified_DataSet(root=args.data_path, sample_ids=train_sample_ids, cpd_cut_len=cpd_max_len,
                                        prt_cut_len=prt_max_len)

    val_data_set = Classified_DataSet(root=args.data_path, sample_ids=val_sample_ids, cpd_cut_len=cpd_max_len,
                                      prt_cut_len=prt_max_len)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    val_batch_sampler = torch.utils.data.BatchSampler(
        val_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_sampler=val_batch_sampler,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    cpd_atom = 78  # Drug embedding dimension
    prt_aa = 50  # Protein embedding dimension
    model_params = {
        'cpd_atom': cpd_atom,  # compound raw feature dimension
        'prt_aa': prt_aa,  # protein aa raw feature dimension
        'bitch_size': batch_size,
        'layers': args.model_layers,  # layers 6
        'd_model': 128,  # Embedded feature dimension 512
        'n_heads': 4,  # head 8
        'dropout': args.dropout,
        'distance_matrix_kernel': 'softmax',  # softmax,exp,sigmoid
        'd_ffn': 1024,  # 2048
        'n_output': 2,
        'activation_fun': 'softmax'  # softmax,relu,sigmoid
    }

    # 实例化模型
    model = graphsformerCPI(**model_params).to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型，可加’find_unused_parameters=True‘参数，检查模型前向传播中没有使用的参数。
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    # 优化器0：实测对此场景效果不好
    optimizer_0 = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler_0 = lr_scheduler.LambdaLR(optimizer_0, lr_lambda=lf)
    # 优化器1
    optimizer_1 = torch.optim.Adam(pg, lr=args.lr)
    scheduler_1 = lr_scheduler.LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1 / (epoch + 1))
    # 优化器2
    optimizer_2 = torch.optim.Adam(pg, lr=args.lr)
    scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=2, gamma=0.5, last_epoch=-1)
    # 优化器3
    optimizer_3 = torch.optim.Adam(pg, lr=args.lr)
    scheduler_3 = lr_scheduler.ReduceLROnPlateau(optimizer_3, mode='min', factor=0.5, patience=4)
    # 优化器4
    # optimizer_inner = torch.optim.Adam(pg, lr=args.lr)
    # optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    optimizers = {"optim_0": [optimizer_0, scheduler_0], "optim_1": [optimizer_1, scheduler_1],
                  "optim_2": [optimizer_2, scheduler_2], "optim_3": [optimizer_3, scheduler_3]}

    optimizer, scheduler = optimizers[args.optimizer_scheduler]
    print("初始化的学习率：", optimizer.param_groups[0]['lr'])

    loss_function = torch.nn.CrossEntropyLoss()

    if not os.path.exists(args.root_log):
        print('creating folder ' + args.root_log)
        os.mkdir(args.root_log)

    attn_dir = './output/attn/'
    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)  # 创建路径,存放attn矩阵

    best_attn = './output/best_attn/'
    if not os.path.exists(best_attn):
        os.makedirs(best_attn)  # 创建路径,存放best_attn矩阵

    pred_true_value_dir = './output/PandT/'
    if not os.path.exists(pred_true_value_dir):
        os.makedirs(pred_true_value_dir)  # 创建路径,存放pred_true矩阵
    pred_true_best_dir = './output/best_PandT/'
    if not os.path.exists(pred_true_best_dir):
        os.makedirs(pred_true_best_dir)  # 创建路径,存放best_pred_true_矩阵

    a_v_dir = [attn_dir, pred_true_value_dir]

    log_file_name = "epochs(" + str(args.epochs) + ")_Bchsize(" + str(args.batch_size) + ")_layer(" + str(
        model_params["layers"]) + ")_dropout(" + str(args.dropout) + ")"
    logs_train = os.path.join(args.root_log, args.data_path.split('/')[
        -1] + '_train_' + args.optimizer_scheduler + '_' + log_file_name + '.txt')
    logs_test = os.path.join(args.root_log, args.data_path.split('/')[
        -1] + '_test_' + args.optimizer_scheduler + '_' + log_file_name + '.txt')

    best_metric = -100000
    for epoch in range(args.epochs):
        # 在分布式模式下，调用set_epoch()每个时期开始时的方法前创建DataLoader迭代器是使混洗跨多个时期正常工作所必需的。否则，将始终使用相同的顺序。
        train_sampler.set_epoch(epoch)

        train_scores, train_mean_loss = classified_train_one_epoch_skt_version(model=model,
                                                                               optimizer=optimizer,
                                                                               data_loader=train_loader,
                                                                               criterion=loss_function,
                                                                               device=device,
                                                                               epoch=epoch,
                                                                               logs=logs_train)

        if rank == 0:
            print(
                "Train: [Epoch {}] [mean_loss:{:3f}]\t[auc:{:3f}]\t[acc:{:3f}]\t[prec:{:3f}]\t[rec:{:3f}]\t[f1:{:3f}]".format(
                    epoch, train_mean_loss, train_scores['auc'], train_scores['acc'], train_scores['precision'],
                    train_scores['recall'], train_scores['f1']))

        if args.optimizer_scheduler == 'optim_3':
            scheduler.step(train_mean_loss)
        else:
            scheduler.step()

        test_scores, test_mean_loss = classified_evaluate_skt_version(model=model, data_loader=val_loader,
                                                                      criterion=loss_function,
                                                                      device=device, epoch=epoch, logs=logs_test,a_v_dir=a_v_dir)

        if rank == 0:
            print(
                "Test: [Epoch {}] [mean_loss:{:3f}]\t[auc:{:3f}]\t[acc:{:3f}]\t[prec:{:3f}]\t[rec:{:3f}]\t[f1:{:3f}]".format(
                    epoch, test_mean_loss, test_scores['auc'], test_scores['acc'], test_scores['precision'],
                    test_scores['recall'], test_scores['f1']))
            tags = ["loss", "auc", "acc", "prec", "rec", "f1", "learning_rate"]
            tb_writer.add_scalar(tags[0], test_mean_loss, epoch)
            tb_writer.add_scalar(tags[1], test_scores['auc'], epoch)
            tb_writer.add_scalar(tags[2], test_scores['acc'], epoch)
            tb_writer.add_scalar(tags[3], test_scores['precision'], epoch)
            tb_writer.add_scalar(tags[4], test_scores['recall'], epoch)
            tb_writer.add_scalar(tags[5], test_scores['f1'], epoch)
            tb_writer.add_scalar(tags[6], optimizer.param_groups[0]["lr"], epoch)

            if best_metric < test_scores['auc']:
                best_metric = test_scores['auc']
                torch.save(model.module.state_dict(),
                           "./weights/best_model({}_{}-epoch).pth".format(args.data_path.split('/')[-1], epoch))

                src_file_list = glob(attn_dir + '*.npz')  # glob获得路径下所有文件
                for srcfile in src_file_list:
                    move_file(srcfile, best_attn)

                src_file_list = glob(pred_true_value_dir + '*.npz')
                for srcfile in src_file_list:
                    move_file(srcfile, pred_true_best_dir)
    # 删除临时缓存文件
    if rank == 0:
        tb_writer.close()
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--optimizer_scheduler', type=str, default='optim_3',
                        help='Select preset optimizer and scheduler')
    parser.add_argument('--prt_threshold', type=float, default=0.9,
                        help='Maximum length threshold of protein sequence (default: 0.8)')
    parser.add_argument('--cpd_threshold', type=float, default=1.0,
                        help='Maximum number of atoms threshold of compound (default: 0.9995)')
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="./data/human")  # human,celegans
    parser.add_argument('--fold', type=int, default=1)  # fold: 验证， fold! :训练

    parser.add_argument('--weights', type=str, default='graphsformerCPI.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),在单机中指使用GPU的数量
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
