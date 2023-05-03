import sys

import numpy as np
from tqdm import tqdm
import torch

from emetrics import regression_scores, classified_scores
from multi_train_utils.distributed_utils import reduce_value, is_main_process

torch.autograd.set_detect_anomaly(True)

def regression_train_one_epoch_skt_version(model, optimizer, data_loader, criterion, device, epoch, logs):
    '''notice： 回归问题训练集上不保存注意力矩阵，不保存预测值和真实值矩阵'''
    model.train()
    train_y_true = []
    train_y_pred = []
    mean_loss = 0.0
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        features, labels = data

        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)
        labels = labels.to(device)

        logits, cpd_enc_attn, prt_enc_attn, cpd_prt_attn = model(cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix,
                                                                 prt_aa_features, prt_contact_map, prt_dist_matrix)

        loss = criterion(logits, labels)
        loss.backward()

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        train_y_true += labels.to('cpu').numpy().flatten().tolist()
        train_y_pred += logits.to('cpu').detach().numpy().flatten().tolist()

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("nan gradient found")
                print("name:", name)
                print("param:", param.grad)
                raise SystemExit

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Train: [Epoch {}] [lr {}]\t[loss {:.3f}]\t".format(epoch,
                                                                                   optimizer.param_groups[0]["lr"],
                                                                                   loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    mean_loss /= (step + 1)
    train_scores = regression_scores(train_y_true, train_y_pred)
    save_log([epoch, optimizer.param_groups[0]["lr"], mean_loss, train_scores['rmse'], train_scores['mse'],
              train_scores['pearson'], train_scores['spearman'], train_scores['ci']], logs)
    ###################释放内存
    torch.cuda.empty_cache()

    return train_scores, mean_loss


@torch.no_grad()
def regression_evaluate_skt_version(model, data_loader, criterion, device, epoch, logs, a_v_dir):
    '''notice： 回归问题预测集上保存注意力矩阵，也保存预测值和真实值矩阵'''
    model.eval()

    test_y_true = []
    test_y_pred = []
    cids,uniprot_ids = [],[]
    mean_loss = 0.0

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        features, labels = data
        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)

        cid = features[6]
        uniprot_id = features[7]

        labels = labels.to(device)

        logits, cpd_enc_attn, prt_enc_attn, cpd_prt_attn = model(cpd_atom_features, cpd_adj_matrix,
                                                                 cpd_dist_matrix, prt_aa_features,
                                                                 prt_contact_map, prt_dist_matrix)

        cpd_enc_attn_list = [x.to('cpu').detach().numpy() for x in cpd_enc_attn]
        prt_enc_attn_list = [x.to('cpu').detach().numpy() for x in prt_enc_attn]
        cpd_prt_attn_list = [x.to('cpu').detach().numpy() for x in cpd_prt_attn]
        if step < 8:
            np.savez(a_v_dir[0] + 'regression_eval_attn_' + str(step), np.array(cpd_enc_attn_list),
                     np.array(prt_enc_attn_list),
                     np.array(cpd_prt_attn_list))

        loss = criterion(logits, labels.to(device))

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        test_y_true += labels.to('cpu').numpy().flatten().tolist()
        test_y_pred += logits.to('cpu').detach().numpy().flatten().tolist()
        cids += cid.tolist()
        uniprot_ids += uniprot_id.tolist()

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Test: [Epoch {}] [loss {:.3f}]\t".format(epoch, mean_loss / (step + 1))

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    mean_loss /= (step + 1)
    test_scores = regression_scores(test_y_true, test_y_pred)
    save_log([epoch, mean_loss, test_scores['rmse'], test_scores['mse'],
              test_scores['pearson'], test_scores['spearman'], test_scores['ci']], logs)
    # save attention
    np.savez(a_v_dir[1] + 'prd_and_lab', np.array(test_y_true), np.array(test_y_pred),np.array(cids),np.array(uniprot_ids))
    ###################释放内存
    torch.cuda.empty_cache()

    return test_scores, mean_loss


def classified_train_one_epoch_skt_version(model, optimizer, data_loader, criterion, device, epoch, logs):
    '''notice： 分类问题训练集上不保存注意力矩阵，不保存预测值和真实值矩阵'''
    model.train()
    train_y_true = []
    train_y_pred = []
    train_y_score = []
    mean_loss = 0.0
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 依次获取所有，参与模型训练或测试
    for step, data in enumerate(data_loader):
        features, labels = data

        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)
        labels = labels.to(device)

        logits, cpd_enc_attn_list, prt_enc_attn_list, cpd_prt_attn_list = model(cpd_atom_features, cpd_adj_matrix,
                                                                                cpd_dist_matrix, prt_aa_features,
                                                                                prt_contact_map, prt_dist_matrix)
        # 计算Loss值
        loss = criterion(logits, labels)
        # 反传梯度
        loss.backward()

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        pred = logits.argmax(dim=1)
        score = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值

        train_y_true += labels.to('cpu').numpy().flatten().tolist()
        train_y_pred += pred.to('cpu').detach().numpy().flatten().tolist()
        train_y_score += score.to('cpu').detach().numpy().flatten().tolist()

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("nan gradient found")
                print("name:", name)
                print("param:", param.grad)
                raise SystemExit
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Train: [Epoch {}] [lr {}]\t[loss {:.3f}]\t".format(epoch,
                                                                                   optimizer.param_groups[0]["lr"],
                                                                                   loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    mean_loss /= (step + 1)
    train_scores = classified_scores(y_true=train_y_true, y_score=train_y_score, y_pred=train_y_pred)
    save_log([epoch, optimizer.param_groups[0]["lr"], mean_loss, train_scores['auc'], train_scores['acc'],
              train_scores['precision'], train_scores['recall'], train_scores['f1']], logs)
    ###################释放内存
    torch.cuda.empty_cache()

    return train_scores, mean_loss


@torch.no_grad()
def classified_evaluate_skt_version(model, data_loader, criterion, device, epoch, logs, a_v_dir):
    '''notice： 分类问题测试集上只保存注意力矩阵，不保存预测值和真实值矩阵'''
    model.eval()
    test_y_true = []
    test_y_pred = []
    test_y_score = []
    cids,uniprot_ids = [],[]
    mean_loss = 0.0

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        features, labels = data

        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)

        cid = features[6]
        uniprot_id = features[7]

        labels = labels.to(device)

        logits, cpd_enc_attn, prt_enc_attn, cpd_prt_attn = model(cpd_atom_features, cpd_adj_matrix,
                                                                 cpd_dist_matrix, prt_aa_features,
                                                                 prt_contact_map, prt_dist_matrix)

        cpd_enc_attn_list = [x.to('cpu').detach().numpy() for x in cpd_enc_attn]
        prt_enc_attn_list = [x.to('cpu').detach().numpy() for x in prt_enc_attn]
        cpd_prt_attn_list = [x.to('cpu').detach().numpy() for x in cpd_prt_attn]
        np.savez(a_v_dir[0] + 'classified_eval_attn_' + str(step), np.array(cpd_enc_attn_list),
                 np.array(prt_enc_attn_list),
                 np.array(cpd_prt_attn_list))
        # 计算Loss值
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值

        test_y_true += labels.to('cpu').numpy().flatten().tolist()
        test_y_pred += preds.to('cpu').detach().numpy().flatten().tolist()
        test_y_score += scores.to('cpu').detach().numpy().flatten().tolist()

        cids += cid.tolist()
        uniprot_ids += uniprot_id.tolist()

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Test: [Epoch {}] [loss {:.3f}]\t".format(epoch, mean_loss / (step + 1))
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    test_scores = classified_scores(y_true=test_y_true, y_score=test_y_score, y_pred=test_y_pred)
    mean_loss /= (step + 1)

    save_log([epoch, mean_loss, test_scores['auc'], test_scores['acc'], test_scores['precision'],
              test_scores['recall'], test_scores['f1']], logs)
    # save attention
    np.savez(a_v_dir[1] + 'classified_prd_and_lab', np.array(test_y_true), np.array(test_y_pred), np.array(cids),
             np.array(uniprot_ids))

    ###################释放内存
    torch.cuda.empty_cache()

    return test_scores, mean_loss


def classified_train_one_epoch(model, optimizer, data_loader, criterion, device, epoch, logs):
    '''notice： 分类问题训练集上不保存注意力矩阵，不保存预测值和真实值矩阵'''
    #  调用模型训练
    model.train()

    from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
    auc = BinaryAUROC(thresholds=None).to(device)
    acc = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    mean_loss = 0.0

    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 依次获取所有，参与模型训练或测试
    for step, data in enumerate(data_loader):
        features, labels = data

        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)
        labels = labels.to(device)

        logits, cpd_enc_attn_list, prt_enc_attn_list, cpd_prt_attn_list = model(cpd_atom_features, cpd_adj_matrix,
                                                                                cpd_dist_matrix, prt_aa_features,
                                                                                prt_contact_map, prt_dist_matrix)
        # 计算Loss值
        loss = criterion(logits, labels)
        # 反传梯度
        loss.backward()

        preds = logits.argmax(dim=1)
        scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值

        t_auc = auc(preds, labels)
        t_acc = acc(scores, labels)
        t_prec = precision(preds, labels)
        t_rec = recall(preds, labels)
        t_f1 = f1(preds, labels)

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("nan gradient found")
                print("name:", name)
                print("param:", param.grad)
                raise SystemExit
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Train: [Epoch {}] [lr {}] [loss {:.3f}] [auc {:.3f}] [acc {:.3f}] [prec {:.3f}] [rec {:.3f}] [f1 {:.3f}]\t".format(
                epoch, optimizer.param_groups[0]["lr"], loss.item(), t_auc.item(), t_acc.item(), t_prec.item(),
                t_rec.item(), t_f1.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    total_auc = auc.compute()
    total_acc = acc.compute()
    total_precision = precision.compute()
    total_recall = recall.compute()
    total_f1 = f1.compute()

    train_scores = {'auc': total_auc.item(), 'acc': total_acc.item(), 'precision': total_precision.item(),
                    'recall': total_recall.item(), 'f1': total_f1.item()}
    mean_loss /= (step + 1)

    save_log([epoch, mean_loss, train_scores['auc'], train_scores['acc'], train_scores['precision'],
              train_scores['recall'], train_scores['f1']], logs)

    auc.reset()
    acc.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    ###################释放内存
    torch.cuda.empty_cache()

    return train_scores, mean_loss


@torch.no_grad()
def classified_evaluate(model, data_loader, criterion, device, epoch, logs, a_v_dir):
    '''notice： 分类问题测试集上只保存注意力矩阵，不保存预测值和真实值矩阵'''
    model.eval()
    from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
    auc = BinaryAUROC(thresholds=None).to(device)
    acc = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    mean_loss = 0.0

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        features, labels = data

        cpd_atom_features = torch.tensor(features[0], dtype=torch.float32, device=device)
        cpd_adj_matrix = torch.tensor(features[1], dtype=torch.float32, device=device)
        cpd_dist_matrix = torch.tensor(features[2], dtype=torch.float32, device=device)

        prt_aa_features = torch.tensor(features[3], dtype=torch.float32, device=device)
        prt_contact_map = torch.tensor(features[4], dtype=torch.float32, device=device)
        prt_dist_matrix = torch.tensor(features[5], dtype=torch.float32, device=device)

        labels = labels.to(device)

        logits, cpd_enc_attn, prt_enc_attn, cpd_prt_attn = model(cpd_atom_features, cpd_adj_matrix,
                                                                 cpd_dist_matrix, prt_aa_features,
                                                                 prt_contact_map, prt_dist_matrix)

        cpd_enc_attn_list = [x.to('cpu').detach().numpy() for x in cpd_enc_attn]
        prt_enc_attn_list = [x.to('cpu').detach().numpy() for x in prt_enc_attn]
        cpd_prt_attn_list = [x.to('cpu').detach().numpy() for x in cpd_prt_attn]
        np.savez(a_v_dir[0] + 'classified_eval_attn_' + str(step), np.array(cpd_enc_attn_list),
                 np.array(prt_enc_attn_list),
                 np.array(cpd_prt_attn_list))
        # 计算Loss值
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值

        auc.update(preds, labels)
        acc.update(scores, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1.update(preds, labels)

        reduced_loss = reduce_value(loss.data, average=True)
        mean_loss += reduced_loss.item()

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Test: [Epoch {}] [loss {:.3f}]\t".format(epoch, mean_loss / (step + 1))
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    total_auc = auc.compute()
    total_acc = acc.compute()
    total_precision = precision.compute()
    total_recall = recall.compute()
    total_f1 = f1.compute()

    test_scores = {'auc': total_auc.item(), 'acc': total_acc.item(), 'precision': total_precision.item(),
                   'recall': total_recall.item(), 'f1': total_f1.item()}
    mean_loss /= (step + 1)

    save_log([epoch, mean_loss, test_scores['auc'], test_scores['acc'], test_scores['precision'],
              test_scores['recall'], test_scores['f1']], logs)

    auc.reset()
    acc.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    ###################释放内存
    torch.cuda.empty_cache()

    return test_scores, mean_loss


def save_log(data, logs_file, mode='a'):
    entries = [str(meter) for meter in data]
    with open(logs_file, mode) as f:
        f.write('\t'.join(entries) + '\n')
