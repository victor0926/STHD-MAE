import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
# import M_1DMae_try01
# import M24121001_1DMaeHe_2L
# import M24122501_2to1DMAE_AimMask_Model
# import M25011904_FCPatchSTA_clsTokenClass_Model
# import M25061902_STHDMAE_Class_Model
import M25062603_STHDMAE_Class_Model
# from M24122501_2to1DMAE_AimMask_Model import MaskedAutoencoderViT
# from E24123101_Ori2DMAE_RandMask_PretrainEngin import train_one_epoch
# from He_utils.misc import NativeScalerWithGradNormCount as NativeScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import tqdm
import torch.nn as nn
import random
import pickle
from scipy.signal import resample


# 随机种子
def seed_torch(seed=233):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 保存模型
def save_model(net, filename):
    torch.save(net.state_dict(), filename)
    print("已将预训练模型保存到{}".format(filename))

# 验证、测试结果
def round_compute(prev):
    prev1 = np.zeros((len(prev), len(prev[0])))
    for i in range(len(prev)):
        all_negative = np.all(prev[i] < 0)
        if all_negative:
            max_negative_index = np.argmax(prev[i])
            prev1[i][max_negative_index] = 1
        else:
            for j in range(len(prev[i])):
                if prev[i][j] >= 0:
                    prev1[i][j] = 1
                else:
                    prev1[i][j] = 0
    return prev1

class EcgDataset(Dataset):
    def __init__(self, ecg_data, labels):
        self.label = labels
        self.n_data = len(labels)
        self.x_data = ecg_data

    def __getitem__(self, idx):
        ori_data = self.x_data[idx]
        # ori_data[1:] = 0
        # ori_data[1:] = ori_data[0]
        return ori_data, self.label[idx]

    def __len__(self):
        return self.n_data

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit', type=str)
    parser.add_argument('--mask_ratio', default=0, type=float)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=233, type=int)
    parser.add_argument('--output_dir', default='./Models/S120601_1DTest01', type=str)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Number of warmup epochs for learning rate schedule')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower learning rate bound for learning rate scheduler')
    return parser

# 训练
def train(model, optimizer, train_loader, val_loader, filename, epoch, N_CLASSES):
    print('start train')
    loss_class = nn.BCEWithLogitsLoss()
    N_EPOCH = epoch
    best_performance = 0

    for epoch in range(1, N_EPOCH + 1):
        model.train()
        total_loss_train = 0
        len_dataloader = len(train_loader)

        for _, (data, label) in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            data = data.cpu().numpy()
            label = label.cpu().numpy()
            data_ten = torch.tensor(data).float().to(DEVICE)
            label_ten = torch.tensor(label).float().to(DEVICE)
            class_output = model(data_ten, mask_ratio=0, epoch=0)
            classification_loss = loss_class(class_output, label_ten)
            classification_loss.backward()
            optimizer.step()
            total_loss_train += classification_loss.item()

        loss_train = total_loss_train / len_dataloader
        print('Training set:')
        print('Epoch: ' + str(epoch) + ', Loss: ' + str(loss_train))

        model.eval()
        true_value = np.zeros((1, N_CLASSES))
        pred_value = np.zeros((1, N_CLASSES))
        out_value = np.zeros((1, N_CLASSES))

        with torch.no_grad():
            for _, (data, label) in enumerate(tqdm.tqdm(val_loader)):
                data = data.cpu().numpy()
                label = label.cpu().numpy()
                data_ten = torch.tensor(data).float().to(DEVICE)
                label_ten = torch.tensor(label).float().to(DEVICE)
                class_output = model(data_ten, mask_ratio=0, epoch=0)
                class_output_np = class_output.cpu().numpy()
                label_np = label_ten.cpu().numpy()
                true_value = np.concatenate((true_value, label_np))
                pred_value = np.concatenate((pred_value, round_compute(class_output_np)))
                out_value = np.concatenate((out_value, class_output_np))

            true_value = true_value[1:]
            pred_value = pred_value[1:]
            out_value = out_value[1:]
            valid_classes = [i for i in range(true_value.shape[1]) if len(np.unique(true_value[:, i])) > 1]
            true_value_valid = true_value[:, valid_classes]
            out_value_valid = out_value[:, valid_classes]
            auc1 = roc_auc_score(true_value_valid, out_value_valid, average='micro')

            print("AUC:{:.4f}".format(auc1))
            acc1 = accuracy_score(true_value, pred_value)
            print("Overall Acc: {:.4f}".format(acc1))

            category_accuracies = []
            for i in range(true_value.shape[1]):
                y_true = true_value[:, i]
                y_pred = pred_value[:, i]
                accuracy = accuracy_score(y_true, y_pred)
                category_accuracies.append(accuracy)

            average_category_accuracy = np.mean(category_accuracies)
            print("Average Category Acc: {:.4f}".format(average_category_accuracy))

            F1_micro = f1_score(true_value, pred_value, average='micro', zero_division=0)
            print('F1_micro: {:.4f}'.format(F1_micro))

            if auc1 > best_performance:
                best_performance = auc1
                print('best_performance: {:.4f}'.format(best_performance))
                save_filename = filename + '_' + str(epoch)+ '_' + str(int(best_performance * 100000)) + '_' + str(int(acc1 * 100000)) + '_' + str(int(F1_micro * 100000)) + '.pth'
                save_model(model, save_filename)

def test(model, val_loader, N_CLASSES):
    model.eval()
    true_value = np.zeros((1, N_CLASSES))
    pred_value = np.zeros((1, N_CLASSES))
    out_value = np.zeros((1, N_CLASSES))
    # lat_value = np.zeros((1, 241,256))
    # two_value = np.zeros((1, 241,256))
    # ali_value = np.zeros((1, 241,256))
    # layermean_value = np.zeros((1, 8, 241, 241))
    # headmean_value = np.zeros((1, 8, 241, 241))

    with torch.no_grad():
        for _, (data, label) in enumerate(tqdm.tqdm(val_loader)):
            data = data.cpu().numpy()
            label = label.cpu().numpy()
            data_ten = torch.tensor(data).float().to(DEVICE)
            label_ten = torch.tensor(label).float().to(DEVICE)
            class_output, layermean_atts, headmearn_atts = model(data_ten, mask_ratio=0, epoch=0)

            class_output_np = class_output.cpu().numpy()
            # layermean_atts_np = layermean_atts.cpu().numpy()
            # headmean_atts_np = headmearn_atts.cpu().numpy()
            label_np = label_ten.cpu().numpy()
            true_value = np.concatenate((true_value, label_np))
            pred_value = np.concatenate((pred_value, round_compute(class_output_np)))
            out_value = np.concatenate((out_value, class_output_np))
            # layermean_value = np.concatenate((layermean_value, layermean_atts_np))
            # headmean_value = np.concatenate((headmean_value, headmean_atts_np))


        true_value = true_value[1:]
        pred_value = pred_value[1:]
        out_value = out_value[1:]
        # layermean_value = layermean_value[1:]
        # headmean_value = headmean_value[1:]
        valid_classes = [i for i in range(true_value.shape[1]) if len(np.unique(true_value[:, i])) > 1]
        true_value_valid = true_value[:, valid_classes]
        out_value_valid = out_value[:, valid_classes]
        auc1 = roc_auc_score(true_value, out_value, average='micro')

        # out_path = 'Qual_Value/'
        # # # method_path = 'STHDMAE_noMRF_linear_PTBXL5C_'
        # # # method_path = 'STHDMAE_noMRF_finrtune_PTBXL5C_'
        # # # method_path = 'STHDMAE_noMRF_linear_PTBXL44C_'
        # # # method_path = 'STHDMAE_noMRF_finrtune_PTBXL44C_'
        # # # method_path = 'STHDMAE_noMRF_linear_PTBXL19C_'
        # # # method_path = 'STHDMAE_noMRF_finrtune_PTBXL19C_'
        # # # method_path = 'STHDMAE_noMRF_linear_Chapman4C_'
        # # method_path = 'STHDMAE_noMRF_finrtune_Chapman4C_'
        # #
        # # np.save(out_path + method_path + 'true.npy', true_value)
        # # np.save(out_path + method_path + 'out.npy', out_value)
        # # np.save(out_path + method_path + 'pred.npy', pred_value)
        # np.save(out_path + 'ptbxl5c_layermean_att.npy', layermean_value)
        # np.save(out_path + 'ptbxl5c_headmean_att.npy', headmean_value)


        print("AUC:{:.4f}".format(auc1))
        acc1 = accuracy_score(true_value, pred_value)
        print("Overall Acc: {:.4f}".format(acc1))

        category_accuracies = []
        for i in range(true_value.shape[1]):
            y_true = true_value[:, i]
            y_pred = pred_value[:, i]
            accuracy = accuracy_score(y_true, y_pred)
            category_accuracies.append(accuracy)

        average_category_accuracy = np.mean(category_accuracies)
        print("Average Category Acc: {:.4f}".format(average_category_accuracy))

        F1_micro = f1_score(true_value, pred_value, average='micro', zero_division=0)
        print('F1_micro: {:.4f}'.format(F1_micro))
        precision_micro = precision_score(true_value, pred_value, average='micro', zero_division=0)
        recall_micro = recall_score(true_value, pred_value, average='micro', zero_division=0)
        print('Precision_micro: {:.4f}'.format(precision_micro))
        print('Recall_micro: {:.4f}'.format(recall_micro))

        F1s = f1_score(true_value, pred_value, average=None, zero_division=0)
        print(F1s)

def main(args):
    # 固定随机数种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device)

    # # 加载数据

    # dataPath = 'Data/PTBXL_Data/all/data/'
    dataPath = 'Data/PTBXL_Data/diagnostic/data/'
    # dataPath = 'Data/PTBXL_Data/subdiagnostic/data/'
    # dataPath = 'Data/PTBXL_Data/superdiagnostic/data/'
    # dataPath = 'Data/PTBXL_Data/rhythm/data/'
    # dataPath = 'Data/PTBXL_Data/form/data/'

    # dataPath = 'Data/PTBXL_Data/superdiagnostic/data/'

    with open(dataPath + 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    result_train_Ori = train_data["y"]
    result_train_Ori = result_train_Ori.astype(np.float32)

    # route01 = './Data/PTBXL_Data/train5_fixR.npy'
    # X_train_Data = np.load(route01, allow_pickle=True)
    # X_train_Data = torch.FloatTensor(X_train_Data).cuda()
    X_train_Ori = train_data["x"].astype(np.float32)
    X_train_Data = np.array([
        resample(X_train_Ori[i], 1000, axis=0) for i in range(X_train_Ori.shape[0])
    ])
    X_train_Data = np.transpose(X_train_Data, (0, 2, 1))
    X_train_Data = torch.FloatTensor(X_train_Data).cuda()

    O_train_Label = torch.FloatTensor(result_train_Ori).cuda()
    train_dataset = EcgDataset(X_train_Data, O_train_Label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    with open(dataPath + 'val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    result_val_Ori = val_data["y"]
    result_val_Ori = result_val_Ori.astype(np.float32)

    # route02 = './Data/PTBXL_Data/val5_fixR.npy'
    # X_val_Data = np.load(route02, allow_pickle=True)
    # X_val_Data = torch.FloatTensor(X_val_Data).cuda()
    X_val_Ori = val_data["x"].astype(np.float32)
    X_val_Data = np.array([
        resample(X_val_Ori[i], 1000, axis=0) for i in range(X_val_Ori.shape[0])
    ])
    X_val_Data = np.transpose(X_val_Data, (0, 2, 1))
    X_val_Data = torch.FloatTensor(X_val_Data).cuda()

    O_val_Label = torch.FloatTensor(result_val_Ori).cuda()
    val_dataset = EcgDataset(X_val_Data, O_val_Label)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    with open(dataPath + 'test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    result_test_Ori = test_data["y"]
    result_test_Ori = result_test_Ori.astype(np.float32)

    # route03 = './Data/PTBXL_Data/test5_fixR.npy'
    # X_test_Data = np.load(route03, allow_pickle=True)
    # X_test_Data = torch.FloatTensor(X_test_Data).cuda()
    X_test_Ori = test_data["x"].astype(np.float32)
    X_test_Data = np.array([
        resample(X_test_Ori[i], 1000, axis=0) for i in range(X_test_Ori.shape[0])
    ])
    X_test_Data = np.transpose(X_test_Data, (0, 2, 1))
    X_test_Data = torch.FloatTensor(X_test_Data).cuda()

    O_test_Label = torch.FloatTensor(result_test_Ori).cuda()
    test_dataset = EcgDataset(X_test_Data, O_test_Label)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # # route01 = './Data/train_MultiResult.npy'
    # route01 = 'Data/Shaoxing_Data/ShaoXing4_train_labels.npy'
    # result_train_Ori = np.load(route01, allow_pickle=True)
    # route1 = 'Data/Shaoxing_Data/ShaoXing4_train_data.npy'
    # X_train_Ori = np.load(route1, allow_pickle=True)
    # X_train_Data = X_train_Ori[:, :, :]
    # X_train_Data = np.transpose(X_train_Data, (0, 2, 1))
    # X_train_Data = torch.FloatTensor(X_train_Data).cuda()
    # O_train_Label = torch.FloatTensor(result_train_Ori).cuda()
    # train_dataset = EcgDataset(X_train_Data, O_train_Label)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #
    # # route02 = './Data/val_MultiResult.npy'
    # route02 = 'Data/Shaoxing_Data/ShaoXing4_val_labels.npy'
    # result_val_Ori = np.load(route02, allow_pickle=True)
    # route2 = 'Data/Shaoxing_Data/ShaoXing4_val_Data.npy'
    # X_val_Ori = np.load(route2, allow_pickle=True)
    # X_val_Data = X_val_Ori[:, :, :]
    # X_val_Data = np.transpose(X_val_Data, (0, 2, 1))
    # X_val_Data = torch.FloatTensor(X_val_Data).cuda()
    # O_val_Label = torch.FloatTensor(result_val_Ori).cuda()
    # val_dataset = EcgDataset(X_val_Data, O_val_Label)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    #
    # route03 = 'Data/Shaoxing_Data/ShaoXing4_test_labels.npy'
    # result_test_Ori = np.load(route03, allow_pickle=True)
    # route3 = 'Data/Shaoxing_Data/ShaoXing4_test_Data.npy'
    # X_test_Ori = np.load(route3, allow_pickle=True)
    # X_test_Data = X_test_Ori[:, :, :]
    # X_test_Data = np.transpose(X_test_Data, (0, 2, 1))
    # X_test_Data = torch.FloatTensor(X_test_Data).cuda()
    # O_test_Label = torch.FloatTensor(result_test_Ori).cuda()
    # test_dataset = EcgDataset(X_test_Data, O_test_Label)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型
    # model_path = "Models/PreTrainModels/R25062301_STHDMAE_noMRF_BigMimic_Try01/epoch_98_best_0.004262751777423546.pth"
    # model_path = "Models/PreTrainModels/R25062602_STHDMAE_noMRF_BigMimic_Try01/epoch_82_best_0.003884768985872027.pth"
    model_path = "Models/PreTrainModels/R25061201_STHDMAE_500_BigMimic_Try01/epoch_98_best_0.004215971741359681.pth"

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    # 提取模型权重
    # model_weights = checkpoint
    model_weights = checkpoint['model']
    # 加载模型权重
    # model = M25011904_FCPatchSTA_clsTokenClass_Model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model = M25062603_STHDMAE_Class_Model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)
    model.load_state_dict(model_weights, strict=False)

    # 冻结编码器部分的参数
    for param in model.patch_embed.parameters():
        param.requires_grad = False  # 冻结 patch_embed 层的参数
    for param  in model.ff_blocks.parameters():
        param.requires_grad = False  # 冻结 transformer blocks
    for param  in model.blocks_l.parameters():
        param.requires_grad = False  # 冻结 transformer blocks
    for param  in model.blocks_t.parameters():
        param.requires_grad = False  # 冻结 transformer blocks
    for param  in model.blocks.parameters():
        param.requires_grad = False  # 冻结 transformer blocks
    for param in model.norm_l.parameters():
        param.requires_grad = False  # 冻结 patch_embed 层的参数
    for param in model.norm_t.parameters():
        param.requires_grad = False  # 冻结 patch_embed 层的参数
    for param in model.norm.parameters():
        param.requires_grad = False  # 冻结 patch_embed 层的参数
    # 仅训练分类头部分
    for param in model.classification.parameters():
        param.requires_grad = True  # 只训练分类器部分

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #


    # # save_filename = 'Models/FineTuneClass/S25062701_STHDMAE_noMRF_linear_PTBXL5C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062702_STHDMAE_noMRF_finetune_PTBXL5C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062703_STHDMAE_noMRF_linear_PTBXL44C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062704_STHDMAE_noMRF_finetune_PTBXL44C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062705_STHDMAE_noMRF_linear_PTBXL19C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062706_STHDMAE_noMRF_finetune_PTBXL19C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062707_STHDMAE_noMRF_linear_Chapman4C12L/best_model'
    # # save_filename = 'Models/FineTuneClass/S25062708_STHDMAE_noMRF_finetune_Chapman4C12L/best_model'
    #
    # # save_filename = 'Models/FineTuneClass/S25062709_STHDMAE_noNoise_finetune_PTBXL5C12L/best_model'
    # save_filename = 'Models/FineTuneClass/S25062710_STHDMAE_noNoise_finetune_PTBXL44C12L/best_model'
    #
    # train(model, optimizer, train_loader, val_loader, save_filename, epoch=args.epochs, N_CLASSES=44)

    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062701_STHDMAE_noMRF_linear_PTBXL5C12L/best_model_11_93540_62598_76743.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062702_STHDMAE_noMRF_finetune_PTBXL5C12L/best_model_10_91847_59338_74105.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062703_STHDMAE_noMRF_linear_PTBXL44C12L/best_model_25_97166_52165_64702.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062704_STHDMAE_noMRF_finetune_PTBXL44C12L/best_model_10_96434_50954_61602.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062705_STHDMAE_noMRF_linear_PTBXL19C12L/best_model_21_92363_46392_56044.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062706_STHDMAE_noMRF_finetune_PTBXL19C12L/best_model_6_90562_41509_51231.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062707_STHDMAE_noMRF_linear_Chapman4C12L/best_model_16_99591_95770_96266.pth",
    # #     map_location=DEVICE)
    # # net_weight = torch.load(
    # #     "Models/FineTuneClass/S25062708_STHDMAE_noMRF_finetune_Chapman4C12L/best_model_48_99349_95907_96119.pth",
    # #     map_location=DEVICE)
    #
    net_weight = torch.load(
        "Models/FineTuneClass/S25062710_STHDMAE_noNoise_finetune_PTBXL44C12L/best_model_31_97105_51839_64411.pth",
        map_location=DEVICE)

    model.load_state_dict(net_weight)
    test(model, test_loader, N_CLASSES=44)


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    parser = get_args_parser()
    args = parser.parse_args([])  # 不从命令行读取参数，直接使用默认值

    # 设置默认参数（如有需要，可直接修改这里）
    args.batch_size = 256
    args.epochs = 50
    args.mask_ratio = 0
    args.output_dir = 'Models/FineTuneClass/S25061801_STHDMAE_FixR_PTBXL5C12L'
    args.lr = 1e-3
    args.seed = 233

    # 运行主函数
    main(args)
