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
import M25013001_FCPatchSTA4_FFlt_FWCAres_RandMask_Model
# from M24122501_2to1DMAE_AimMask_Model import MaskedAutoencoderViT
from E25010505_Ori2DMAE_Fusion_RandMask_PretrainEngin import train_one_epoch
from He_utils.misc import NativeScalerWithGradNormCount as NativeScaler
import matplotlib.pyplot as plt


class ECGDataset(Dataset):
    def __init__(self, signals, texts):
        self.signals = signals
        self.target_length = 1000  # 目标长度
        self.texts = texts

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_length = signal.shape[-1]

        # if signal_length < self.target_length:
        #     # 零填充到目标长度
        #     padding = self.target_length - signal_length
        #     signal = torch.cat((torch.tensor(signal, dtype=torch.float32),
        #                         torch.zeros((signal.shape[0], padding), dtype=torch.float32)), dim=-1)
        # else:
        #     signal = torch.tensor(signal[:, :self.target_length], dtype=torch.float32)
        signal = torch.tensor(signal[:, :self.target_length], dtype=torch.float32)

        text = self.texts[idx]
        text = torch.tensor(text, dtype=torch.float32)

        return signal, text


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit', type=str)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./Models/S120601_1DTest01', type=str)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Number of warmup epochs for learning rate schedule')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower learning rate bound for learning rate scheduler')
    return parser


def main(args):
    # 固定随机数种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device)

    # 加载数据
    # train_file_path = "E:/ECG_Data/ESTT_dt0_train_2L.npz"
    # # train_file_path = "E:/ECG_Data/ESTT_dt0_val_2L.npz"
    # val_file_path = "E:/ECG_Data/ESTT_dt0_val_2L.npz"
    # train_file_path = "E:/Paper04_Data/ESTT_dt0_train_26LMM.npz"
    # val_file_path = "E:/Paper04_Data/ESTT_dt0_val_26LMM.npz"
    # train_data = np.load(train_file_path)["pair_sig"]
    # val_data = np.load(val_file_path)["pair_sig"]

    # train_file_path = "E:/Paper04_Data/PTBXL_128_Train_MM1.npy"
    # val_file_path = "E:/Paper04_Data/PTBXL_128_val_MM1.npy"
    # train_data = np.load(train_file_path, allow_pickle=True)
    # val_data = np.load(val_file_path, allow_pickle=True)

    # route1 = './Data/Mimic_Data/big_data.npy'
    route1 = './Data/Mimic_Data/train_data.npy'
    # route1 = './Data/Mimic_Data/val_data.npy'
    X_Train_Ori = np.load(route1, allow_pickle=True)
    X_Train_Ori = np.transpose(X_Train_Ori, (0, 2, 1))
    route2 = './Data/Mimic_Data/val_data.npy'
    X_Val_Ori = np.load(route2, allow_pickle=True)
    X_Val_Ori = np.transpose(X_Val_Ori, (0, 2, 1))

    # route01 = 'F:/Paper03_Data/MIMIC4withLabel/big_embedding.npy'
    route01 = 'F:/Paper03_Data/MIMIC4withLabel/train_embedding.npy'
    # route01 = 'F:/Paper03_Data/MIMIC4withLabel/val_embedding.npy'
    desc_Train_Ori = np.load(route01, allow_pickle=True)
    route02 = 'F:/Paper03_Data/MIMIC4withLabel/val_embedding.npy'
    desc_Val_Ori = np.load(route02, allow_pickle=True)

    train_dataset = ECGDataset(X_Train_Ori, desc_Train_Ori)
    val_dataset = ECGDataset(X_Val_Ori, desc_Val_Ori)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 模型
    # model = M_1DMae_try01.__dict__[args.model](norm_pix_loss=False)
    # model = M_1DMae_try01.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    # model = M24121001_1DMaeHe_2L.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    # model = M24122501_2to1DMAE_AimMask_Model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model = M25013001_FCPatchSTA4_FFlt_FWCAres_RandMask_Model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # checkpoint = torch.load("Models/PreTrainModels/R25061201_STHDMAE_500_BigMimic_Try01/epoch_270.pth", map_location=device)
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # # 如果有调度器：
    # # lr_scheduler.load_state_dict(checkpoint['scheduler'])
    #
    # # 上次训练停在第 checkpoint['epoch'] 轮（0-based），
    # # 所以下次要从下一轮开始：
    # start_epoch = checkpoint['epoch'] + 1

    # 损失缩放器
    loss_scaler = NativeScaler()

    # 模型保存路径
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练过程
    print(f"Start training for {args.epochs} epochs")
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
    # for epoch in range(start_epoch, args.epochs):
        # 训练
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, loss_scaler, log_writer=None, args=args
        )

        # # 验证
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for samples in val_loader:
        #         samples = samples.to(device, non_blocking=True)
        #         with torch.cuda.amp.autocast():
        #             loss, pred_sig, _, _ = model(samples, mask_ratio=args.mask_ratio)
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)
        # 验证阶段，添加信号对比绘制
        model.eval()
        val_loss = 0
        first_batch_visualized = False  # 用于控制只绘制一次
        start_epoch = 0
        final_epoch = 100

        with torch.no_grad():
            for samples, texts in val_loader:
                samples = samples.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    loss, pred_sig, _, _ = model(samples, texts, epoch, mask_ratio=args.mask_ratio)
                val_loss += loss.item()

                # 只绘制第一个 batch 的第一条信号
                if not first_batch_visualized:
                    first_batch_visualized = True

                    # 获取第一个 batch 的第一条信号的第一个通道
                    original_signal = samples[10, 0].cpu().numpy()
                    reconstructed_signal = pred_sig[10, 0].cpu().numpy()

                    # 绘制原始信号和重建信号，限制幅值在 0-1 范围
                    plt.figure(figsize=(10, 5))
                    plt.plot(original_signal, label="Original Signal", linewidth=1)
                    plt.plot(reconstructed_signal, label="Reconstructed Signal", linewidth=1)
                    plt.title("Original vs Reconstructed Signal (First Batch, First Sample, First Channel)")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.grid(True)

                    # # 设置幅值范围为 0-1
                    # plt.ylim(0, 1)

                    plt.show()

        val_loss /= len(val_loader)

        # 保存权重
        save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, save_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_best_{best_val_loss}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, best_save_path)

        # 打印日志
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_stats['loss']:.4f}, Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args([])  # 不从命令行读取参数，直接使用默认值

    # 设置默认参数（如有需要，可直接修改这里）
    args.batch_size = 128
    args.epochs = 100
    args.mask_ratio = 0.50
    args.output_dir = 'Models/PreTrainModels/R25062405_STHDMAE_AS_SmallMimic_Try01'
    args.lr = 1e-4
    args.seed = 42

    # 运行主函数
    main(args)
