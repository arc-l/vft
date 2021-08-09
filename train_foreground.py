import torch
from models import reinforcement_net
from dataset import ForegroundDataset
import argparse
import time
import datetime
import os
from constants import PUSH_Q, GRASP_Q, NUM_ROTATION
from torch.utils.tensorboard import SummaryWriter

import log_utils
import torch_utils


def parse_args():
    default_params = {
        "lr": 1e-6,
        "batch_size": 16,
        "t_0": 5,  # CosineAnnealing, start  1 6 16 36 76
        "t_mult": 2,  # CosineAnnealing, period 5 10 20 40
        "eta_min": 1e-15,  # CosineAnnealing, minimum lr
        "epochs": 36,  # CosineAnnealing, should end before warm start
        "loss_beta": 1,
        "num_rotation": NUM_ROTATION,
    }

    parser = argparse.ArgumentParser(description="Train foreground")

    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        default=default_params["lr"],
        help="Enter the learning rate",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        default=default_params["batch_size"],
        type=int,
        help="Enter the batchsize for training and testing",
    )
    parser.add_argument(
        "--t_0",
        action="store",
        default=default_params["t_0"],
        type=int,
        help="The t_0 of CosineAnnealing",
    )
    parser.add_argument(
        "--t_mult",
        action="store",
        default=default_params["t_mult"],
        type=int,
        help="The t_mult of CosineAnnealing",
    )
    parser.add_argument(
        "--eta_min",
        action="store",
        default=default_params["eta_min"],
        type=float,
        help="The eta_min of CosineAnnealing",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        default=default_params["epochs"],
        type=int,
        help="Enter the epoch for training",
    )
    parser.add_argument(
        "--loss_beta",
        action="store",
        default=default_params["loss_beta"],
        type=int,
        help="The beta of SmoothL1Loss",
    )
    parser.add_argument(
        "--num_rotation",
        action="store",
        default=default_params["num_rotation"],
        type=int,
        help="Number of rotation",
    )
    parser.add_argument("--dataset_root", action="store", help="Enter the path to the dataset")
    parser.add_argument(
        "--pretrained_model", action="store", help="The path to the pretrained model"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Testing and visualizing"
    )

    args = parser.parse_args()

    return args


class ForegroundTrainer:
    def __init__(self, args):
        self.params = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "t_0": args.t_0,  # CosineAnnealing, start  0 4 12 28
            "t_mult": args.t_mult,  # CosineAnnealing, period 4 8 16
            "eta_min": args.eta_min,  # CosineAnnealing, minimum lr
            "epochs": args.epochs,  # CosineAnnealing, should end before warm start
            "loss_beta": args.loss_beta,
            "num_rotation": args.num_rotation,
        }

        self.dataset_root = args.dataset_root
        self.pretrained_model = args.pretrained_model
        self.test = args.test
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if not self.test:
            self.log_dir = os.path.join(self.dataset_root, "runs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            timestamp_value = datetime.datetime.fromtimestamp(time.time())
            time_name = timestamp_value.strftime("%Y-%m-%d-%H-%M")
            self.log_dir = os.path.join(self.log_dir, time_name)
            self.tb_logger = SummaryWriter(self.log_dir)
            self.logger = log_utils.setup_logger(self.log_dir, "Foreground")

    def main(self):
        model = reinforcement_net(True)
        model = model.to(self.device)
        criterion_push = torch.nn.SmoothL1Loss(beta=self.params["loss_beta"], reduction="none")
        criterion_grasp = torch.nn.SmoothL1Loss(beta=self.params["loss_beta"], reduction="none")
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.9,
            weight_decay=2e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.params["t_0"],
            T_mult=self.params["t_mult"],
            eta_min=self.params["eta_min"],
            last_epoch=-1,
            verbose=False,
        )
        start_epoch = 0

        if self.pretrained_model is not None:
            checkpoint = torch.load(self.pretrained_model)
            model.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            # prev_params = checkpoint["params"]

        if self.test:
            data_loader = self._get_data_loader("test", 1, shuffle=False, test=True)
            criterion = torch.nn.SmoothL1Loss(reduction="none")
            self._test(model, data_loader)
        else:
            self.logger.info(f"Hyperparameters: {self.params}")
            if self.pretrained_model is not None:
                self.logger.info(f"Start from the pretrained model: {self.pretrained_model}")
                # self.logger.info(f"Previous Hyperparameters: {prev_params}")

            data_loader_train = self._get_data_loader(
                "train", self.params["batch_size"], shuffle=True
            )
            data_loader_test = self._get_data_loader("test", max(1, self.params["batch_size"] // 2))

            for epoch in range(start_epoch, self.params["epochs"]):
                # warmup start
                if epoch == 0:
                    warmup_factor = 0.001
                    warmup_iters = min(1000, len(data_loader_train) - 1)
                    current_lr_scheduler = torch_utils.warmup_lr_scheduler(
                        optimizer, warmup_iters, warmup_factor
                    )
                else:
                    current_lr_scheduler = lr_scheduler

                train_loss = self._train_one_epoch(
                    model,
                    criterion_push,
                    criterion_grasp,
                    optimizer,
                    data_loader_train,
                    current_lr_scheduler,
                    epoch,
                )
                evaluate_loss = self._evaluate(
                    model, criterion_push, criterion_grasp, data_loader_test
                )

                save_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "params": self.params,
                }
                torch.save(save_state, os.path.join(self.log_dir, f"foreground_model-{epoch}.pth"))

                self.tb_logger.add_scalars(
                    "Epoch_Loss", {"train": train_loss, "test": evaluate_loss}, epoch
                )
                self.tb_logger.flush()

            self.tb_logger.add_hparams(
                self.params, {"hparam/train": train_loss, "hparam/test": evaluate_loss}
            )
            self.logger.info("Training completed!")

    def _train_one_epoch(
        self,
        model,
        criterion_push,
        criterion_grasp,
        optimizer,
        data_loader,
        lr_scheduler,
        epoch,
        print_freq=50,
    ):
        model.train()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", log_utils.SmoothedValue(window_size=1, fmt="{value:.12f}"))
        metric_logger.add_meter("loss", log_utils.SmoothedValue())
        metric_logger.add_meter("grasp_loss", log_utils.SmoothedValue())
        metric_logger.add_meter("push_loss", log_utils.SmoothedValue())
        header = "Epoch: [{}]".format(epoch)
        losses = []
        n_iter = 0
        total_iters = len(data_loader)

        for (color_images, depth_images, push_targets, grasp_targets) in metric_logger.log_every(
            data_loader, print_freq, self.logger, header
        ):
            color_images = color_images.to(self.device, non_blocking=True)
            depth_images = depth_images.to(self.device, non_blocking=True)
            push_targets = push_targets.to(self.device, non_blocking=True)
            grasp_targets = grasp_targets.to(self.device, non_blocking=True)

            output = model(color_images, depth_images, use_push=False)

            weights_push = torch.ones(push_targets.shape)
            weights_grasp = torch.ones(grasp_targets.shape)
            weights_push[push_targets > 0] = 2
            weights_grasp[grasp_targets > 0] = 2

            loss_push = criterion_push(output[0], push_targets) * weights_push.cuda()
            loss_push = loss_push.sum() / push_targets.size(0)
            loss_grasp = criterion_grasp(output[1], grasp_targets) * weights_grasp.cuda()
            loss_grasp = loss_grasp.sum() / grasp_targets.size(0)

            optimizer.zero_grad()
            if epoch != 0:
                loss_push.backward()
            loss_grasp.backward()
            loss = loss_push + loss_grasp
            optimizer.step()

            # log
            log_loss = loss.item()
            log_loss_push = loss_push.item()
            log_loss_grasp = loss_grasp.item()
            log_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(
                loss=log_loss, lr=log_lr, grasp_loss=log_loss_grasp, push_loss=log_loss_push
            )
            self.tb_logger.add_scalar("Step/Loss/Train", log_loss, total_iters * epoch + n_iter)
            self.tb_logger.add_scalar(
                "Step/Loss/Train/Push", log_loss, total_iters * epoch + n_iter
            )
            self.tb_logger.add_scalar(
                "Step/Loss/Train/Grasp", log_loss, total_iters * epoch + n_iter
            )
            self.tb_logger.add_scalar("Step/LR", log_lr, total_iters * epoch + n_iter)
            losses.append(log_loss)

            if epoch == 0:
                lr_scheduler.step()
            n_iter += 1
        if epoch != 0:
            lr_scheduler.step(epoch)

        return sum(losses) / len(losses)

    @torch.no_grad()
    def _evaluate(self, model, criterion_push, criterion_grasp, data_loader, print_freq=10):
        model.eval()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", log_utils.SmoothedValue(window_size=len(data_loader)))
        metric_logger.add_meter("grasp_loss", log_utils.SmoothedValue())
        metric_logger.add_meter("push_loss", log_utils.SmoothedValue())
        losses = []
        header = "Test:"

        for (color_images, depth_images, push_targets, grasp_targets) in metric_logger.log_every(
            data_loader, print_freq, self.logger, header
        ):
            color_images = color_images.to(self.device, non_blocking=True)
            depth_images = depth_images.to(self.device, non_blocking=True)
            push_targets = push_targets.to(self.device, non_blocking=True)
            grasp_targets = grasp_targets.to(self.device, non_blocking=True)

            output = model(color_images, depth_images, use_push=False)

            weights_push = torch.ones(push_targets.shape)
            weights_grasp = torch.ones(grasp_targets.shape)
            weights_push[push_targets > 0] = 2
            weights_grasp[grasp_targets > 0] = 2

            loss_push = criterion_push(output[0], push_targets) * weights_push.cuda()
            loss_push = loss_push.sum() / push_targets.size(0)
            loss_grasp = criterion_grasp(output[1], grasp_targets) * weights_grasp.cuda()
            loss_grasp = loss_grasp.sum() / grasp_targets.size(0)
            loss = loss_push + loss_grasp

            log_loss = loss.item()
            log_loss_push = loss_push.item()
            log_loss_grasp = loss_grasp.item()
            metric_logger.update(loss=log_loss, grasp_loss=log_loss_grasp, push_loss=log_loss_push)
            losses.append(log_loss)

        return sum(losses) / len(losses)

    def _get_data_loader(self, folder, batch_size, shuffle=False, test=False):
        """Get data loader."""
        path = os.path.join(self.dataset_root, folder)
        dataset = ForegroundDataset(path, self.params["num_rotation"])
        if not test:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=False
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False
            )

        return data_loader

    @torch.no_grad()
    def _test(self, model, data_loader):
        import torchvision
        import matplotlib.pyplot as plt
        from PIL import Image, ImageStat

        torch.manual_seed(1)

        model.eval()

        ite = iter(data_loader)
        for _ in range(5):
            color_img_pil, depth_img_pil, push_target_img_pil, grasp_target_img_pil = next(ite)
        color_img_pil_train = color_img_pil.to(self.device)
        depth_img_pil_train = depth_img_pil.to(self.device)

        outputs = model(color_img_pil_train, depth_img_pil_train)
        push = outputs[0][0].cpu()
        grasp = outputs[1][0].cpu()
        push *= 1 / PUSH_Q
        push[push > 1] = 1
        push[push < 0] = 0
        grasp *= 1 / GRASP_Q
        grasp[grasp > 1] = 1
        grasp[grasp < 0] = 0

        new_push = push.clone()
        new_grasp = grasp.clone()
        new_push[new_push > 0.5] = 1
        new_push[new_push <= 0.5] = 0
        new_grasp[new_grasp > 0.5] = 1
        new_grasp[new_grasp <= 0.5] = 0

        to_pil = torchvision.transforms.ToPILImage()
        img1 = to_pil(color_img_pil[0])
        img2 = to_pil(depth_img_pil[0])
        img3 = to_pil(push_target_img_pil[0])
        img4 = to_pil(grasp_target_img_pil[0])
        img5 = to_pil(push)
        img6 = to_pil(grasp)
        img7 = to_pil(new_push)
        img8 = to_pil(new_grasp)

        titles = [
            "Color",
            "Depth",
            "Target_push",
            "Target_grasp",
            "predicted push",
            "predicted grasp",
            "binary predicted push",
            "binary predicted grasp",
        ]
        images = [img1, img2, img3, img4, img5, img6, img7, img8]

        for i in range(len(images)):
            plt.subplot(2, 4, i + 1), plt.imshow(images[i], "gray")
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    trainer = ForegroundTrainer(args)
    trainer.main()


# def get_data_loader(dataset_root, batch_size):
#     # use our dataset and defined transformations
#     dataset = ForegroundDataset(dataset_root, 16)
#     # dataset_test = ForegroundDataset(dataset_root, 16)

#     # split the dataset in train and test set
#     indices = torch.randperm(len(dataset)).tolist()
#     start_point = 5
#     dataset = torch.utils.data.Subset(dataset, indices[start_point:])
#     dataset_test = torch.utils.data.Subset(dataset, indices[:start_point])

#     # define training and validation data loaders
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
#     )
#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=batch_size, shuffle=False, num_workers=1
#     )

#     return data_loader, data_loader_test


# def train_one_epoch(
#     model,
#     criterion_push,
#     criterion_grasp,
#     optimizer,
#     data_loader,
#     device,
#     epoch,
#     print_freq,
#     resume=False,
# ):
#     """
#     https://github.com/pytorch/vision/blob/master/references/detection/engine.py
#     """
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
#     header = "Epoch: [{}]".format(epoch)

#     lr_scheduler = None
#     if epoch == 0 and not resume:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

#     for color_images, depth_images, push_targets, grasp_targets in metric_logger.log_every(
#         data_loader, print_freq, header
#     ):
#         color_images = color_images.to(device)
#         depth_images = depth_images.to(device)
#         push_targets = push_targets.to(device)
#         grasp_targets = grasp_targets.to(device)

#         optimizer.zero_grad()

#         output_probs = model(color_images, depth_images)

#         weights = torch.ones(grasp_targets.shape)
#         # if it doesn't converge, just restart, expecting the loss to below 60. it should below 100 very soon
#         weights[grasp_targets > 0] = 2

#         loss1 = criterion_push(output_probs[0], push_targets)
#         loss1 = loss1.sum() / push_targets.size(0)
#         loss1.backward()
#         loss2 = criterion_grasp(output_probs[1], grasp_targets) * weights.cuda()
#         loss2 = loss2.sum() / grasp_targets.size(0)
#         loss2.backward()
#         losses = loss1 + loss2

#         optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses.cpu())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger


# def main(args):
#     data_loader, data_loader_test = get_data_loader(
#         args.dataset_root, args.batch_size, args.fine_tuning_num
#     )

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     model = reinforcement_net(True)  # TODO: remove use_cuda in model, replace with device
#     if args.resume:
#         # model.load_state_dict(torch.load('data/pre_train/foreground_model.pth'))
#         model.load_state_dict(torch.load(os.path.join(args.dataset_root, "foreground_model.pth")))

#     criterion_push = torch.nn.SmoothL1Loss(reduction="none")
#     criterion_grasp = torch.nn.SmoothL1Loss(reduction="none")
#     # criterion_push = torch.nn.BCEWithLogitsLoss()
#     # criterion_grasp = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))

#     # construct an optimizer
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=2e-5)
#     # optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=2e-5)

#     # and a learning rate scheduler which decreases the learning rate by 10x every 1 epochs
#     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
#     # for large dataset
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.5)
#     # for small dataset, expect ~ 50 epochs
#     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#     for epoch in range(args.epochs):
#         # train for one epoch, printing every 10 iterations
#         train_one_epoch(
#             model,
#             criterion_push,
#             criterion_grasp,
#             optimizer,
#             data_loader,
#             device,
#             epoch,
#             print_freq=20,
#             resume=args.resume,
#         )
#         # update the learning rate
#         lr_scheduler.step()
#         # evaluate on the test dataset
#         # evaluate(model, criterion, data_loader_test, device=device)

#         torch.save(model.state_dict(), os.path.join(args.dataset_root, "foreground_model.pth"))


# @torch.no_grad()
# def test():
#     import torchvision
#     import matplotlib.pyplot as plt
#     from PIL import Image, ImageStat

#     torch.manual_seed(2)

#     # data_loader, data_loader_test = get_data_loader('data/pre_train/', 1)
#     data_loader, data_loader_test = get_data_loader("logs/real-maskrcnn/data", 1)
#     # data_loader, data_loader_test = get_data_loader('logs/final-pretrain/data', 1)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = reinforcement_net(True)
#     # model.load_state_dict(torch.load('data/pre_train/foreground_model.pth'))
#     model.load_state_dict(torch.load("logs/random-pretrain/data/foreground_model.pth"))
#     # model.load_state_dict(torch.load('logs/real-maskrcnn/data/foreground_model.pth'))
#     # model.load_state_dict(torch.load('logs_push/final/data/foreground_model.pth'))
#     model.eval().to(device)
#     sig = torch.nn.Sigmoid()

#     ite = iter(data_loader)
#     for _ in range(6):
#         color_img_pil, depth_img_pil, push_target_img_pil, grasp_target_img_pil = next(ite)
#     color_img_pil_train = color_img_pil.to(device)
#     depth_img_pil_train = depth_img_pil.to(device)

#     outputs = model(color_img_pil_train, depth_img_pil_train)
#     # push = sig(outputs[0][0]).cpu()
#     # grasp = sig(outputs[1][0]).cpu()
#     push = outputs[0][0].cpu()
#     grasp = outputs[1][0].cpu()
#     push *= 1 / PUSH_Q
#     push[push > 1] = 1
#     push[push < 0] = 0
#     grasp *= 1 / GRASP_Q
#     grasp[grasp > 1] = 1
#     grasp[grasp < 0] = 0

#     new_push = push.clone()
#     new_grasp = grasp.clone()
#     new_push[new_push > 0.5] = 1
#     new_push[new_push <= 0.5] = 0
#     new_grasp[new_grasp > 0.5] = 1
#     new_grasp[new_grasp <= 0.5] = 0

#     to_pil = torchvision.transforms.ToPILImage()
#     img1 = to_pil(color_img_pil[0])
#     img2 = to_pil(depth_img_pil[0])
#     img3 = to_pil(push_target_img_pil[0])
#     img4 = to_pil(grasp_target_img_pil[0])
#     img5 = to_pil(push)
#     img6 = to_pil(grasp)
#     img7 = to_pil(new_push)
#     img8 = to_pil(new_grasp)

#     titles = [
#         "Color",
#         "Depth",
#         "Target_push",
#         "Target_grasp",
#         "predicted push",
#         "predicted grasp",
#         "binary predicted push",
#         "binary predicted grasp",
#     ]
#     images = [img1, img2, img3, img4, img5, img6, img7, img8]

#     for i in range(len(images)):
#         plt.subplot(2, 4, i + 1), plt.imshow(images[i], "gray")
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
#     plt.show()
#     # plt.savefig('test_pre.png')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train foreground")

#     parser.add_argument(
#         "--dataset_root", dest="dataset_root", action="store", help="Enter the path to the dataset"
#     )
#     parser.add_argument(
#         "--epochs",
#         dest="epochs",
#         action="store",
#         type=int,
#         default=30,
#         help="Enter the epoch for training",
#     )
#     parser.add_argument(
#         "--batch_size",
#         dest="batch_size",
#         action="store",
#         type=int,
#         default=16,
#         help="Enter the batchsize for training and testing",
#     )
#     parser.add_argument(
#         "--test", dest="test", action="store_true", default=False, help="Testing and visualizing"
#     )
#     parser.add_argument(
#         "--lr", dest="lr", action="store", type=float, default=1e-6, help="Enter the learning rate"
#     )
#     parser.add_argument(
#         "--real_fine_tuning", dest="real_fine_tuning", action="store_true", default=False, help=""
#     )
#     parser.add_argument(
#         "--fine_tuning_num",
#         dest="fine_tuning_num",
#         action="store",
#         type=int,
#         default=16500,
#         help="1500 action, one action contains 11 images",
#     )
#     parser.add_argument(
#         "--resume",
#         dest="resume",
#         action="store_true",
#         default=False,
#         help="Enter the path to the dataset",
#     )

#     args = parser.parse_args()

#     if args.resume:
#         args.epochs = 10
#     else:
#         args.fine_tuning_num = None

#     if args.test:
#         test()
#     else:
#         main(args)