import torch
from torchvision import transforms as T
from push_net import PushPredictionNet
from dataset import PushPredictionMultiDataset, ClusterRandomSampler
import argparse
import time
import datetime
import os
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from constants import (
    PUSH_DISTANCE,
    COLOR_MEAN,
    COLOR_STD,
)


import torch_utils
import log_utils


def parse_args():
    default_params = {
        "lr": 1e-3,
        "batch_size": 64,
        "t_0": 5,  # CosineAnnealing, start  1 6 16 36 76
        "t_mult": 2,  # CosineAnnealing, period 5 10 20 40
        "eta_min": 1e-8,  # CosineAnnealing, minimum lr
        "epochs": 76,  # CosineAnnealing, should end before warm start
        "loss_beta": 2,
        "distance": PUSH_DISTANCE,
    }

    parser = argparse.ArgumentParser(description="Train Push Prediction")

    parser.add_argument(
        "--lr", action="store", default=default_params["lr"], type=float, help="The learning rate"
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        default=default_params["batch_size"],
        type=int,
        help="The batch size for training and testing",
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
        help="The epoch for training, should end before warm start of CosineAnnealing",
    )
    parser.add_argument(
        "--loss_beta",
        action="store",
        default=default_params["loss_beta"],
        type=int,
        help="The beta of SmoothL1Loss",
    )
    parser.add_argument(
        "--distance",
        action="store",
        default=default_params["distance"],
        type=float,
        help="The distance of one push",
    )
    parser.add_argument(
        "--dataset_root", action="store", required=True, help="The path to the dataset"
    )
    parser.add_argument(
        "--pretrained_model", action="store", help="The path to the pretrained model"
    )
    parser.add_argument(
        "--len_dataset",
        action="store",
        default=-1,
        type=int,
        help="The number of push dataset should be used",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Testing and visualizing"
    )
    parser.add_argument("--verify", action="store_true", default=False, help="Verify the dataset")
    parser.add_argument(
        "--test_plot",
        action="store_true",
        default=False,
        help="Testing with colorful visualization",
    )
    parser.add_argument(
        "--symmetric_diff", action="store_true", default=False, help="Compute symmetric_diff"
    )

    args = parser.parse_args()
    if (args.test or args.test_plot or args.symmetric_diff) and args.pretrained_model is None:
        parser.error("--test, --test_plot, and --symmetric_diff require --pretrained_model.")

    return args


class PushPredictionTrainer:
    def __init__(self, args):
        self.params = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "t_0": args.t_0,  # CosineAnnealing, start  0 4 12 28
            "t_mult": args.t_mult,  # CosineAnnealing, period 4 8 16
            "eta_min": args.eta_min,  # CosineAnnealing, minimum lr
            "epochs": args.epochs,  # CosineAnnealing, should end before warm start
            "loss_beta": args.loss_beta,
            "distance": args.distance,
        }

        self.dataset_root = args.dataset_root
        self.pretrained_model = args.pretrained_model
        self.len_dataset = args.len_dataset
        self.test = args.test
        self.verify = args.verify
        self.test_plot = args.test_plot
        self.symmetric_diff = args.symmetric_diff
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if not (self.test or self.verify or self.test_plot or self.symmetric_diff):
            self.log_dir = os.path.join(self.dataset_root, "runs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            timestamp_value = datetime.datetime.fromtimestamp(time.time())
            time_name = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
            self.log_dir = os.path.join(self.log_dir, time_name)
            self.tb_logger = SummaryWriter(self.log_dir)
            self.logger = log_utils.setup_logger(self.log_dir, "Push Prediction")

    def main(self):
        model = PushPredictionNet()
        model = model.to(self.device)
        criterion = torch.nn.SmoothL1Loss(beta=self.params["loss_beta"])
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4,
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
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            # prev_params = checkpoint["params"]

        if self.test:
            data_loader = self._get_data_loader("test", 1, shuffle=True, test=True)
            criterion = torch.nn.MSELoss(reduction="none")
            self._test(model, criterion, data_loader)
        elif self.verify:
            data_loader = self._get_data_loader("test", 1, test=True)
            self._verify_dataset(model, data_loader)
        elif self.test_plot:
            data_loader = self._get_data_loader("test", 1, shuffle=True, test=True)
            criterion = torch.nn.SmoothL1Loss(reduction="none")
            self._test_plot(model, criterion, data_loader)
        elif self.symmetric_diff:
            data_loader = self._get_data_loader("test", 1, test=True)
            criterion = torch.nn.MSELoss(reduction="none")
            self._symmetric_diff(model, criterion, data_loader)
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
                    model, criterion, optimizer, data_loader_train, current_lr_scheduler, epoch
                )
                evaluate_loss = self._evaluate(model, criterion, data_loader_test)

                save_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "params": self.params,
                }
                torch.save(
                    save_state, os.path.join(self.log_dir, f"push_prediction_model-{epoch}.pth")
                )

                self.tb_logger.add_scalars(
                    "Epoch_Loss", {"train": train_loss, "test": evaluate_loss}, epoch
                )
                self.tb_logger.flush()

            self.tb_logger.add_hparams(
                self.params, {"hparam/train": train_loss, "hparam/test": evaluate_loss}
            )
            self.logger.info("Training completed!")

    def _train_one_epoch(
        self, model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq=100
    ):
        model.train()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", log_utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", log_utils.SmoothedValue())
        header = "Epoch: [{}]".format(epoch)
        losses = []
        n_iter = 0
        total_iters = len(data_loader)

        for (
            prev_color_img,
            _,
            _,
            _,
            used_binary_img,
            prev_poses,
            _,
            action,
            delta,
            _,
            _,
            _,
            _,
            binary_objs_total,
            num_obj,
        ) in metric_logger.log_every(data_loader, print_freq, self.logger, header):
            used_binary_img_gpu = used_binary_img.to(
                self.device, non_blocking=True, dtype=torch.float
            )
            prev_poses_gpu = prev_poses.to(self.device, non_blocking=True)
            action_gpu = action.to(self.device, non_blocking=True)
            binary_objs_total_gpu = binary_objs_total.to(self.device, non_blocking=True)
            target_gpu = delta.to(self.device, non_blocking=True)

            # forward
            output = model(
                prev_poses_gpu, action_gpu, used_binary_img_gpu, binary_objs_total_gpu, num_obj[0]
            )
            # get loss
            loss = criterion(output, target_gpu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            log_loss = loss.item()
            log_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=log_loss, lr=log_lr)
            self.tb_logger.add_scalar("Step/Loss/Train", log_loss, total_iters * epoch + n_iter)
            self.tb_logger.add_scalar("Step/LR", log_lr, total_iters * epoch + n_iter)
            losses.append(log_loss)

            if epoch == 0:
                lr_scheduler.step()
            n_iter += 1
        if epoch != 0:
            lr_scheduler.step(epoch)

        return sum(losses) / len(losses)

    @torch.no_grad()
    def _evaluate(self, model, criterion, data_loader, print_freq=20):
        model.eval()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", log_utils.SmoothedValue(window_size=len(data_loader)))
        losses = []
        header = "Test:"

        for (
            _,
            _,
            _,
            _,
            used_binary_img,
            prev_poses,
            _,
            action,
            delta,
            _,
            _,
            _,
            _,
            binary_objs_total,
            num_obj,
        ) in metric_logger.log_every(data_loader, print_freq, self.logger, header):
            used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)
            target = delta.to(self.device, non_blocking=True)

            output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
            loss = criterion(output, target)

            log_loss = loss.item()
            metric_logger.update(loss=log_loss)
            losses.append(log_loss)

        return sum(losses) / len(losses)

    def _get_data_loader(self, folder, batch_size, len_dataset=None, shuffle=False, test=False):
        """Get data loader, group data with the same number of objects.

        With ClusterRandomSamplerThe shuffle should be False, drop_last is not used, so it can be False.
        """
        path = os.path.join(self.dataset_root, folder)
        dataset = PushPredictionMultiDataset(path, self.params["distance"], False, len_dataset)
        if not test:
            sampler = ClusterRandomSampler(dataset, batch_size, shuffle)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False
            )

        return data_loader

    # @torch.no_grad()
    # def _compute_overlap(self, prev_poses, outputs, prev_color_imgs, mask_objs, num_obj):
    #     batch_size = prev_poses.size(0)
    #     overlaps = np.zeros(batch_size)
    #     prev_poses = prev_poses.numpy()
    #     outputs = outputs.numpy()
    #     num_obj = num_obj.item()
    #     mask_objs = mask_objs.numpy()
    #     inv_normalize = T.Normalize(
    #         mean=[
    #             -COLOR_MEAN[0] / COLOR_STD[0],
    #             -COLOR_MEAN[1] / COLOR_STD[1],
    #             -COLOR_MEAN[2] / COLOR_STD[2],
    #         ],
    #         std=[1 / COLOR_STD[0], 1 / COLOR_STD[1], 1 / COLOR_STD[2]],
    #     )

    #     for batch_idx in range(batch_size):
    #         ori_image = inv_normalize(prev_color_imgs[batch_idx])
    #         ori_image = ori_image.permute(1, 2, 0).numpy()
    #         ori_image[ori_image < 0] = 0
    #         ori_image *= 255
    #         ori_image = ori_image.astype(np.uint8)
    #         ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2GRAY)
    #         ori_image = cv2.threshold(ori_image, 50, 255, cv2.THRESH_BINARY)[1]
    #         new_image = np.zeros(ori_image.shape)
    #         for obj_idx in range(num_obj):
    #             mask = mask_objs[batch_idx][obj_idx]
    #             points = np.argwhere(mask == 255)
    #             points = np.expand_dims(points, axis=0)
    #             prev_pose = prev_poses[batch_idx][obj_idx * 2 : obj_idx * 2 + 2]
    #             output = outputs[batch_idx][obj_idx * 3 : obj_idx * 3 + 3]
    #             M = cv2.getRotationMatrix2D((prev_pose[0], prev_pose[1]), -output[2], 1)
    #             M[0, 2] += output[0]
    #             M[1, 2] += output[1]
    #             new_points = cv2.transform(points, M)[0]
    #             valid_points = np.logical_and(
    #                 np.logical_and(new_points[:, 0] <= 223, new_points[:, 0] >= 0),
    #                 np.logical_and(new_points[:, 1] <= 223, new_points[:, 1] >= 0),
    #             )
    #             new_points = tuple(np.transpose(new_points[valid_points]))
    #             new_image[new_points] = 255
    #         ori_area = np.sum(ori_image == 255)
    #         new_area = np.sum(new_image == 255)
    #         overlaps[batch_idx] = ori_area - new_area
    #         if overlaps[batch_idx] < 0:
    #             cv2.imshow("new", new_image)
    #             cv2.imshow("ori", ori_image)
    #             cv2.waitKey()
    #             cv2.destroyAllWindows()
    #         assert overlaps[batch_idx] >= 0, (ori_area, new_area)

    #     norm = np.linalg.norm(overlaps, ord=1)
    #     if norm == 0:
    #         overlaps = np.ones(batch_size)
    #     else:
    #         overlaps = overlaps / norm * batch_size
    #     overlaps += 1
    #     norm = np.linalg.norm(overlaps, ord=1)
    #     overlaps = overlaps / norm * batch_size

    #     overlaps = overlaps.reshape((batch_size, 1))
    #     overlaps = np.tile(overlaps, 3 * num_obj)

    #     return overlaps

    @torch.no_grad()
    def _test(self, model, criterion, data_loader):
        import matplotlib.pyplot as plt
        import math
        from constants import colors_upper, colors_lower

        torch.manual_seed(1)

        inv_normalize = T.Normalize(
            mean=[
                -COLOR_MEAN[0] / COLOR_STD[0],
                -COLOR_MEAN[1] / COLOR_STD[1],
                -COLOR_MEAN[2] / COLOR_STD[2],
            ],
            std=[1 / COLOR_STD[0], 1 / COLOR_STD[1], 1 / COLOR_STD[2]],
        )

        model.eval()

        images = []
        refs = []
        for i, data in enumerate(data_loader):
            (
                prev_color_img,
                prev_depth_img,
                next_color_img,
                next_depth_img,
                used_binary_img,
                prev_poses,
                next_poses,
                action,
                delta,
                prev_ref,
                next_ref,
                action_start_ori,
                action_end_ori,
                binary_objs_total,
                num_obj,
            ) = data

            prev_color_img = prev_color_img.to(self.device, non_blocking=True)
            prev_depth_img = prev_depth_img.to(self.device, non_blocking=True)
            next_color_img = next_color_img.to(self.device, non_blocking=True)
            next_depth_img = next_depth_img.to(self.device, non_blocking=True)
            used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(self.device, non_blocking=True)
            next_poses = next_poses.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            delta = delta.to(self.device, non_blocking=True)
            action_start_ori = action_start_ori.to(self.device, non_blocking=True)
            action_end_ori = action_end_ori.to(self.device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)

            output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
            target = delta
            loss = criterion(output, target)
            output = output[0].cpu().numpy()
            target = target[0].cpu().numpy()
            # output = target
            output_xy = []
            output_a = []
            for num_idx in range(num_obj):
                output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
                output_a.append(output[num_idx * 3 + 2])
                # no move
                # output_xy.append([0, 0])
                # output_a.append(0)
                # move in the direction as the action
                # output_xy.append([args.distance / 0.2, 0])
                # output_a.append(0)
            print(i)
            print(prev_ref[0])
            print(next_ref[0])
            np.set_printoptions(precision=3, suppress=True)
            print("output", output)
            print("target", target)
            print("action", action_start_ori.cpu().numpy())
            print("loss", loss.cpu().numpy())
            loss = loss.cpu().numpy()[0]

            next_color_img = inv_normalize(next_color_img[0])
            next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
            imdepth = next_color_img
            imdepth[imdepth < 0] = 0
            imdepth[imdepth > 0] = 255
            imdepth = imdepth.astype(np.uint8)
            imdepth = cv2.cvtColor(imdepth, cv2.COLOR_RGB2BGR)

            prev_color_img = inv_normalize(prev_color_img[0])
            prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
            imgcolor = prev_color_img
            imgcolor[imgcolor < 0] = 0
            imgcolor *= 255
            imgcolor = imgcolor.astype(np.uint8)
            imgcolor = cv2.GaussianBlur(imgcolor, (5, 5), 0)
            imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

            prev_poses = prev_poses[0].cpu().numpy()
            next_poses = next_poses[0].cpu().numpy()
            action_start_ori = action_start_ori[0].cpu().numpy()
            action_end_ori = action_end_ori[0].cpu().numpy()
            action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
            action = action[0].cpu().numpy()
            action_start_tile = np.tile(action[:2], num_obj[0])
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            next_poses += action_start_ori_tile
            next_poses -= action_start_tile
            print("prev poses", prev_poses)
            print("next poses", next_poses)

            for ci in range(num_obj):
                color = cv2.inRange(imgcolor, colors_lower[ci], colors_upper[ci])
                contours, _ = cv2.findContours(color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                found = False
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        contours = contour
                        found = True
                        break
                if not found:
                    continue
                cv2.drawContours(
                    imdepth,
                    [contours],
                    -1,
                    (255 / int(num_obj) * (ci + 1), 255, 255 - 255 / int(num_obj) * (ci + 1)),
                    1,
                )
                cnt_rotated = rotate_contour(contours, -output_a[ci])
                cnt_rotated_translated = cnt_rotated + [output_xy[ci][1], output_xy[ci][0]]
                cnt_rotated_translated = np.rint(cnt_rotated_translated).astype(np.int32)
                cv2.drawContours(
                    imdepth,
                    [cnt_rotated_translated],
                    -1,
                    (255 / int(num_obj) * (ci + 1), 150, 255 - 255 / int(num_obj) * (ci + 1)),
                    2,
                )

            for pi in range(num_obj):
                cv2.circle(
                    imdepth,
                    (int(round(prev_poses[pi * 2 + 1])), int(round(prev_poses[pi * 2]))),
                    2,
                    (255, 0, 255),
                    -1,
                )
                cv2.circle(
                    imdepth,
                    (int(round(next_poses[pi * 2 + 1])), int(round(next_poses[pi * 2]))),
                    2,
                    (255, 255, 0),
                    -1,
                )

            # action
            cv2.circle(
                imdepth,
                (int(round(action_start_ori[1])), int(round(action_start_ori[0]))),
                5,
                (255, 0, 0),
                -1,
            )
            cv2.circle(
                imdepth,
                (int(round(action_end_ori[1])), int(round(action_end_ori[0]))),
                5,
                (0, 0, 255),
                -1,
            )

            # if math.sqrt(loss[0]) > 5 or math.sqrt(loss[1]) > 5 or math.sqrt(loss[3]) > 5 or math.sqrt(loss[4]) > 5 or math.sqrt(loss[2]) > 5 or math.sqrt(loss[5]) > 5:
            images.append(cv2.cvtColor(imdepth, cv2.COLOR_BGR2RGB))
            refs.append(prev_ref[0])
            if len(images) == 28:
                for i in range(len(images)):
                    plt.subplot(math.ceil(len(images) / 7), 7, i + 1), plt.imshow(images[i], "gray")
                    plt.title(refs[i][:7])
                    plt.xticks([]), plt.yticks([])
                plt.show()
                # plt.savefig("test.png", dpi=600)
                input_str = input("One more?")
                if input_str == "y":
                    images = []
                    refs = []
                else:
                    break

    @torch.no_grad()
    def _test_plot(self, model, criterion, data_loader):
        import torchvision
        import matplotlib.pyplot as plt
        from PIL import Image, ImageStat
        import math
        from constants import colors_upper, colors_lower

        torch.manual_seed(1)

        inv_normalize = T.Normalize(
            mean=[
                -COLOR_MEAN[0] / COLOR_STD[0],
                -COLOR_MEAN[1] / COLOR_STD[1],
                -COLOR_MEAN[2] / COLOR_STD[2],
            ],
            std=[1 / COLOR_STD[0], 1 / COLOR_STD[1], 1 / COLOR_STD[2]],
        )

        model.eval()

        images = []
        refs = []
        for i, data in enumerate(data_loader):
            (
                prev_color_img,
                prev_depth_img,
                next_color_img,
                next_depth_img,
                used_binary_img,
                prev_poses,
                next_poses,
                action,
                delta,
                prev_ref,
                next_ref,
                action_start_ori,
                action_end_ori,
                binary_objs_total,
                num_obj,
            ) = data

            prev_color_img = prev_color_img.to(self.device, non_blocking=True)
            prev_depth_img = prev_depth_img.to(self.device, non_blocking=True)
            next_color_img = next_color_img.to(self.device, non_blocking=True)
            next_depth_img = next_depth_img.to(self.device, non_blocking=True)
            used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(self.device, non_blocking=True)
            next_poses = next_poses.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            delta = delta.to(self.device, non_blocking=True)
            action_start_ori = action_start_ori.to(self.device, non_blocking=True)
            action_end_ori = action_end_ori.to(self.device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)

            output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
            target = delta
            loss = criterion(output, target)
            output = output[0].cpu().numpy()
            target = target[0].cpu().numpy()
            output_xy = []
            output_a = []
            for num_idx in range(num_obj):
                output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
                output_a.append(output[num_idx * 3 + 2])
            print(i)
            print(prev_ref[0])
            print(next_ref[0])
            np.set_printoptions(precision=3, suppress=True)
            print("output", output)
            print("target", target)
            print("action", action_start_ori.cpu().numpy())
            print("loss", loss.cpu().numpy())
            loss = loss.cpu().numpy()[0]

            # background
            next_color_img = inv_normalize(next_color_img[0])
            next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
            imnext = next_color_img
            imnext[imnext < 0] = 0
            imnext *= 255
            imnext = imnext.astype(np.uint8)
            imnext = cv2.cvtColor(imnext, cv2.COLOR_RGB2BGR)

            prev_color_img = inv_normalize(prev_color_img[0])
            prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
            imgcolor = prev_color_img
            imgcolor[imgcolor < 0] = 0
            imgcolor *= 255
            imgcolor = imgcolor.astype(np.uint8)
            imgcolor = cv2.GaussianBlur(imgcolor, (5, 5), 0)
            imgcolorhsv = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

            prev_poses = prev_poses[0].cpu().numpy()
            next_poses = next_poses[0].cpu().numpy()
            action_start_ori = action_start_ori[0].cpu().numpy()
            action_end_ori = action_end_ori[0].cpu().numpy()
            action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
            action = action[0].cpu().numpy()
            action_start_tile = np.tile(action[:2], num_obj[0])
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            next_poses += action_start_ori_tile
            next_poses -= action_start_tile
            print("prev poses", prev_poses)
            print("next poses", next_poses)

            newimg = np.zeros_like(imnext)
            for ci in range(num_obj):
                color = cv2.inRange(imgcolorhsv, colors_lower[ci], colors_upper[ci])

                if np.sum(color == 255) > 100:
                    points = np.argwhere(color == 255)
                    points = np.expand_dims(points, axis=0)
                    M = cv2.getRotationMatrix2D(
                        (prev_poses[ci * 2], prev_poses[ci * 2 + 1]), -output[ci * 3 + 2], 1
                    )
                    M[0, 2] += output[ci * 3]
                    M[1, 2] += output[ci * 3 + 1]
                    new_points = cv2.transform(points, M)
                    newimg[tuple(np.transpose(new_points))] = imgcolor[tuple(np.transpose(points))]

            # action
            cv2.arrowedLine(
                imnext,
                (action_start_ori[1], action_start_ori[0]),
                (action_end_ori[1], action_end_ori[0]),
                (255, 255, 255),
                2,
                tipLength=0.4,
            )
            cv2.arrowedLine(
                imgcolor,
                (action_start_ori[1], action_start_ori[0]),
                (action_end_ori[1], action_end_ori[0]),
                (255, 255, 255),
                2,
                tipLength=0.4,
            )

            newimg = cv2.medianBlur(newimg, 5)
            newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
            newimg = cv2.addWeighted(newimg, 0.3, imnext, 0.7, 0)
            images.append(imgcolor)
            refs.append(prev_ref[0][3:7])
            images.append(cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB))
            refs.append("prediction of " + str(prev_ref[0][3:7]))
            if len(images) == 32:
                for i in range(len(images)):
                    # cv2.imwrite(
                    #     "figures/push-prediction-plot/" + refs[i] + ".png",
                    #     cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR),
                    # )
                    plt.subplot(math.ceil(len(images) / 8), 8, i + 1), plt.imshow(images[i], "gray")
                    plt.title(refs[i])
                    plt.xticks([]), plt.yticks([])
                plt.show()
                # plt.savefig('test.png', dpi=400)
                input_str = input("One more?")
                if input_str == "y":
                    images = []
                    refs = []
                else:
                    break

    @torch.no_grad()
    def _verify_dataset(self, model, data_loader):
        import torchvision
        import matplotlib.pyplot as plt
        from PIL import Image, ImageStat
        import math
        from constants import DEPTH_MEAN, DEPTH_STD

        torch.manual_seed(1)

        inv_normalize_color = T.Normalize(
            mean=[
                -COLOR_MEAN[0] / COLOR_STD[0],
                -COLOR_MEAN[1] / COLOR_STD[1],
                -COLOR_MEAN[2] / COLOR_STD[2],
            ],
            std=[1 / COLOR_STD[0], 1 / COLOR_STD[1], 1 / COLOR_STD[2]],
        )
        inv_normalize_depth = T.Normalize(
            mean=[-DEPTH_MEAN[0] / DEPTH_STD[0]], std=[1 / DEPTH_STD[0]]
        )

        model.eval()

        for i, data in enumerate(data_loader):
            (
                prev_color_img,
                prev_depth_img,
                next_color_img,
                next_depth_img,
                used_binary_img,
                prev_poses,
                next_poses,
                action,
                delta,
                prev_ref,
                next_ref,
                action_start_ori,
                action_end_ori,
                binary_objs_total,
                num_obj,
            ) = data

            binary_objs_total = binary_objs_total[0].numpy().astype(np.uint8)
            num_obj = len(binary_objs_total)
            for i in range(num_obj):
                temp = binary_objs_total[i]
                temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
                temp *= 255
                temp = temp.astype(np.uint8)
                cv2.imshow(str(i), temp)

            np.set_printoptions(precision=3, suppress=True)
            prev_poses = prev_poses[0].numpy().astype(int)
            action = action[0].numpy().astype(int)
            action_start_tile = np.tile(action[:2], num_obj)
            print("prev poses", prev_poses)

            img = inv_normalize_color(prev_color_img[0])
            img = img.permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img *= 255
            img = img.astype(np.uint8)
            cv2.imshow("prev color", img)

            img = inv_normalize_depth(prev_depth_img[0])
            img = img.permute(1, 2, 0).numpy()
            cv2.imshow("prev depth", img)

            img = used_binary_img[0, 0].numpy().astype(int)
            img *= 255
            img = img.astype(np.uint8)
            for pi in range(num_obj):
                cv2.circle(
                    img, (prev_poses[pi * 2 + 1], prev_poses[pi * 2]), 2, (120, 102, 255), -1
                )
            cv2.imshow("prev binary", img)

            img = used_binary_img[0][1].numpy()
            img *= 255
            img = img.astype(np.uint8)
            cv2.circle(img, (action[1], action[0]), 2, (120, 102, 255), -1)
            cv2.circle(img, (action[3], action[2]), 2, (120, 102, 255), -1)
            cv2.imshow("action", img)

            img = inv_normalize_color(next_color_img[0])
            img = img.permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img *= 255
            img = img.astype(np.uint8)
            cv2.imshow("next color", img)

            img = inv_normalize_depth(next_depth_img[0])
            img = img.permute(1, 2, 0).numpy()
            cv2.imshow("next depth", img)

            action_start_ori = action_start_ori[0].numpy().astype(int)
            action_end_ori = action_end_ori[0].numpy().astype(int)
            action_start_ori_tile = np.tile(action_start_ori, num_obj)

            prev_imdepth = prev_depth_img[0].cpu().permute(1, 2, 0).numpy()
            prev_imdepth[prev_imdepth <= 0] = 0
            prev_imdepth[prev_imdepth > 0] = 255
            prev_imdepth = np.repeat(prev_imdepth, 3, axis=2)
            prev_imdepth = prev_imdepth.astype(np.uint8)
            cv2.circle(prev_imdepth, (action_start_ori[1], action_start_ori[0]), 5, (255, 0, 0), -1)
            cv2.circle(prev_imdepth, (action_end_ori[1], action_end_ori[0]), 5, (0, 0, 255), -1)
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            for pi in range(num_obj):
                cv2.circle(
                    prev_imdepth, (prev_poses[pi * 2 + 1], prev_poses[pi * 2]), 2, (255, 0, 255), -1
                )
            print("prev poses", prev_poses)

            next_imdepth = next_depth_img[0].cpu().permute(1, 2, 0).numpy()
            next_imdepth[next_imdepth <= 0] = 0
            next_imdepth[next_imdepth > 0] = 255
            next_imdepth = np.repeat(next_imdepth, 3, axis=2)
            next_imdepth = next_imdepth.astype(np.uint8)
            cv2.circle(next_imdepth, (action_start_ori[1], action_start_ori[0]), 5, (255, 0, 0), -1)
            cv2.circle(next_imdepth, (action_end_ori[1], action_end_ori[0]), 5, (0, 0, 255), -1)
            next_poses = next_poses[0].numpy().astype(int)
            next_poses += action_start_ori_tile
            next_poses -= action_start_tile
            for pi in range(num_obj):
                cv2.circle(
                    next_imdepth, (next_poses[pi * 2 + 1], next_poses[pi * 2]), 2, (255, 255, 0), -1
                )
            print("next poses", next_poses)

            delta = delta[0].numpy()
            print("delta", delta)

            cv2.imshow("prev imdepth", prev_imdepth)
            cv2.imshow("next imdepth", next_imdepth)

            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k == ord("q"):  # ESC
                break

    @torch.no_grad()
    def _symmetric_diff(self, model, criterion, data_loader):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        from constants import (
            colors_upper,
            colors_lower,
        )

        inv_normalize = T.Normalize(
            mean=[
                -COLOR_MEAN[0] / COLOR_STD[0],
                -COLOR_MEAN[1] / COLOR_STD[1],
                -COLOR_MEAN[2] / COLOR_STD[2],
            ],
            std=[1 / COLOR_STD[0], 1 / COLOR_STD[1], 1 / COLOR_STD[2]],
        )

        model.eval()

        total_symmetric_difference = []
        total_area = []
        total_num = len(data_loader.dataset)
        print(total_num)
        for i, data in enumerate(data_loader):
            (
                prev_color_img,
                prev_depth_img,
                next_color_img,
                next_depth_img,
                used_binary_img,
                prev_poses,
                next_poses,
                action,
                delta,
                prev_ref,
                next_ref,
                action_start_ori,
                action_end_ori,
                binary_objs_total,
                num_obj,
            ) = data

            prev_color_img = prev_color_img.to(self.device, non_blocking=True)
            prev_depth_img = prev_depth_img.to(self.device, non_blocking=True)
            next_color_img = next_color_img.to(self.device, non_blocking=True)
            next_depth_img = next_depth_img.to(self.device, non_blocking=True)
            used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(self.device, non_blocking=True)
            next_poses = next_poses.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            delta = delta.to(self.device, non_blocking=True)
            action_start_ori = action_start_ori.to(self.device, non_blocking=True)
            action_end_ori = action_end_ori.to(self.device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)

            output = model(
                prev_poses, action, used_binary_img, binary_objs_total, num_obj[0]
            )  # output = model(action, prev_poses)
            target = delta
            loss = criterion(output, target)
            output = output[0].cpu().numpy()
            target = target[0].cpu().numpy()
            num_obj = num_obj[0].cpu().item()
            output_xy = []
            output_a = []
            for num_idx in range(num_obj):
                output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
                output_a.append(output[num_idx * 3 + 2])
                # no move
                # output_xy.append([0, 0])
                # output_a.append(0)
                # move in the direction as the action
                # if num_idx == 0:
                #     output_xy.append([(args.distance / 0.2), 0])
                #     output_a.append(0)
                # else:
                #     output_xy.append([0, 0])
                #     output_a.append(0)
            print(i)
            print(prev_ref[0])
            # print('output', output_x_y1.numpy(), output_a1.numpy(), output_x_y2.numpy(), output_a2.numpy())
            # print('target', target.numpy())
            # print('action', action_start_ori.cpu().numpy())
            # print('loss', loss.cpu().numpy())

            # ===== symmetric difference =====
            prev_poses = prev_poses[0].cpu().numpy().astype(int)
            action_start_ori = action_start_ori[0].cpu().numpy().astype(int)
            action_start_ori_tile = np.tile(action_start_ori, num_obj)
            action = action[0].cpu().numpy().astype(int)
            action_start_tile = np.tile(action[:2], num_obj)
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            next_img = next_depth_img[0].cpu().permute(1, 2, 0).squeeze().numpy()
            pred_img_colors = [np.zeros((320, 320), dtype=np.uint8) for i in range(num_obj)]
            prev_color_img = inv_normalize(prev_color_img[0])
            prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
            imgcolor = prev_color_img
            imgcolor *= 255
            imgcolor = imgcolor.astype(np.uint8)
            imgcolor = cv2.medianBlur(imgcolor, 5)
            imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

            # prediction
            for ci in range(num_obj):
                color = cv2.inRange(imgcolor, colors_lower[ci], colors_upper[ci])
                points = np.argwhere(color == 255)
                points = np.expand_dims(points, axis=0)
                M = cv2.getRotationMatrix2D(
                    (prev_poses[ci * 2], prev_poses[ci * 2 + 1]), -output_a[ci], 1
                )
                M[0, 2] += output_xy[ci][0]
                M[1, 2] += output_xy[ci][1]
                points = cv2.transform(points, M)
                points[0, :, 0] += 48
                points[0, :, 1] += 48
                pred_img_colors[ci][tuple(np.transpose(points[0]))] = 255
                pred_img_colors[ci] = pred_img_colors[ci][48 : (320 - 48), 48 : (320 - 48)]
                pred_img_colors[ci] = cv2.medianBlur(pred_img_colors[ci], 5)

            # ground truth
            next_color_img = inv_normalize(next_color_img[0])
            next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
            next_img_color = next_color_img
            next_img_color[next_img_color < 0] = 0
            next_img_color *= 255
            next_img_color = next_img_color.astype(np.uint8)
            imgcolor = cv2.cvtColor(next_img_color, cv2.COLOR_RGB2HSV)
            next_img_colors = []
            for ci in range(num_obj):
                next_img_color = cv2.inRange(imgcolor, colors_lower[ci], colors_upper[ci])
                next_img_colors.append(next_img_color)
                total_area.append(np.sum(next_img_color == 255))

            # intersection
            for ci in range(num_obj):
                intersection_color = np.zeros_like(next_img)
                intersection_color[
                    np.logical_and(pred_img_colors[ci] == 255, next_img_colors[ci] == 255)
                ] = 255
                union_color = np.zeros_like(next_img)
                union_color[
                    np.logical_or(pred_img_colors[ci] == 255, next_img_colors[ci] == 255)
                ] = 255
                diff_color = union_color - intersection_color
                total_symmetric_difference.append(np.sum(diff_color == 255))

        print(np.average(total_area))
        print(np.std(total_area))

        diff_union = np.array(total_symmetric_difference) / np.array(total_area)
        print(np.average(diff_union))
        print(np.std(diff_union))
        np.savetxt("test.txt", diff_union)

        plt.hist(diff_union, weights=np.ones(len(diff_union)) / len(diff_union), range=(0, 2))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        # plt.show()
        plt.savefig("test.png")


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(round(M["m10"] / M["m00"]))
    cy = int(round(M["m01"] / M["m00"]))

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    # thetas = thetas + angle
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]

    return cnt_rotated


if __name__ == "__main__":
    args = parse_args()
    trainer = PushPredictionTrainer(args)
    trainer.main()