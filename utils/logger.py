import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import os
import utils.util as util
import torch
import torchvision


class Log:
    def __init__(self, project_info: dict, logger_name: str, is_active: bool, log_dir: str):
        self.project_info = project_info
        self.logger_name = logger_name.lower()
        self.is_active = is_active
        self.log_dir = log_dir
        self.logger = self.init_logger()

    def init_logger(self):
        if not self.is_active:
            return DefaultLogger()

        if self.logger_name == "wandb":
            return WandbLogger(self.project_info, self.log_dir, self.is_active)
        elif self.logger_name == "tensorboard":
            return TBLogger(self.project_info, self.log_dir, self.is_active)
        else:
            raise ValueError(f"Invalid logger type: {self.logger_name}")


# ================= ABSTRACT LOGGER =================
class ILog(ABC):
    @abstractmethod
    def init(self, name):
        pass

    @abstractmethod
    def log_scaler(self, scalers: dict, step: int):
        pass

    @abstractmethod
    def log_model(self, model, epoch, step, loss, acc):
        pass

    @abstractmethod
    def log_image(self, batch, step, stage):
        pass

    def alert(self, text):
        print("| Warning |", text)


# ================= DEFAULT LOGGER =================
class DefaultLogger(ILog):
    def init(self, name):
        pass

    def log_scaler(self, scalers: dict, step: int):
        pass

    def log_model(self, model, epoch, step, loss, acc):
        pass

    def log_image(self, batch, step, stage):
        pass


# ================= TENSORBOARD LOGGER =================
class TBLogger(ILog):
    def __init__(self, project_info, log_dir: str, is_active: bool):
        self.is_active = is_active
        self.log_dir = log_dir
        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.log_dir_edited = os.path.join(self.log_dir, f"tb_{date}")
        os.makedirs(self.log_dir_edited, exist_ok=True)

        util.writeJson(project_info, os.path.join(self.log_dir_edited, "info.json"))
        self.init(self.log_dir_edited)

    def init(self, log_dir):
        if self.is_active:
            self.logger = SummaryWriter(log_dir)

    def log_scaler(self, scalers: dict, step: int):
        if self.is_active:
            for k, v in scalers.items():
                self.logger.add_scalar(k, v, step)

    def log_model(self, model, epoch, step, loss, acc):
        if self.is_active:
            models_dir = os.path.join(self.log_dir_edited, "models")
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(
                models_dir,
                f"net_best_epoch_{epoch}__iter_{step}__loss_{round(loss, 4)}__acc_{round(acc, 4)}.pth"
            )
            torch.save(model.state_dict(), save_path)

    def log_image(self, batch, step, stage):
        if self.is_active:
            grid = torchvision.utils.make_grid(batch)
            self.logger.add_image(f"{stage}/images", grid, step)


# ================= WANDB LOGGER =================
class WandbLogger(ILog):
    def __init__(self, project_info, log_dir: str, is_active: bool):
        self.is_active = is_active
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.init(project_info.get("name", "default_project"), self.log_dir)

        wandb.run.name = f"{project_info.get('archname', 'model')}_{wandb.run.id}"
        util.writeJson(project_info, os.path.join(wandb.run.dir, "info.json"))

    def init(self, name, log_dir):
        if self.is_active:
            wandb.init(project=name, dir=log_dir)

    def log_scaler(self, scalers: dict, step: int):
        if self.is_active:
            wandb.log(scalers, step=step)

    def log_model(self, model, epoch, step, loss, acc):
        if self.is_active:
            models_dir = os.path.join(wandb.run.dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(
                models_dir,
                f"net_best_epoch_{epoch}__iter_{step}__loss_{round(loss, 4)}__acc_{round(acc, 4)}.pth"
            )
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

    def log_image(self, batch, step, stage):
        if self.is_active:
            grid = torchvision.utils.make_grid(batch)
            image = torchvision.transforms.ToPILImage()(grid.cpu())
            wandb.log({f"{stage}/images": wandb.Image(image)}, step=step)
