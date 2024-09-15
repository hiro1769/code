import torch
import wandb
from loss_meter import LossMeter
from math import inf
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        """
        :param patience: How many epochs to wait after last time validation loss improved.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss  # Initialize the best loss with the first validation loss
        elif val_loss < self.best_loss - self.min_delta:
            # If validation loss decreases significantly, reset the counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Otherwise, increase the counter
            self.counter = self.counter + 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, config=None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config
        self.model = model
        self.val_count = 0
        self.train_epoch = 0
        self.train_step_count = len(self.gen_set[0][0])
        self.val_step_count = len(self.gen_set[0][1])
        self.count = 1
        self.num_count = 0
        self.best_val_loss = inf
        self.early_stopping = EarlyStopping(patience=30, min_delta=0.001)  # 设定早停
        
        if config["wandb"]["wandb_on"]:
            wandb.init(
                entity=self.config["wandb"]["entity"],
                project=self.config["wandb"]["project"],
                notes=self.config["wandb"]["notes"],
                tags=self.config["wandb"]["tags"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

    def train(self, epoch, data_loader):
        total_loss_meter = LossMeter()
        step_loss_meter = LossMeter()
        pre_step = self.train_step_count
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            torch.cuda.empty_cache()
            print(f"Processing batch {batch_idx} of epoch {epoch}")
            print("epoch:", epoch, loss.get_loss_dict_for_print("train"))
            self.num_count = self.num_count + 1
            wandb.log(loss.get_loss_dict_for_print("train"), step=self.num_count)
            if ((batch_idx + 1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or (self.train_step_count == pre_step and batch_idx == len(data_loader) - 1):
                if self.config["wandb"]["wandb_on"]:
                    wandb.log(step_loss_meter.get_avg_results(), step=self.train_step_count * self.count)
                    wandb.log({"step_lr": self.model.scheduler.get_last_lr()[0]}, step=self.train_step_count * self.count)
                self.model.scheduler.step(self.count)
                step_loss_meter.init()

        if self.config["wandb"]["wandb_on"]:
            wandb.log(total_loss_meter.get_avg_results(), step=self.train_step_count * self.count)
            self.train_epoch = self.train_epoch + 1
        self.model.save("train")

    def test(self, epoch, data_loader, save_best_model):
        total_loss_meter = LossMeter()
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))
            print("epoch:", epoch, loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step=self.train_step_count * self.count)
            self.count = self.count + 1

        # Early stopping logic
        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.model.save("val")
                print("Best model saved with validation loss:", self.best_val_loss)

        # Check if we should stop early
        self.early_stopping(avg_total_loss["total_val"])
        if self.early_stopping.early_stop:
            print("Early stopping triggered!")
            return True  # Stop the training

        return False

    def run(self):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        for epoch in range(60):
            self.train(epoch, train_data_loader)
            should_stop = self.test(epoch, val_data_loader, True)
            if should_stop:  # Early stopping condition met
                break

    # def run(self):
    #     train_data_loader = self.gen_set[0][0]
    #     val_data_loader = self.gen_set[0][1]
    #     for epoch in range(60):
    #         self.train(epoch, train_data_loader)
    #         self.test(epoch, val_data_loader, True)#保存最好的模型添加早停