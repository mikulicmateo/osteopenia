import datetime
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.OsteopeniaDataset import OsteopeniaDataset
from tqdm import tqdm


class Trainer:
    def __init__(self, config_file_path, model, model_name, optimizer, loss_fn, start_epoch):
        with open(config_file_path, "r") as config_file:
            self.config_dict = json.load(config_file)

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.full_dataset = OsteopeniaDataset(
            self.config_dict['osteopenia_dataset_csv_path'],
            self.config_dict['desired_image_size']
        )

        self.training_data, self.validation_data, self.test_data = \
            torch.utils.data.random_split(self.full_dataset, self.config_dict['n_split_train_val_test'])

        self.batch_size = self.config_dict['batch_size']
        self.num_workers = self.config_dict['num_workers']

        self.training_dl = self.create_dataloader(data=self.training_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        self.validation_dl = self.create_dataloader(data=self.validation_data, batch_size=self.batch_size,
                                                    num_workers=self.num_workers)
        self.test_dl = self.create_dataloader(data=self.test_data, batch_size=self.batch_size,
                                              num_workers=self.num_workers)

        self.model_name = model_name
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.max_epoch = self.config_dict['max_epoch']
        self.start_epoch = start_epoch
        self.epochs_to_train = self.max_epoch - start_epoch
        self.early_stop = self.config_dict['early_stop']
        self.patience = self.config_dict['patience']

    def create_dataloader(self, data, batch_size, num_workers, shuffle=True):
        return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def train_epoch(self, epoch_num):
        self.model.train()
        losses = []
        correct = 0

        loop = tqdm(self.training_dl, leave=False, unit="batch", mininterval=0)
        loop.set_description(f'Epoch {epoch_num}')
        for features, labels in loop:
            features = features.float().to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(features)
            labels = labels.unsqueeze(1)
            loss = self.loss_fn(predictions, labels.float())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            accuracy = (predictions.round() == labels).float().mean()
            correct += (predictions.round() == labels).float().sum()
            losses.append(float(loss.item()))

            loop.set_postfix(
                loss=float(loss.item()),
                accuracy=float(accuracy)
            )
            # TODO weighted scores

        return correct / len(self.training_data), np.mean(losses)

    def val_epoch(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            correct = 0
            loop = tqdm(self.validation_dl, leave=False, unit="batch", mininterval=0)
            for features, labels in loop:
                features = features.float().to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(features)
                labels = labels.unsqueeze(1)
                loss = self.loss_fn(predictions, labels.float())

                accuracy = (predictions.round() == labels).float().mean()
                correct += (predictions.round() == labels).float().sum()
                losses.append(float(loss.item()))

                loop.set_postfix(
                    loss=float(loss.item()),
                    accuracy=float(accuracy)
                )

        return correct / len(self.validation_data), np.mean(losses)

    def create_model_state_dict(self, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': validation_loss,
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy
        }
        return model_state

    def save_model(self, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch, best):
        model_state = self.create_model_state_dict(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch)
        torch.save(model_state, f"model/trained/last-{self.model_name}.pt")
        if best:
            torch.save(model_state, f"model/trained/best-{self.model_name}.pt")

    def train(self):

        best_val_loss = float('inf')
        best_val_acc = 0
        best_epoch = 0
        last_epoch=0
        history = defaultdict(list)

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):
            last_epoch = epoch
            train_accuracy, train_loss = self.train_epoch(epoch)
            print(f"Training Loss: {train_loss}, Training accuracy: {train_accuracy}")

            validation_accuracy, validation_loss = self.val_epoch()
            print(f"Validation Loss: {validation_loss}, Val accuracy: {validation_accuracy}")

            self.save_model(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch, False)
            if validation_accuracy > best_val_acc:
                self.save_model(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch, True)
                best_val_acc = validation_accuracy
                best_epoch = epoch

            if validation_loss > best_val_loss:
                best_val_loss = validation_loss

            if self.early_stop and (epoch - best_epoch) > self.patience:
                print(f'Early stop @ epoch {epoch}, best epoch {best_epoch}')
                break

            history['train_acc'].append(train_accuracy.cpu().detach().numpy())
            history['val_acc'].append(validation_accuracy.cpu().detach().numpy())
            history['train_loss'].append(train_loss)
            history['val_loss'].append(validation_loss)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # Plot training and validation accuracy
        plt.plot(history['train_acc'], label='train accuracy')
        plt.plot(history['val_acc'], label='validation accuracy')

        # Graph chars
        plt.title('Training history')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig("training_acc.png")

        # Plot training and validation accuracy
        plt.plot(history['train_loss'], label='train loss')
        plt.plot(history['val_loss'], label='validation loss')

        # Graph chars
        plt.title('Training history')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig("training_loss.png")

        print("Finished training")

        if self.early_stop:
            self.start_epoch = best_epoch
            self.epochs_to_train = self.max_epoch - self.start_epoch
        else:
            self.start_epoch = last_epoch
            self.epochs_to_train = self.max_epoch - self.start_epoch