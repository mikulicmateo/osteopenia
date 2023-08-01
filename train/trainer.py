import datetime
import json
from torch import nn
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.OsteopeniaDataset import OsteopeniaDataset
from tqdm import tqdm
import pandas as pd
import xlsxwriter


class Trainer:
    def __init__(self, config_dict, model, model_name, optimizer,
                 loss_fn, start_epoch, results_output_dir, model_loaded=False):

        self.model_loaded = model_loaded
        self.config_dict = config_dict

        if torch.cuda.is_available():
            self.device = self.config_dict["device"]
        else:
            self.device = "cpu"

        # self.full_dataset = OsteopeniaDataset(
        #    self.config_dict['osteopenia_dataset_csv_path'],
        #    self.config_dict['mean'],
        #    self.config_dict['std'],
        #    self.config_dict['desired_image_size']
        # )

        # self.training_data, self.validation_data, self.test_data = \
        #     torch.utils.data.random_split(self.full_dataset, self.config_dict['n_split_train_val_test'])
        self.data_folder_path = os.path.join(os.getcwd(), 'data')

        self.training_data = OsteopeniaDataset(
            os.path.join(self.data_folder_path, 'training_dataset.csv'),
            self.config_dict['mean'],
            self.config_dict['std'],
            self.config_dict['horizontal_flip_probability'],
            self.config_dict['rotation_probability'],
            self.config_dict['rotation_angle'],
            self.config_dict['desired_image_size']
        )

        self.validation_data = OsteopeniaDataset(
            os.path.join(self.data_folder_path, 'validation_dataset.csv'),
            self.config_dict['mean'],
            self.config_dict['std'],
            0,
            0,
            0,
            self.config_dict['desired_image_size']
        )

        self.test_data = OsteopeniaDataset(
            os.path.join(self.data_folder_path, 'test_dataset.csv'),
            self.config_dict['mean'],
            self.config_dict['std'],
            0,
            0,
            0,
            self.config_dict['desired_image_size']
        )

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
        if not self.model_loaded:
            self.model = nn.DataParallel(model, device_ids=self.config_dict["device_ids"])
            self.model = self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.max_epoch = self.config_dict['max_epoch']
        self.start_epoch = start_epoch
        self.epochs_to_train = self.max_epoch - start_epoch
        self.early_stop = self.config_dict['early_stop']
        self.patience = self.config_dict['patience']

        self.results_output_dir = results_output_dir
        if not os.path.exists(self.results_output_dir):
            os.makedirs(self.results_output_dir)

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

    def evaluate_epoch(self, dataloader, test=False):
        self.model.eval()
        with torch.no_grad():
            losses = []
            correct = 0
            loop = tqdm(dataloader, leave=False, unit="batch", mininterval=0)
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

        if test:
            return correct / len(self.test_data), np.mean(losses)
        return correct / len(self.validation_data), np.mean(losses)

    def val_epoch(self):
        return self.evaluate_epoch(self.validation_dl)

    def test_model(self):
        return self.evaluate_epoch(self.test_dl, True)

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.float().to(self.device)
            prediction = self.model(image)
        return prediction.float()

    def create_model_state_dict(self, train_loss, train_accuracy, validation_loss, validation_accuracy, epoch):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.module.state_dict(),
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
        model_state = self.create_model_state_dict(train_loss, train_accuracy, validation_loss, validation_accuracy,
                                                   epoch)
        torch.save(model_state, f"model/trained/last-{self.model_name}.pt")
        if best:
            torch.save(model_state, f"model/trained/best-{self.model_name}.pt")

    def export_metrics_to_xlsx(self, best_epoch, best_score, training_dict, validation_dict):
        # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir +
                                 f"{self.model_name}" +
                                 f"_{type(self.optimizer).__name__}: " +
                                 f"{self.config_dict['learning_rate']}" +
                                 f"_{type(self.loss_fn).__name__}" +
                                 f"_frozen={(not self.model_loaded)}" +
                                 f"_{best_epoch}" + f"_{best_score:5f}.xlsx", engine='xlsxwriter')

        # Generate dataframes
        _df_train = pd.DataFrame.from_dict(training_dict)
        _df_valid = pd.DataFrame.from_dict(validation_dict)

        _df_train.to_excel(_writer, sheet_name="Training", index=False)
        _df_valid.to_excel(_writer, sheet_name="Validation", index=False)
        _writer.close()

    def train(self):

        best_val_loss = float('inf')
        best_val_acc = 0
        best_epoch = 0
        last_epoch = 0
        training_history = defaultdict(list)
        validation_history = defaultdict(list)

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"Epoch {epoch}")
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

            training_history['train_acc'].append(train_accuracy.cpu().detach().numpy())
            validation_history['val_acc'].append(validation_accuracy.cpu().detach().numpy())
            training_history['train_loss'].append(train_loss)
            validation_history['val_loss'].append(validation_loss)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        self.export_metrics_to_xlsx(best_epoch,
                                    best_val_acc,
                                    training_history,
                                    validation_history)

        if self.early_stop:
            self.start_epoch = best_epoch
            self.epochs_to_train = self.max_epoch - self.start_epoch
        else:
            self.start_epoch = last_epoch
            self.epochs_to_train = self.max_epoch - self.start_epoch