import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import os

os.chdir("model/results/")

if not os.path.exists("plots"):
    os.makedirs("plots")
    

for file in os.listdir():
    if file == "plots":
        continue
    
    current_dir = os.getcwd()

    dataframe = pd.read_excel(file, sheet_name=['Training', 'Validation'])

    file_split = file.split('_') 
    if file_split[0] == "efficientnet":
        folder_name = f"{file_split[0]}_{file_split[1]}"
    elif file_split[0] == "mobilenetv3":
        folder_name = f"{file_split[0]}_{file_split[1]}"
    else:
        folder_name = file_split[0]

    os.chdir("plots/")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure()
    plt.plot(dataframe["Training"]['train_acc'], label='train accuracy')
    plt.plot(dataframe["Validation"]['val_acc'], label='validation accuracy')
    # Graph chars
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f"{folder_name}/{file}_accuracy.png")
    plt.close()

    plt.figure()
    # Plot training and validation accuracy
    plt.plot(dataframe["Training"]['train_loss'], label='train loss')
    plt.plot(dataframe["Validation"]['val_loss'], label='validation loss')
    # Graph chars
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f"{folder_name}/{file}_loss.png")
    plt.close()

    os.chdir(current_dir)