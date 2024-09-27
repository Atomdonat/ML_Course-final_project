import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, repeat
import re


def get_lr_and_wd_and_data_from_output(filename: str, to_vars: bool = True):
    # Step 1: split into iteration
    criterion_line = "criterion:    nn.CrossEntropyLoss"
    parts = []
    current_part = []

    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == criterion_line:
                parts.append(''.join(current_part))
                current_part = []
            else:
                current_part.append(line)

    parts.append(''.join(current_part))  # Add the last part

    data = []
    current_data = {}
    for i, part in enumerate(parts):
        if i == 0:
            continue
        # Step 2: get weights
        current_data.clear()
        lr_pattern = r"lr:\s+(\S+)"
        weight_decay_pattern = r"weight_decay:\s+(\S+)"

        lr_match = re.search(lr_pattern, part)
        weight_decay_match = re.search(weight_decay_pattern, part)

        if lr_match and weight_decay_match:
            lr = float(lr_match.group(1))
            weight_decay = float(weight_decay_match.group(1))
            current_data = {"learn_rate": lr, "weight_decay": weight_decay}

        # Step 3: extract data
        # Initialize lists to hold the values
        lists = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": []
        }

        # Split the log by lines
        lines = part.strip().split('\n')

        for line in lines:
            if 'train Loss' in line:
                parts = line.split()
                lists["train_loss"].append(float(parts[2]))
                lists["train_accuracy"].append(float(parts[4]))
            elif 'val Loss' in line:
                parts = line.split()
                lists["validation_loss"].append(float(parts[2]))
                lists["validation_accuracy"].append(float(parts[4]))

        current_data["data"] = lists
        # print(current_data)
        data.append(current_data)
        # print(data[-1])
        if to_vars:
            lrstr = str(f"{current_data['learn_rate']:.0e}").replace("-", "_")
            wdstr = str(f"{current_data['weight_decay']:.0e}").replace("-", "_")
            print(f"_{lrstr}_{wdstr} = {current_data['data']}")
        else:
            print(f"learn rate {current_data['learn_rate']}: weight decay {current_data['weight_decay']}: {current_data['data']}")


def plot_training(
        plot_data1=None, label1="",
        plot_data2=None, label2="",
        plot_data3=None, label3="",
        plot_data4=None, label4="",
        plot_data5=None, label5="",
        plot_data6=None, label6="",
        epochs_count=5,
        const_lr: bool = True,
        titel: str = "NaN"
):
    if all([
        plot_data1 is None,
        plot_data2 is None,
        plot_data3 is None,
        plot_data4 is None,
        plot_data5 is None,
        plot_data6 is None,
    ]):
        raise ValueError("atleast one plot must be provided")

    _x = range(0, epochs_count)

    # Plot Trendlines for each Accuracy
    def pad_list(lst, length):
        return list(chain(lst, repeat(0, max(0, length - len(lst)))))

    for plot_data in [plot_data1,plot_data2,plot_data3,plot_data4,plot_data5,plot_data6]:
        if plot_data is not None:
            for key,value in plot_data.items():
                plot_data[key] = pad_list(value, epochs_count)

    l_max = 2.7  #np.ceil(max(plot_data["train_loss"] + plot_data["scratch_train_loss"] + plot_data["validation_loss"] + plot_data["scratch_validation_loss"]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey='row')

    if plot_data1:
        axes[0, 0].plot(_x, plot_data1["train_accuracy"], label=label1, color='darkviolet', linewidth=1)
        axes[0, 1].plot(_x, plot_data1["validation_accuracy"], label=label1, color='darkviolet', linewidth=1)
        axes[1, 0].plot(_x, plot_data1["train_loss"], label=label1, color='darkviolet', linewidth=1)
        axes[1, 1].plot(_x, plot_data1["validation_loss"], label=label1, color='darkviolet', linewidth=1)

    if plot_data2:
        axes[0, 0].plot(_x, plot_data2["train_accuracy"], label=label2, color='blue', linewidth=1)
        axes[0, 1].plot(_x, plot_data2["validation_accuracy"], label=label2, color='blue', linewidth=1)
        axes[1, 0].plot(_x, plot_data2["train_loss"], label=label2, color='blue', linewidth=1)
        axes[1, 1].plot(_x, plot_data2["validation_loss"], label=label2, color='blue', linewidth=1)

    if plot_data3:
        axes[0, 0].plot(_x, plot_data3["train_accuracy"], label=label3, color='limegreen', linewidth=1)
        axes[0, 1].plot(_x, plot_data3["validation_accuracy"], label=label3, color='limegreen', linewidth=1)
        axes[1, 0].plot(_x, plot_data3["train_loss"], label=label3, color='limegreen', linewidth=1)
        axes[1, 1].plot(_x, plot_data3["validation_loss"], label=label3, color='limegreen', linewidth=1)

    if plot_data4:
        axes[0, 0].plot(_x, plot_data4["train_accuracy"], label=label4, color='gold', linewidth=1)
        axes[0, 1].plot(_x, plot_data4["validation_accuracy"], label=label4, color='gold', linewidth=1)
        axes[1, 0].plot(_x, plot_data4["train_loss"], label=label4, color='gold', linewidth=1)
        axes[1, 1].plot(_x, plot_data4["validation_loss"], label=label4, color='gold', linewidth=1)

    if plot_data5:
        axes[0, 0].plot(_x, plot_data5["train_accuracy"], label=label5, color='orange', linewidth=1)
        axes[0, 1].plot(_x, plot_data5["validation_accuracy"], label=label5, color='orange', linewidth=1)
        axes[1, 0].plot(_x, plot_data5["train_loss"], label=label5, color='orange', linewidth=1)
        axes[1, 1].plot(_x, plot_data5["validation_loss"], label=label5, color='orange', linewidth=1)

    if plot_data6:
        axes[0, 0].plot(_x, plot_data6["train_accuracy"], label=label6, color='red', linewidth=1)
        axes[0, 1].plot(_x, plot_data6["validation_accuracy"], label=label6, color='red', linewidth=1)
        axes[1, 0].plot(_x, plot_data6["train_loss"], label=label6, color='red', linewidth=1)
        axes[1, 1].plot(_x, plot_data6["validation_loss"], label=label6, color='red', linewidth=1)

    # Plot settings
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].yaxis.tick_right()
    axes[0, 0].yaxis.set_label_position("right")
    axes[0, 0].set_ylim((0, 1.1))
    axes[0, 0].set_yticks(np.arange(0, 1.1, 0.1))
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].yaxis.tick_right()
    axes[1, 0].yaxis.set_label_position("right")
    axes[1, 0].set_ylim((0, l_max))
    axes[1, 0].set_yticks(np.arange(0, l_max, l_max/10))

    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    if const_lr:
        a = "learn_rate"
        a0 = "Learning Rate"
        a1 = "Weight Decays"
    else:
        a = "weight_decay"
        a0 = "Weight Decay"
        a1 = "Learning Rates"

    # fig.suptitle(f"Constant {a0} of {titel} with different {a1}")
    fig.suptitle("Trained Models (Pretrained lr=1e-3, wd=5e-4; Scratch lr=5e-4, wd=5e-4)")
    plt.tight_layout()

    plt.savefig(f"E:/Programmieren/Github_Repositories/ML_Course-final_project/training_history/plots/pretrained_lr_1e_3_wd_5e_4.png", dpi=300, bbox_inches='tight')
    plt.show()

    return plt


def plot_to_grid(plots: list, rows: int):
    cols = len(plots) // rows

    fig, axs = plt.subplots(rows, cols)

    # Plot data in each subplot
    for i, ax in enumerate(axs.flat):
        ax.imshow(plots[i])
        ax.axis('off')  # Turn off axis labels and ticks

    fig.suptitle(f"Constant Learning Rates with different Weight Decays")
    # plt.savefig(f"E:/Programmieren/Github_Repositories/ML_Course-final_project/training_history/plots/scratch_finetuning/constant_learn_rates.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    output_file = "helper_scripts/scratch_parameter_tuning.txt"
    get_lr_and_wd_and_data_from_output(output_file, True)

    _1e_03_5e_04 = {'train_loss': [0.5632, 0.1971, 0.1514, 0.1288, 0.1123, 0.1034, 0.0946, 0.0887, 0.082, 0.0772, 0.0747, 0.0718, 0.0675, 0.0652, 0.0637],
                    'train_accuracy': [0.8725, 0.9506, 0.9584, 0.9635, 0.969, 0.9688, 0.9716, 0.9744, 0.9758, 0.9775, 0.9782, 0.978, 0.9796, 0.9808, 0.981],
                    'validation_loss': [0.2247, 0.1613, 0.1331, 0.1187, 0.1089, 0.103, 0.0976, 0.094, 0.0899, 0.0897, 0.0867, 0.0846, 0.0843, 0.0821, 0.0803],
                    'validation_accuracy': [0.94, 0.9548, 0.9619, 0.9631, 0.9669, 0.9667, 0.9689, 0.9698, 0.9702, 0.9707, 0.972, 0.9722, 0.9724, 0.972, 0.973]}
    _5e_04_5e_04 = {'train_loss': [1.5354, 0.9885, 0.7108, 0.5577, 0.464, 0.4051, 0.329, 0.3007, 0.262, 0.2413, 0.2131, 0.167, 0.1596, 0.1397, 0.1229],
                    'train_accuracy': [0.4568, 0.6432, 0.7438, 0.8029, 0.8345, 0.8584, 0.8847, 0.8936, 0.9047, 0.9151, 0.9245, 0.9428, 0.943, 0.953, 0.9559],
                    'validation_loss': [1.1951, 0.8597, 0.6374, 0.4911, 0.4844, 0.3655, 0.3986, 0.32, 0.293, 0.2716, 0.2854, 0.2907, 0.2853, 0.2748, 0.2688],
                    'validation_accuracy': [0.5683, 0.6952, 0.7728, 0.8187, 0.8281, 0.8726, 0.8619, 0.8878, 0.8989, 0.903, 0.9044, 0.9031, 0.9057, 0.9115, 0.9109]}

    plot_training(
        plot_data1=None,  label1="",
        plot_data2=_1e_03_5e_04,  label2="Pretrained",
        plot_data3=None,  label3="",
        plot_data4=None, label4="",
        plot_data5=None, label5="",
        plot_data6=_5e_04_5e_04,  label6="Scratch",
        epochs_count=15, const_lr=False, titel="")
#     plot_training(
#         plot_data1=_1e_04_5e_05, label1="0.00010",
#         plot_data2=_5e_04_5e_05, label2="0.00050",
#         plot_data3=_1e_03_5e_05, label3="0.00100",
#         plot_data4=None, label4="",
#         plot_data5=_25e_04_5e_05, label5="0.00250",
#         plot_data6=_5e_03_5e_05, label6="0.00500",
#         epochs_count=15, const_lr=False, titel="5e-4")

    # plist = [
    #     plot_training(plot_data2=_5e_5__1e_2, plot_data3=_5e_5__5e_3, plot_data5=_5e_5__5e_4, plot_data6=_5e_5__5e_5, epochs_count=15, const_lr=True, titel="5e-5"),
    #     plot_training(plot_data2=_8e_5__1e_2, plot_data3=_8e_5__5e_3, plot_data5=_8e_5__5e_4, plot_data6=_8e_5__5e_5, epochs_count=15, const_lr=True, titel="8e-5"),
    #     plot_training(plot_data2=_1e_4__1e_2, plot_data3=_1e_4__5e_3, plot_data5=_1e_4__5e_4, plot_data6=_1e_4__5e_5, epochs_count=15, const_lr=True, titel="1e-4"),
    #     plot_training(plot_data2=_2e_4__1e_2, plot_data3=_2e_4__5e_3, plot_data5=_2e_4__5e_4, plot_data6=_2e_4__5e_5, epochs_count=15, const_lr=True, titel="2e-4"),
    #     plot_training(plot_data2=_25e_5__1e_2, plot_data3=_25e_5__5e_3, plot_data5=_25e_5__5e_4, plot_data6=_25e_5__5e_5, epochs_count=15, const_lr=True, titel="25e-5"),
    #     plot_training(plot_data2=_5e_4__1e_2, plot_data3=_5e_4__5e_3, plot_data5=_5e_4__5e_4, plot_data6=_5e_4__5e_5, epochs_count=15, const_lr=True, titel="5e-4")
    # ]
    # plot_to_grid(plist,2)
