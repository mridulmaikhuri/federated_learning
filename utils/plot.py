import matplotlib.pyplot as plt

def filter_none(x_list, y_list):
    filtered_x, filtered_y = zip(*[(x, y) for x, y in zip(x_list, y_list) if y is not None])
    return list(filtered_x), list(filtered_y)

def plot_dashboard(history):    
    # Create a single figure with subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  

    # === Loss over Rounds ===
    rounds_loss, train_loss = filter_none(history["round"], history["train_loss"])
    rounds_val_loss, val_loss = filter_none(history["round"], history["val_loss"])
    
    axs[0, 0].plot(rounds_loss, train_loss, label="Train Loss", marker="o")
    axs[0, 0].plot(rounds_val_loss, val_loss, label="Validation Loss", marker="s")
    axs[0, 0].set_title("Loss Over Rounds")
    axs[0, 0].set_xlabel("Round")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # === Accuracy over Rounds ===
    rounds_acc, train_acc = filter_none(history["round"], history["train_accuracy"])
    rounds_val_acc, val_acc = filter_none(history["round"], history["val_accuracy"])
    
    axs[0, 1].plot(rounds_acc, train_acc, label="Train Accuracy", marker="o")
    axs[0, 1].plot(rounds_val_acc, val_acc, label="Validation Accuracy", marker="s")
    axs[0, 1].set_title("Accuracy Over Rounds")
    axs[0, 1].set_xlabel("Round")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    # === Loss Gap (Train - Val) ===
    loss_gap = [tl - vl for tl, vl in zip(train_loss, val_loss)]
    axs[1, 0].plot(rounds_loss, loss_gap, label="Train - Val Loss Gap", marker="o", color="red")
    axs[1, 0].set_title("Train vs Validation Loss Gap")
    axs[1, 0].set_xlabel("Round")
    axs[1, 0].set_ylabel("Loss Gap")
    axs[1, 0].legend()

    # === Accuracy Gap (Train - Val) ===
    acc_gap = [ta - va for ta, va in zip(train_acc, val_acc)]
    axs[1, 1].plot(rounds_acc, acc_gap, label="Train - Val Accuracy Gap", marker="o", color="purple")
    axs[1, 1].set_title("Train vs Validation Accuracy Gap")
    axs[1, 1].set_xlabel("Round")
    axs[1, 1].set_ylabel("Accuracy Gap")
    axs[1, 1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show all plots in a single frame
    plt.show()

if __name__ == "__main__":
    pass
