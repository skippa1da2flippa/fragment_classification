import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_curves(csv_paths: list[str], names: list[str], plot_name: str | None = None):
    for i, (path, name) in enumerate(zip(csv_paths, names)):
        df = pd.read_csv(path)
        df = df.dropna(subset=['GAT_val_acc'])
        acc = df['GAT_val_acc']
        epochs = df['epoch']
        plt.plot(epochs, acc, label=name, color=f'C{i}')
        # for j, (x, y) in enumerate(zip(epochs, acc)):
        #     if j % 5 == 0:
        #         plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
        max_acc = acc.max()
        max_epoch = epochs[acc.idxmax()]
        plt.scatter(max_epoch, max_acc, color=f'C{i}', s=50)
        plt.annotate(f'{max_acc:.3f}', (max_epoch, max_acc), textcoords="offset points", xytext=(0, 5), ha='center')


    plot_name = plot_name if plot_name else "accuracy_plot.png"

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_name)
    print(f"Plot saved as '{plot_name}'")

    

