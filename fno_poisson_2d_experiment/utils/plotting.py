import matplotlib.pyplot as plt


def plot_field(field, title):
    plt.imshow(field, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.show()
