import matplotlib.pyplot as plt


def plot_points(pos):
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


def plot_points_sol(pos, X):
    plt.scatter(pos[:, 0], pos[:, 1])
    for node, [a, b] in enumerate(X):
        if a != -1:
            plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'k-')
        if b != -1:
            plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'k-')
    plt.show()


def plot_points_sol_intermediate(pos, X, X_int):
    plt.scatter(pos[:, 0], pos[:, 1], s=80, alpha=0.8)
    for node, [a, b] in enumerate(X):
        plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'k-')
        # plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'k-')
    for node, [a, b] in enumerate(X_int):
        if a != -1:
            plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'r-')
        if b != -1:
            plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'r-')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # plt.savefig('filename.eps', format='eps')
    # input()
    plt.show()
