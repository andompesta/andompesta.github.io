from typing import Callable
import numpy as np
import torch
from matplotlib import pyplot as plt, animation


def get_device():
    return "cpu"


def plot_vdp_trajectories(
    ts: np.ndarray,
    Y: np.ndarray,
    ode_rhs: Callable,
):
    N_ = 10
    Y = Y.transpose(1, 0, 2)
    min_x, min_y = Y.min(axis=0).min(axis=0)
    max_x, max_y = Y.max(axis=0).max(axis=0)

    xs1_, xs2_ = np.meshgrid(
        np.linspace(min_x, max_x, N_),
        np.linspace(min_y, max_y, N_),
    )

    Z = np.array(
        [
            xs1_.T.flatten(),
            xs2_.T.flatten(),
        ]
    ).T
    Z = torch.from_numpy(Z).float()
    Z = Z.to(get_device()).contiguous()

    with torch.no_grad():
        F = ode_rhs(None, Z).detach().cpu().numpy()

    F /= ((F**2).sum(-1, keepdims=True)) ** (0.25)
    Z = Z.detach().cpu().numpy()

    fig = plt.figure(
        1,
        [15, 7.5],
        constrained_layout=True,
    )
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:, 0])

    N = Y.shape[0]
    if N > 3:
        print("Plotting the first 3 data sequences.")
        N = 3

    ax1.set_xlabel("State $x_1$", fontsize=17)
    ax1.set_ylabel("State $x_2$", fontsize=17)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)

    Z = Z.reshape(N_, N_, -1)
    F = F.reshape(N_, N_, -1)
    h1 = ax1.quiver(
        Z[:, :, 0],
        Z[:, :, 1],
        F[:, :, 0],
        F[:, :, 1],
        np.linalg.norm(F, axis=-1),
        cmap="viridis",
    )

    for n in range(N):
        (h2,) = ax1.plot(
            Y[n, 0, 0],
            Y[n, 0, 1],
            "o",
            fillstyle="none",
            markersize=11.0,
            linewidth=2.0,
        )
        (h3,) = ax1.plot(
            Y[n, :, 0],
            Y[n, :, 1],
            "-",
            color=h2.get_color(),
            linewidth=3.0,
        )
    plt.legend(
        [h1, h2, h3],
        ["Vector field", "Initial value", "Observed sequence"],
        loc="lower right",
        fontsize=20,
        bbox_to_anchor=(1.5, 0.05),
    )

    ax2 = fig.add_subplot(gs[0, 1:])
    for n in range(N):
        (h4,) = ax2.plot(ts, Y[n, :, 0])
    ax2.set_xlabel("time", fontsize=17)
    ax2.set_ylabel("State $x_1$", fontsize=17)

    ax3 = fig.add_subplot(gs[1, 1:])
    for n in range(N):
        (h5,) = ax3.plot(ts, Y[n, :, 1])
    ax3.set_xlabel("time", fontsize=17)
    ax3.set_ylabel("State $x_2$", fontsize=17)

    plt.show()
    plt.close()


def plot_ode(
    t: np.ndarray,
    X: np.ndarray,
    ode_rhs: Callable,
    Xhat: np.ndarray | None = None,
    L: int = 1,
    return_fig: bool = False,
):
    N_ = 10
    X = X.transpose(1, 0, 2)

    if Xhat is not None:
        Xhat = Xhat.transpose(1, 0, 2)
    min_x, min_y = X.min(axis=0).min(axis=0)
    max_x, max_y = X.max(axis=0).max(axis=0)
    xs1_, xs2_ = np.meshgrid(
        np.linspace(min_x, max_x, N_),
        np.linspace(min_y, max_y, N_),
    )
    Z = np.array(
        [
            xs1_.T.flatten(),
            xs2_.T.flatten(),
        ]
    ).T
    Z = torch.from_numpy(Z).float()
    Z = torch.stack([Z] * L)
    Z = Z.to(get_device()).contiguous()

    with torch.no_grad():
        F = ode_rhs(None, Z).detach().cpu().numpy()

    F /= ((F**2).sum(-1, keepdims=True)) ** (0.25)
    Z = Z.detach().cpu().numpy()

    fig = plt.figure(
        num=1,
        figsize=[7.5, 5],
        constrained_layout=True,
    )
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:, 0])

    ax1.set_xlabel("State $x_1$", fontsize=10)
    ax1.set_ylabel("State $x_2$", fontsize=10)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    for Z_, F_ in zip(Z, F):
        Z_ = Z_.reshape(N_, N_, -1)
        F_ = F_.reshape(N_, N_, -1)
        h1 = ax1.quiver(
            Z_[:, :, 0],
            Z_[:, :, 1],
            F_[:, :, 0],
            F_[:, :, 1],
            np.linalg.norm(F_, axis=-1),
            cmap="viridis",
        )

    if Xhat is None:  # only plotting data
        for X_ in X:
            (h2,) = ax1.plot(
                X_[0, 0],
                X_[0, 1],
                "o",
                fillstyle="none",
                markersize=11.0,
                linewidth=2.0,
            )
            (h3,) = ax1.plot(
                X_[:, 0],
                X_[:, 1],
                "-",
                color=h2.get_color(),
                linewidth=3.0,
            )
    else:  # plotting data and fits, set the color correctly!
        (h2,) = ax1.plot(
            X[0, 0, 0],
            X[0, 0, 1],
            "o",
            color="firebrick",
            fillstyle="none",
            markersize=11.0,
            linewidth=2.0,
        )
        (h3,) = ax1.plot(
            X[0, :, 0],
            X[0, :, 1],
            "-",
            color="firebrick",
            linewidth=3.0,
        )
    if Xhat is not None and Xhat.ndim == 3:
        Xhat = np.expand_dims(Xhat, 0)
    if Xhat is None:
        plt.legend(
            [h1, h2, h3],
            ["Vector field", "Initial value", "State trajectory"],
            loc="lower right",
            fontsize=10,
            bbox_to_anchor=(1.5, 0.05),
        )
    else:
        for xhat in Xhat:
            (h4,) = ax1.plot(
                xhat[0, :, 0],
                xhat[0, :, 1],
                "-",
                color="royalblue",
                linewidth=3.0,
            )
        if Xhat.shape[0] > 1:
            ax1.plot(
                X[0, :, 0],
                X[0, :, 1],
                "-",
                color="firebrick",
                linewidth=5.0,
            )
        plt.legend(
            [h1, h2, h3, h4],
            ["Vector field", "Initial value", "Data sequence", "Forward simulation"],
            loc="lower right",
            fontsize=20,
            bbox_to_anchor=(1.5, 0.05),
        )

    ax2 = fig.add_subplot(gs[0, 1:])
    if Xhat is None:  # only plotting data
        for X_ in X:
            (h4,) = ax2.plot(
                t,
                X_[:, 0],
                linewidth=3.0,
            )
    else:  # plotting data and fits, set the color correctly!
        (h4,) = ax2.plot(
            t,
            X[0, :, 0],
            color="firebrick",
            linewidth=3.0,
        )
    if Xhat is not None:
        for xhat in Xhat:
            ax2.plot(
                t,
                xhat[0, :, 0],
                color="royalblue",
                linewidth=3.0,
            )
        if Xhat.shape[0] > 1:
            ax2.plot(
                t,
                X[0, :, 0],
                color="firebrick",
                linewidth=5.0,
            )
    ax2.set_xlabel("time", fontsize=10)
    ax2.set_ylabel("State $x_1$", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1:])

    if Xhat is None:  # only plotting data
        for X_ in X:
            (h5,) = ax3.plot(t, X_[:, 1], linewidth=3.0)
    else:  # plotting data and fits, set the color correctly!
        (h5,) = ax3.plot(
            t,
            X[0, :, 1],
            color="firebrick",
            linewidth=3.0,
        )
    if Xhat is not None:
        for xhat in Xhat:
            ax3.plot(
                t,
                xhat[0, :, 1],
                color="royalblue",
                linewidth=3.0,
            )
        if Xhat.shape[0] > 1:
            ax3.plot(
                t,
                X[0, :, 1],
                color="firebrick",
                linewidth=5.0,
            )
    ax3.set_xlabel(
        "time",
        fontsize=10,
    )
    ax3.set_ylabel(
        "State $x_2$",
        fontsize=10,
    )

    if return_fig:
        return fig, ax1, h3, h4, h5
    else:
        plt.show()


def plot_vdp_animation(
    t: np.ndarray,
    X: np.ndarray,
    ode_rhs: Callable,
):
    fig, ax1, h3, h4, h5 = plot_ode(
        t=t,
        X=X,
        ode_rhs=ode_rhs,
        return_fig=True,
    )

    def animate(i):
        h3.set_data(X[: (i + 1) * 5, 0, 0], X[: (i + 1) * 5, 0, 1])
        h4.set_data(t[: (i + 1) * 5], X[: (i + 1) * 5, 0, 0])
        h5.set_data(t[: (i + 1) * 5], X[: (i + 1) * 5, 0, 1])
        ax1.set_title(
            "State trajectory until t={:.2f}".format(5 * t[i].item()),
            fontsize=17,
        )
        return (
            h3,
            h4,
            h5,
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=100,
        interval=100,
        blit=True,
    )
    plt.close()
    return anim
