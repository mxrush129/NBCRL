import numpy as np
import sympy as sp
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle
from utils.Config import CegisConfig
from RL.Env import Zones, Example
import mpl_toolkits.mplot3d.art3d as art3d


class Draw:
    def __init__(self, example: Example, b1, b2=None):
        self.ex = example
        self.b1 = b1
        self.b2 = b2

    def draw(self):
        fig = plt.figure()
        ax = plt.gca()

        ax.add_patch(self.draw_zone(self.ex.l1, 'grey', 'local_1'))
        ax.add_patch(self.draw_zone(self.ex.l2, 'blue', 'local_2'))
        ax.add_patch(self.draw_zone(self.ex.I, 'g', 'init'))
        ax.add_patch(self.draw_zone(self.ex.U, 'r', 'unsafe'))
        ax.add_patch(self.draw_zone(self.ex.g1, 'bisque', 'guard_1'))
        ax.add_patch(self.draw_zone(self.ex.g2, 'orange', 'guard_2'))

        l1, l2 = self.ex.l1, self.ex.l2

        self.plot_vector_field(l1, self.ex.f1, 'slategrey')
        self.plot_vector_field(l2, self.ex.f2, 'cornflowerblue')

        self.plot_barrier(l1, self.b1, 'orchid')
        self.plot_barrier(l2, self.b2, 'purple')

        plt.xlim(min(l1.low[0], l2.low[0]) - 1, max(l1.up[0], l2.up[0]) + 1)
        plt.ylim(min(l1.low[1], l2.low[1]) - 1, max(l1.up[1], l2.up[1]) + 1)
        ax.set_aspect(1)
        plt.legend(loc='lower left')
        plt.savefig(f'picture/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def draw_continuous(self):
        fig = plt.figure()
        ax = plt.gca()

        ax.add_patch(self.draw_zone(self.ex.D_zones, 'black', 'local_1'))
        ax.add_patch(self.draw_zone(self.ex.I_zones, 'g', 'init'))
        ax.add_patch(self.draw_zone(self.ex.U_zones, 'r', 'unsafe'))

        l1 = self.ex.D_zones

        self.plot_vector_field(l1, self.ex.f)

        self.plot_barrier(l1, self.b1, 'b')

        plt.xlim(l1.low[0] - 1, l1.up[0] + 1)
        plt.ylim(l1.low[1] - 1, l1.up[1] + 1)
        ax.set_aspect(1)
        plt.legend()
        plt.savefig(f'picture/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def draw_3d_continuous(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        self.plot_barrier_3d(ax, self.ex.D_zones, self.b1, 'gold')

        init = self.draw_zone(self.ex.I_zones, color='g', label='init')
        ax.add_patch(init)
        art3d.pathpatch_2d_to_3d(init, z=0, zdir="z")

        unsafe = self.draw_zone(self.ex.U_zones, color='r', label='unsafe')
        ax.add_patch(unsafe)
        art3d.pathpatch_2d_to_3d(unsafe, z=0, zdir="z")
        # ax.view_init(elev=-164, azim=165)
        plt.savefig(f'picture/{self.ex.name}_3d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_barrier_3d(self, ax, zone, b, color):
        low, up = zone.low, zone.up
        x = np.linspace(low[0], up[0], 100)
        y = np.linspace(low[1], up[1], 100)
        X, Y = np.meshgrid(x, y)
        s_x = sp.symbols(['x1', 'x2'])
        lambda_b = sp.lambdify(s_x, b, 'numpy')
        plot_b = lambda_b(X, Y)

        ax.plot_surface(X, Y, plot_b, rstride=5, cstride=5, alpha=0.5, cmap='cool')

    def plot_barrier(self, zone, hx, color):
        low, up = zone.low, zone.up
        x = np.linspace(low[0], up[0], 100)
        y = np.linspace(low[1], up[1], 100)

        X, Y = np.meshgrid(x, y)

        s_x = sp.symbols(['x1', 'x2'])
        fun_hx = sp.lambdify(s_x, hx, 'numpy')
        value = fun_hx(X, Y)
        plt.contour(X, Y, value, 0, alpha=0.8, colors=color)

    def plot_vector_field(self, zone: Zones, f, color='grey'):
        low, up = zone.low, zone.up
        xv = np.linspace(low[0], up[0], 10)
        yv = np.linspace(low[1], up[1], 10)
        Xd, Yd = np.meshgrid(xv, yv)

        DX, DY = f[0](Xd, Yd), f[1](Xd, Yd)
        DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
        DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)

        plt.streamplot(Xd, Yd, DX, DY, linewidth=0.3,
                       density=0.8, arrowstyle='-|>', arrowsize=1, color=color)

    def draw_zone(self, zone: Zones, color, label, fill=False):
        if zone.shape == 'ball':
            circle = Circle(zone.center, np.sqrt(zone.r), color=color, label=label, fill=fill, linewidth=1.5)
            return circle
        else:
            w = zone.up[0] - zone.low[0]
            h = zone.up[1] - zone.low[1]
            box = Rectangle(zone.low, w, h, color=color, label=label, fill=fill, linewidth=1.5)
            return box


if __name__ == '__main__':
    pass
