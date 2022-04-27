import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from turtle import color

def plot_Output_JointPlot():
    # for QGA
    data_QGA = np.loadtxt('QGA_output.dat')
    # plot the first column as x, and second column as y
    x = data_QGA[:, 0]
    y = data_QGA[:, 1]
    plt.plot(x, y, color="blueviolet", linewidth=2)

    data_QSSO = np.loadtxt('QSSO_output.dat')
    data_pbest = np.loadtxt('QSSO_pbest.dat')
    # plot the first column as x, and second column as y
    x = data_QSSO[:, 0]
    y = data_QSSO[:, 1]
    pbest_list = []

    color = cm.rainbow(np.linspace(0, 1, 50))

    for i in range(1, 31):
        pbest_list.append(data_pbest[data_pbest[:, 1] == i, 2])

    plt.plot(x, y, linewidth=2, color='r')
    for i in range(30):
        plt.plot(pbest_list[i], color="lightgray", linewidth=0.3)
    plt.xlabel('Generation')
    plt.ylabel('Global best Fitness')
    plt.xlim(0.0, 300.0)

    plt.show()

plot_Output_JointPlot()
