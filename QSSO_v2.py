import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
# Define here the population size (number of solutions)
N = 30
Nbit = 1
Nvar = 10
# Define here the chromosome length (How many bits in each chromosome)
Genome = Nbit*Nvar
generation_max = 20    # Define here the maximum number of iterations

# Define SSO parameters
Cg = 0.7                # define Cg
Cp = 0.9                # define Cp
# Cw = 0.9              # define Cw

#########################################################
# VARIABLES ALGORITHM                                   #
#########################################################

popSize = N+1
gbest_idx = 0
genomeLength = Genome+1
top_bottom = 3
QuBitZero = np.array([[1], [0]])
QuBitOne = np.array([[0], [1]])
AlphaBeta = np.empty([top_bottom])
fitness = np.zeros([popSize])
# qpv: quantum chromosome (or population vector, QPV)
qpv = np.empty([popSize, genomeLength, top_bottom])
qpv_sqr = np.empty([popSize, genomeLength, top_bottom])
nqpv = np.empty([popSize, genomeLength, top_bottom])
# chromosome: classical chromosome
chromosome = np.empty([popSize, genomeLength], dtype=np.int)
gbest_chrom = np.empty([genomeLength], dtype=np.int)
gbest_fitness = 0
pbest_chrom = np.empty([popSize, genomeLength], dtype=np.int)
pbest_fitness = np.empty([popSize])
# Record best chromosome of current iteration
best_chrom = np.empty([generation_max], dtype=np.int)
best_pchrom = np.empty([generation_max], dtype=np.int)

# Initialization global variables
theta = 0.01 * math.pi
iteration = 0
the_best_chrom = 0
generation = 0

#########################################################
# QUANTUM POPULATION INITIALIZATION                     #
#########################################################


def Init_population():
    # Hadamard gate
    r2 = math.sqrt(2.0)
    h = np.array([[1/r2, 1/r2], [1/r2, -1/r2]])

    # Rotation Q-gate
    theta = 0
    rot = np.empty([2, 2])

    # Initial population array (individual x chromosome)
    i = 0
    j = 1
    for i in range(popSize):
        for j in range(1, genomeLength):
            theta = np.random.uniform(0, 1)*90
            theta = math.radians(theta)
            rot[0, 0] = math.cos(theta)
            rot[0, 1] = -math.sin(theta)
            rot[1, 0] = math.sin(theta)
            rot[1, 1] = math.cos(theta)
            AlphaBeta[0] = (h[0][0]*QuBitZero[0])+(h[0][1]*QuBitZero[1])
            AlphaBeta[1] = (h[1][0]*QuBitZero[0])+(h[1][1]*QuBitZero[1])

            # Quantum state alpha_beta
            qpv[i, j, 0] = np.around(AlphaBeta[0], 4)
            qpv[i, j, 1] = np.around(AlphaBeta[1], 4)

            # alpha squared
            qpv_sqr[i, j, 0] = np.around(pow(AlphaBeta[0], 2), 2)
            # beta squared
            qpv_sqr[i, j, 1] = np.around(pow(AlphaBeta[1], 2), 2)

#########################################################
# SHOW QUANTUM POPULATION                               #
#########################################################


def Show_population():
    '''
    Print the initialized population status:
    Each chromosome contain 4 bits.
    '''
    i = 0
    j = 1
    # i from 0(gbest_idx) to population size
    for i in range(popSize):
        if i == 0:
            print()
            print(f"qpv_sqr gbest :")
        else:
            print()
            print(f"qpv_sqr {i} :")
        for j in range(1, genomeLength):
            print(qpv_sqr[i, j, 0], end="")
            print("||", end="")
        print()
        for j in range(1, genomeLength):
            print(qpv_sqr[i, j, 1], end="")
            print("||", end="")
    print()

#########################################################
# SSO Measure                                         #
#########################################################
# p_alpha: probability of finding qubit in alpha state


def SSO_Measure():
    for i in range(1, popSize):
        # print(f"chromosome {i}:", end=" ")
        for j in range(1, genomeLength):
            SSO_rnd = random.random()
            measure_rnd = random.random()
            if (SSO_rnd < Cg):
                if measure_rnd <= qpv_sqr[0, j, 0]:
                    chromosome[i, j] = 0
                else:
                    chromosome[i, j] = 1
            elif (SSO_rnd < Cp):
                if measure_rnd <= qpv_sqr[i, j, 0]:
                    chromosome[i, j] = 0
                else:
                    chromosome[i, j] = 1
            else:
                if measure_rnd <= 0.5:
                    chromosome[i, j] = 0
                else:
                    chromosome[i, j] = 1
            # print(chromosome[i, j], end="")
        # print()
    # print()


#########################################################
# FITNESS EVALUATION                                    #
#########################################################

def Fitness_evaluation(generation):
    global gbest_chrom
    global pbest_chrom
    global gbest_fitness
    global pbest_fitness
    i = 1
    j = 1
    fitness_total = 0
    sum_sqr = 0
    fitness_average = 0
    variance = 0
    # for i in range(1, popSize):
    #     fitness[i] = 0

#########################################################
# Define your problem in this section. For instance:    #
#                                                       #
# Let f(x)=abs(x-5/2+sin(x)) be a function that takes   #
# values in the range 0<=x<=15. Within this range f(x)  #
# has a maximum value at x=11 (binary is equal to 1011) #
#########################################################
    for i in range(1, popSize):
        # x used to accumulate the result translate from binary to decimal value
        x = 0
        for j in range(1, genomeLength):
            # translate from binary to decimal value
            x = x+chromosome[i, j]*pow(2, genomeLength-j-1)
        # replaces the value of x in the function f(x)
        # print(f"chromosome[{i}][{2}]: {chromosome[i]}")
        y = np.fabs((x-5)/(2+np.sin(x)))
        # the fitness value is calculated below:
        # (Note that in this example is multiplied
        # by a scale value, e.g. 100)
        fitness[i] = y*100
#########################################################

        # print("fitness = ", i, " ", fitness[i])
        fitness_total = fitness_total+fitness[i]
    # print()
    fitness_average = fitness_total/N
    i = 1
    while i <= N:
        sum_sqr = sum_sqr+pow(fitness[i]-fitness_average, 2)
        i = i+1
    variance = sum_sqr/N
    if variance <= 1.0e-4:
        variance = 0.0

    # Best chromosome update
    the_best_chrom = 0
    fitness_max = fitness[1]
    for i in range(1, popSize):
        if fitness[i] >= fitness_max:
            fitness_max = fitness[i]
            the_best_chrom = i
        if fitness[i] >= gbest_fitness:
            gbest_fitness = fitness[i]
            gbest_chrom = chromosome[i]
        if fitness[i] >= pbest_fitness[i]:
            pbest_fitness[i] = fitness[i]
            pbest_chrom[i] = chromosome[i]
    print(f"best_chrom[{generation}] = {the_best_chrom}")
    best_chrom[generation] = the_best_chrom

    # Statistical output
    # f = open("QSSO_output.dat", "a")
    # f.write(str(generation)+" "+str(fitness_average)+"\n")
    # f.write(" \n")
    # f.close()
    # print("Population size = ", popSize - 1)
    # print("mean fitness = ", fitness_average)
    # print("variance = ", variance, " Std. deviation = ", math.sqrt(variance))
    print("Global best fitness = ", gbest_fitness)
    print("Global best chromosome = ", gbest_chrom[1:])
    # print("fitness sum = ", fitness_total)


#########################################################
# Rotation Matrix                                       #
#########################################################

def get_rot(delta_theta):
    rot = np.empty([2, 2])
    rot[0, 0] = math.cos(delta_theta)
    rot[0, 1] = -math.sin(delta_theta)
    rot[1, 0] = math.sin(delta_theta)
    rot[1, 1] = math.cos(delta_theta)
    return rot

#########################################################
# Rotation angle Lookup                                 #
#########################################################


def get_theta(f_x, f_b, xi, bi, alpha, beta):
    global theta
    delta_theta = 0
    cond = (f_x > f_b)

    # if 𝑓(𝑥) > 𝑓(gbest): FALSE
    if cond == False:
        # if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:
        if xi == 0 and bi == 1:
            # Define the rotation angle: delta_theta (e.g. 0.0785398163)
            if (alpha * beta > 0):
                delta_theta = theta
            elif (alpha * beta < 0):
                delta_theta = -theta
            elif (beta == 0):
                delta_theta = -theta
        if xi == 1 and bi == 0:
            # Define the rotation angle: delta_theta (e.g. -0.0785398163)
            if (alpha * beta > 0):
                delta_theta = -theta
            elif (alpha * beta < 0):
                delta_theta = theta
            elif (alpha == 0):
                delta_theta = -theta
    elif cond == True:
        # if 𝑓(𝑥) > 𝑓(gbest): TRUE
        if xi == 0 and bi == 1:
            # Define the rotation angle: delta_theta (e.g. -0.0785398163)
            if (alpha * beta > 0):
                delta_theta = -theta
            elif (alpha * beta < 0):
                delta_theta = theta
            elif (alpha == 0):
                delta_theta = -theta
        if xi == 1 and bi == 0:
            # Define the rotation angle: delta_theta (e.g. -0.0785398163)
            if (alpha * beta > 0):
                delta_theta = theta
            elif (alpha * beta < 0):
                delta_theta = -theta
            elif (beta == 0):
                delta_theta = -theta
    return delta_theta

#########################################################
# QUANTUM ROTATION GATE                                 #
#########################################################


def rotation():
    rot = np.empty([2, 2])
    # Lookup table of the rotation angle
    for i in range(1, popSize):
        for j in range(1, genomeLength):
            g_alpha = qpv[0, j, 0]
            g_beta = qpv[0, j, 1]
            p_alpha = qpv[i, j, 0]
            p_beta = qpv[i, j, 1]
            delta_theta = 0
            # if 𝑓(𝑥) > 𝑓(gbest): True or False
            delta_theta = get_theta(
                fitness[i], gbest_fitness, chromosome[i, j], gbest_chrom[j], g_alpha, g_beta)
            rot = get_rot(delta_theta)
            nqpv[0, j, 0] = (rot[0, 0]*qpv[0, j, 0]) + (rot[0, 1]*qpv[0, j, 1])
            nqpv[0, j, 1] = (rot[1, 0]*qpv[0, j, 0]) + (rot[1, 1]*qpv[0, j, 1])
            qpv[0, j, 0] = nqpv[0, j, 0]
            qpv[0, j, 1] = nqpv[0, j, 1]
            # update alpha, beta squared
            qpv_sqr[0, j, 0] = np.around(pow(qpv[0, j, 0], 2), 3)
            qpv_sqr[0, j, 1] = np.around(pow(qpv[0, j, 1], 2), 3)
            # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

            #   -------pbest-----------------

            # if 𝑓(𝑥) > 𝑓(pbest): True or False
            delta_theta = get_theta(
                fitness[i], pbest_fitness[i], chromosome[i, j], pbest_chrom[i, j], p_alpha, p_beta)
            rot = get_rot(delta_theta)
            nqpv[i, j, 0] = (rot[0, 0]*qpv[i, j, 0]) + (rot[0, 1]*qpv[i, j, 1])
            nqpv[i, j, 1] = (rot[1, 0]*qpv[i, j, 0]) + (rot[1, 1]*qpv[i, j, 1])
            qpv[i, j, 0] = nqpv[i, j, 0]
            qpv[i, j, 1] = nqpv[i, j, 1]
            # update alpha, beta squared
            qpv_sqr[i, j, 0] = np.around(pow(qpv[i, j, 0], 2), 3)
            qpv_sqr[i, j, 1] = np.around(pow(qpv[i, j, 1], 2), 3)
            # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

#########################################################
# X-PAULI QUANTUM MUTATION GATE                         #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit


def mutation(pop_mutation_rate, mutation_rate):

    for i in range(1, popSize):
        up = np.random.random_integers(100)
        up = up/100
        if up <= pop_mutation_rate:
            for j in range(1, genomeLength):
                um = np.random.random_integers(100)
                um = um/100
                if um <= mutation_rate:
                    nqpv[i, j, 0] = qpv[i, j, 1]
                    nqpv[i, j, 1] = qpv[i, j, 0]
                else:
                    nqpv[i, j, 0] = qpv[i, j, 0]
                    nqpv[i, j, 1] = qpv[i, j, 1]
        else:
            for j in range(1, genomeLength):
                nqpv[i, j, 0] = qpv[i, j, 0]
                nqpv[i, j, 1] = qpv[i, j, 1]
    for i in range(1, popSize):
        for j in range(1, genomeLength):
            qpv[i, j, 0] = nqpv[i, j, 0]
            qpv[i, j, 1] = nqpv[i, j, 1]

#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
# Read the Docs in http://matplotlib.org/1.4.1/index.html


def plot_Output():
    data = np.loadtxt('QSSO_output.dat')
    # plot the first column as x, and second column as y
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Fitness average')
    plt.xlim(0.0, 550.0)
    plt.show()

########################################################
#                                                      #
# MAIN PROGRAM                                         #
#                                                      #
########################################################


def Q_GA():
    generation = 0
    print("============== GENERATION: ", generation,
          " =========================== ")
    print()
    Init_population()
    # Show_population()
    SSO_Measure()
    Fitness_evaluation(generation)
    while (generation < generation_max-1):
        print(
            f"The best of generation[{generation}]: {best_chrom[generation]}")
        print()
        print("============== GENERATION: ", generation +
              1, " =========================== ")
        print()
        rotation()
        # mutation(0.01, 0.001)
        # Show_population()
        generation = generation+1
        SSO_Measure()
        Fitness_evaluation(generation)


print("""QUANTUM SSO""")
input("Press Enter to continue...")
time_start = time.time()  # 開始計時
Q_GA()
time_end = time.time()
time_c = time_end - time_start  # 執行所花時間
print('time cost', time_c, 's')
# plot_Output()
