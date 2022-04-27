import math
import time
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

# Initial clock
t = 0
Ntask = 100
Ncore = random.randint(10, 20)

# Define QSSO parameters
N = 20                  # population size
Cg = 0.7                # define Cg
Cp = 0.9                # define Cp
# Cw = 0.9              # define Cw

# Initialization QSSO global variables
generation_max = 50    # Define here the maximum number of iterations
iteration = 0
the_best_chrom = 0
generation = 0
popSize = N+1
theta = 0.01 * math.pi


class TaskQueue():
    def __init__(self, Ntask):
        self.min_c, self.max_c = 30, 60
        self.task_queue = np.empty([Ntask, 5])
        self.task_queue[:, 0] = np.arange(1, Ntask + 1, 1)
        self.task_queue[:, 1] = np.random.uniform(0, 0.05 * Ntask, Ntask)
        # Ri = random numbers between [0, 4 * m]
        for task in self.task_queue:
            task[2] = np.random.uniform(task[1], task[1] + 4 * Ncore)
            # task[2] = task[1]
        self.task_queue[:, 3] = np.random.uniform(
            self.min_c, self.max_c, Ntask)
        # Di = random numbers between [Ri + 2*Ci, Ri + 3*Ci]
        for task in self.task_queue:
            task[4] = np.random.uniform(
                task[2] + 3 * task[3], task[2] + 4 * task[3])
        # sorted by Ai
        self.task_queue = self.sort_task(1)
        self.run_task_queue = self.task_queue[:, 0]
        self.ready_queue = np.zeros([0])

    def sort_task(self, target_col):
        '''
        Initialize task_queue with Ai, Ri, Ci and Di for each real-time task i.
        array structure: [index, Ai, Ri, Ci, Di]
        array sorted parameter: [index->0, Ai->1, Ri->2, Ci->3, Di->4]
        '''
        return self.task_queue[np.argsort(self.task_queue[:, target_col])]

    def get_info(self, idx):
        '''
        Return the full task's info of target index.
        '''
        # return self.task_queue[np.isin(tq.task_queue[:, 0],idx)][:, [2, 3, 4]] # Get tasks info only with Ri->2, Ci->3, Di->4
        return self.task_queue[np.isin(self.task_queue[:, 0], idx)]

    def get_ready_queue(self, target_col=4):
        '''
        Get the ready_queue by tasks which Ai <= current time t
        '''
        global t
        dequeue_task = self.get_info(self.run_task_queue)[
            self.get_info(self.run_task_queue)[:, 1] <= t]
        if dequeue_task.size == 0:
            return
        ready_queue = np.concatenate(
            (self.get_info(self.ready_queue), self.get_info(dequeue_task)))
        self.ready_queue = ready_queue[np.argsort(
            ready_queue[:, target_col])][:, 0]
        self.run_task_queue = self.task_queue[self.task_queue[:, 1] > t][:, 0]

    def leave_ready_queue(self, m):
        '''
        Let m task leave ready queue to do QSSO
        '''
        self.ready_queue = self.ready_queue[m:]


class Core():
    def __init__(self):
        self.STi = 0
        self.FTi = 0
        self.STnest = 0
        # Initialize a dispatcher contain full info of task
        self.dispatcher = np.zeros([Ntask, 5])

    def append_task(self, task, target_col=4):  # Sort by col=4 (EDF)
        self.dispatcher[Ntask-1] = task
        self.dispatcher = self.dispatcher[np.argsort(
            self.dispatcher[:, target_col])][::-1]

    def get_dispatcher(self):
        '''
        get_dispatcher which is sorted by EDF
        '''
        return self.dispatcher[self.dispatcher[:, 0] != 0][::-1]

    def simulate(self, task, cpv, i):
        # copy the original core
        # append target task from chromosome and sort by EDF
        self.append_task(task)
        self.fittness_eval(cpv, i)

    def schedule_gbest(self, task):
        self.append_task(task)
        self.FTi = 0
        self.STi = 0

        if self.get_dispatcher().shape[0] == 0:
            print("Nothing in dispatcher")
            return
        for task in self.get_dispatcher():
            # [index->0, Ai->1, Ri->2, Ci->3, Di->4]
            self.FTi = self.STi + task[3]
            self.STi = self.FTi

    def fittness_eval(self, cpv, i):

        if self.get_dispatcher().shape[0] == 0:
            print("Nothing in dispatcher")
            return

        for task in self.get_dispatcher():
            # [index->0, Ai->1, Ri->2, Ci->3, Di->4]
            print("task: ", task)
            FTi = self.STi + task[3]
            print("Before Scheh STi: ", self.STi)
            print("FTi: ", FTi)
            print("Ri: ", task[2])
            print("Ci: ", task[3])
            print("Di: ", task[4])
            if(task[2] > self.STi):
                print("Ri > STi")
            if self.STi > task[4] - task[3]:
                print("STi > Di - Ci")
            if task[2] + task[3] > FTi:
                print("Ri + Ci > FTi")
            if FTi > task[4]:
                print("FTi > Di")
            print()

            if task[2] <= self.STi and self.STi <= task[4] - task[3] and task[2] + task[3] <= FTi and FTi <= task[4]:
                print("Scheduled!")
                cpv.fitness[i] += 1
                print(f"if After schedule STi: {self.STi + task[3]}")
                return
            else:
                print("Fail to schedule")


class QuantumPop():
    def __init__(self, m):
        # Quantum population
        global popSize
        global N
        self.popSize = popSize
        self.Nbit = math.ceil(math.log(m, 2))
        self.Nvar = m + 1
        self.Genome = self.Nbit*m
        self.top_bottom = 3
        self.genomeLength = self.Genome+1
        # qpv: quantum chromosome (or population vector, QPV)
        self.qpv = np.empty([self.popSize, self.genomeLength, self.top_bottom])
        self.qpv_sqr = np.empty(
            [self.popSize, self.genomeLength, self.top_bottom])
        self.nqpv = np.empty(
            [self.popSize, self.genomeLength, self.top_bottom])
        self.QuBitZero = np.array([[1], [0]])
        self.QuBitOne = np.array([[0], [1]])
        self.theta = 0.01 * math.pi
        self.the_best_chrom = 0
        self.AlphaBeta = np.empty([self.top_bottom])

    def Init_population(self):
        # Hadamard gate
        r2 = math.sqrt(2.0)
        h = np.array([[1/r2, 1/r2], [1/r2, -1/r2]])

        # Rotation Q-gate
        theta = 0
        rot = np.empty([2, 2])

        # Initial population array (individual x chromosome)
        i = 0
        j = 1
        for i in range(self.popSize):
            for j in range(1, self.genomeLength):
                theta = np.random.uniform(0, 1)*90
                theta = math.radians(theta)
                rot[0, 0] = math.cos(theta)
                rot[0, 1] = -math.sin(theta)
                rot[1, 0] = math.sin(theta)
                rot[1, 1] = math.cos(theta)
                self.AlphaBeta[0] = (h[0][0]*self.QuBitZero[0]) + \
                    (h[0][1]*self.QuBitZero[1])
                self.AlphaBeta[1] = (h[1][0]*self.QuBitZero[0]) + \
                    (h[1][1]*self.QuBitZero[1])

                # Quantum state alpha_beta
                self.qpv[i, j, 0] = np.around(self.AlphaBeta[0], 4)
                self.qpv[i, j, 1] = np.around(self.AlphaBeta[1], 4)

                # alpha squared
                self.qpv_sqr[i, j, 0] = np.around(pow(self.AlphaBeta[0], 2), 2)
                # beta squared
                self.qpv_sqr[i, j, 1] = np.around(pow(self.AlphaBeta[1], 2), 2)

    def Show_population(self):
        '''
        Print the initialized population status:
        Each chromosome contain 4 bits.
        '''
        i = 0
        j = 1
        # i from 0(gbest_idx) to population size
        for i in range(self.popSize):
            if i == 0:
                print()
                print(f"qpv_sqr gbest :")
            else:
                print()
                print(f"qpv_sqr {i} :")
            for j in range(1, self.genomeLength):
                print(self.qpv_sqr[i, j, 0], end="")
                print("||", end="")
            print()
            for j in range(1, self.genomeLength):
                print(self.qpv_sqr[i, j, 1], end="")
                print("||", end="")
        print()

    def get_rot(self, delta_theta):
        rot = np.empty([2, 2])
        rot[0, 0] = math.cos(delta_theta)
        rot[0, 1] = -math.sin(delta_theta)
        rot[1, 0] = math.sin(delta_theta)
        rot[1, 1] = math.cos(delta_theta)
        return rot

    def get_theta(self, f_x, f_b, xi, bi, alpha, beta):
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

    def rotation(self, cpv):
        rot = np.empty([2, 2])
        # Lookup table of the rotation angle
        for i in range(1, self.popSize):
            for j in range(1, self.genomeLength):
                g_alpha = self.qpv[0, j, 0]
                g_beta = self.qpv[0, j, 1]
                p_alpha = self.qpv[i, j, 0]
                p_beta = self.qpv[i, j, 1]
                delta_theta = 0
                # if 𝑓(𝑥) > 𝑓(gbest): True or False
                delta_theta = self.get_theta(
                    cpv.fitness[i], cpv.gbest_fitness, cpv.chromosome[i, j], cpv.gbest_chrom[j], g_alpha, g_beta)
                rot = self.get_rot(delta_theta)
                self.nqpv[0, j, 0] = (
                    rot[0, 0]*self.qpv[0, j, 0]) + (rot[0, 1]*self.qpv[0, j, 1])
                self.nqpv[0, j, 1] = (
                    rot[1, 0]*self.qpv[0, j, 0]) + (rot[1, 1]*self.qpv[0, j, 1])
                self.qpv[0, j, 0] = self.nqpv[0, j, 0]
                self.qpv[0, j, 1] = self.nqpv[0, j, 1]
                # update alpha, beta squared
                self.qpv_sqr[0, j, 0] = np.around(pow(self.qpv[0, j, 0], 2), 5)
                self.qpv_sqr[0, j, 1] = np.around(pow(self.qpv[0, j, 1], 2), 5)
                # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

                #   -------pbest-----------------

                # if 𝑓(𝑥) > 𝑓(pbest): True or False
                delta_theta = self.get_theta(
                    cpv.fitness[i], cpv.pbest_fitness[i], cpv.chromosome[i, j], cpv.pbest_chrom[i, j], p_alpha, p_beta)
                rot = self.get_rot(delta_theta)
                self.nqpv[i, j, 0] = (
                    rot[0, 0]*self.qpv[i, j, 0]) + (rot[0, 1]*self.qpv[i, j, 1])
                self.nqpv[i, j, 1] = (
                    rot[1, 0]*self.qpv[i, j, 0]) + (rot[1, 1]*self.qpv[i, j, 1])
                self.qpv[i, j, 0] = self.nqpv[i, j, 0]
                self.qpv[i, j, 1] = self.nqpv[i, j, 1]
                # update alpha, beta squared
                self.qpv_sqr[i, j, 0] = np.around(pow(self.qpv[i, j, 0], 2), 5)
                self.qpv_sqr[i, j, 1] = np.around(pow(self.qpv[i, j, 1], 2), 5)
                # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:


class ClassicalPop():
    def __init__(self, m):
        # Classical population
        global popSize
        global N
        self.popSize = popSize
        self.Nbit = math.ceil(math.log(m, 2))
        self.Nvar = m + 1
        self.Genome = self.Nbit*m
        self.top_bottom = 3
        self.genomeLength = self.Genome+1
        self.fitness = np.zeros([self.popSize])
        self.the_best_chrom = 0
        self.chromosome = np.empty(
            [self.popSize, self.genomeLength], dtype=np.int)
        self.gbest_chrom = np.empty([self.genomeLength], dtype=np.int)
        self.gbest_fitness = 0
        self.pbest_chrom = np.empty(
            [self.popSize, self.genomeLength], dtype=np.int)
        self.pbest_fitness = np.empty([self.popSize])
        # Record best chromosome of current iteration
        self.best_chrom = np.empty([generation_max], dtype=np.int)
        self.best_pchrom = np.empty([generation_max], dtype=np.int)
        self.schedule = np.empty([self.popSize, self.Nvar], dtype=np.int)
        self.fitness_max = 0

    def SSO_Measure(self, QuantumPop):
        for i in range(1, self.popSize):
            # print(f"chromosome {i}:", end=" ")
            for j in range(1, self.genomeLength):
                SSO_rnd = random.random()
                measure_rnd = random.random()
                if (SSO_rnd < Cg):
                    if measure_rnd <= QuantumPop.qpv_sqr[0, j, 0]:
                        self.chromosome[i, j] = 0
                    else:
                        self.chromosome[i, j] = 1
                elif (SSO_rnd < Cp):
                    if measure_rnd <= QuantumPop.qpv_sqr[i, j, 0]:
                        self.chromosome[i, j] = 0
                    else:
                        self.chromosome[i, j] = 1
                else:
                    if measure_rnd <= 0.5:
                        self.chromosome[i, j] = 0
                    else:
                        self.chromosome[i, j] = 1
                # print(chromosome[i, j], end="")
            # print()
        # print()

    def get_schedule(self):
        '''
        Input binary scheduling and output permutation scheduling list
        ref of permutation trimming: A Hybrid Quantum-Inspired Genetic Algorithm for Flow
        ref of random key: Genetic Algorithms and Random Keys for Sequencing and Optimization Shop Scheduling
        '''
        for i in range(1, popSize):
            bin_sch_str = "".join(str(bin_num)
                                  for bin_num in self.chromosome[i, 1:])
            self.schedule[i, 1:] = np.argsort(np.array(
                [int(bin_sch_str[i:i + self.Nbit], 2) for i in range(0, len(bin_sch_str), self.Nbit)])) + 1
    # 5 cores, 5 tasks [4, 5, 6, 7, 6]
    # r = get_schedule("100101110111110", 3, 5)
    # array([1, 2, 3, 5, 4])

    def show_schedule(self):
        # cpv.schedule[0, :] not def
        # cpv.schedule[:, 0] not def
        for i in range(1, popSize):
            print(f"cpv.schedule_{i}: {cpv.schedule[i, 1:]}")


########################################################
#                                                      #
# MAIN PROGRAM                                         #
#                                                      #
########################################################
# Start Count the current time
start_t = time.time()
success_tasks = 0
unscheduled_tasks = 100
# Initialize task queue for Ntask with fully information
TaskQ = TaskQueue(100)
core = [Core() for i in range(Ncore + 1)]  # core[0] not define

# Initialized Schedule: First assignment
t = int(time.time() - start_t + 1)      # get current time
print("t: ", t)
TaskQ.get_ready_queue()
m = min(TaskQ.ready_queue.size, Ncore)  # Get the chromosome size
Nvar = m + 1
# Initialize Quantum population
qpv = QuantumPop(m)
qpv.Init_population()
# qpv.Show_population()

# Initialize Classical population
cpv = ClassicalPop(m)
cpv.SSO_Measure(qpv)
# value = task order in ready queue, index = which processor it should be scheduled
cpv.get_schedule()

# Schedule first task to each core
for j in range(1, Nvar):
    task = TaskQ.get_info(TaskQ.ready_queue[cpv.schedule[1, j] - 1])[0]
    core[j].append_task(task)
    core[j].STi += task[3]
    print(f"core[{j}].STi_first task: {core[j].STi}")
# Set initialized gbest, pbest
for i in range(1, popSize):
    if cpv.fitness[i] >= cpv.fitness_max:
        cpv.fitness_max = cpv.fitness[i]
        cpv.the_best_chrom = i
    if cpv.fitness[i] >= cpv.gbest_fitness:
        cpv.gbest_fitness = cpv.fitness[i]
        cpv.gbest_chrom = cpv.chromosome[i]
    if cpv.fitness[i] >= cpv.pbest_fitness[i]:
        cpv.pbest_fitness[i] = cpv.fitness[i]
        cpv.pbest_chrom[i] = cpv.chromosome[i]
# Run to this code, m tasks would leave ready queue.
TaskQ.leave_ready_queue(m)
unscheduled_tasks -= m
time.sleep(1)  # 這邊sleep 之後寫好SSO就拿掉

# Dynamic Assign tasks into Ready Queue
while (TaskQ.run_task_queue.size != 0 or TaskQ.ready_queue.size != 0):

    t = int(time.time() - start_t + 1)
    print("t: ", t)
    if TaskQ.run_task_queue.size != 0:
        TaskQ.get_ready_queue()
    m = min(TaskQ.ready_queue.size, Ncore)
    print("m: ", m)
    Nvar = m + 1
    # Initialize Quantum population for current generation
    qpv = QuantumPop(m)
    qpv.Init_population()
    # qpv.Show_population()

    # Initialize Classical population for current generation
    cpv = ClassicalPop(m)
    cpv.SSO_Measure(qpv)
    cpv.get_schedule()

    print()
    print(" ============== Initialization =========================== ")
    print()

    # Initialized first iteration of current tasks' set
    for i in range(1, popSize):
        print(f"-------chrom {i}-------")
        core_simulate = copy.deepcopy(core)
        for j in range(1, Nvar):
            print(f"----core_[{j}]----")
            core_simulate[j].simulate(TaskQ.get_info(
                TaskQ.ready_queue[cpv.schedule[i, j] - 1]), cpv, i)
        # Set initialized gbest, pbest
        if cpv.fitness[i] >= cpv.fitness_max:
            cpv.fitness_max = cpv.fitness[i]
            cpv.the_best_chrom = i
        if cpv.fitness[i] >= cpv.gbest_fitness:
            cpv.gbest_fitness = cpv.fitness[i]
            cpv.gbest_chrom = cpv.chromosome[i]
        if cpv.fitness[i] >= cpv.pbest_fitness[i]:
            cpv.pbest_fitness[i] = cpv.fitness[i]
            cpv.pbest_chrom[i] = cpv.chromosome[i]

    # Iteration
    generation = 0
    while generation < generation_max-1:
        print()
        print("============== GENERATION: ", generation +
              1, " =========================== ")
        print()

        # Apply rotation gate
        qpv.rotation(cpv)
        cpv = ClassicalPop(m)
        cpv.SSO_Measure(qpv)
        cpv.get_schedule()

        # Do the iteration
        for i in range(1, popSize):
            print(f"========== chrom {i} ==========")
            core_simulate = copy.deepcopy(core)
            for j in range(1, Nvar):
                print(f"=== core_[{j}] ===")
                core_simulate[j].simulate(TaskQ.get_info(
                    TaskQ.ready_queue[cpv.schedule[i, j] - 1]), cpv, i)
            # Set initialized gbest, pbest
            if cpv.fitness[i] >= cpv.fitness_max:
                cpv.fitness_max = cpv.fitness[i]
                cpv.the_best_chrom = i
            if cpv.fitness[i] >= cpv.gbest_fitness:
                cpv.gbest_fitness = cpv.fitness[i]
                cpv.gbest_chrom = cpv.chromosome[i]
            if cpv.fitness[i] >= cpv.pbest_fitness[i]:
                cpv.pbest_fitness[i] = cpv.fitness[i]
                cpv.pbest_chrom[i] = cpv.chromosome[i]
            print(f"Chrom_{i} fitness: {cpv.fitness[i]}")
        cpv.best_chrom[generation] = cpv.the_best_chrom
        gbest_schedule = cpv.schedule[cpv.the_best_chrom]
        print(f"best_chrom[{generation}] = Chrom_{cpv.the_best_chrom}")
        print("gbest_schedule: ", gbest_schedule[1:])

        # Increment the generation
        generation += 1
    # Schedule the gbest solution into each official core
    for j in range(1, Nvar):
        core[j].schedule_gbest(TaskQ.get_info(
            TaskQ.ready_queue[cpv.schedule[cpv.the_best_chrom][j] - 1]))
        print(f"core[{j}].dispatcher: {core[j].get_dispatcher()}")
        print(f"core[{j}].STi_gen {generation}: {core[j].STi}")

    print(f"gbest_fitness: ", cpv.gbest_fitness)
    success_tasks += cpv.gbest_fitness
    # Run to this code, m tasks would leave ready queue.
    TaskQ.leave_ready_queue(m)
    unscheduled_tasks -= m

end_t = time.time()
time_cost = int(end_t - start_t)
print(f"TaskQ.run_task_queue.size: {TaskQ.run_task_queue.size}")
print(f"TaskQ.ready_queue.size: {TaskQ.ready_queue.size}")
print(f"unscheduled_tasks: {unscheduled_tasks}")
print(f"run_task_queue: {TaskQ.run_task_queue}")
print(f"Ready_queue: {TaskQ.ready_queue}")
print(f"time_cost: {time_cost}")
print("SUCCESSFUL NUMBER: ", success_tasks)
print("END OF SCHEDULING")
