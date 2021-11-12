import numpy as np
import random as rand
import matplotlib.pyplot as plt

playerName = "myAgent"
nPercepts = 75  #This is the number of percepts
nActions = 5    #This is the number of actions
all_fitness = list()
games = 1000

# Training schedule
trainingSchedule = [('random', games)]

'''
This is the class for creating a creature, evaluating the old generation and
creating a new generation using selection, crossover and mutation, and converting
percepts + chromosomes into actions.
'''
class MyCreature:

    # Constructor for the MyCreature class where a 2D array is created in the variable chromosome.
    # The array is of size 75 composed of arrays of the size of 5 with random ints between 0 and 1.
    # 75 * 5 = 375 therefore there are 375 random genes in total.
    # A variable fitness is also created to hold the creatures fitness.
    def __init__(self):
        self.chromosome = np.random.rand(nPercepts, nActions)
        self.fitness = 0

    # AgentFunction method takes percepts as a parameter (3D array) then flattens
    # them into a 1D array. This 1D array of percepts is then multiplied with the chromosomes
    # using the numpy dot function to get the total final array of size 5.
    def AgentFunction(self, percepts):

        actions = np.dot(percepts.flatten(), self.chromosome)
        return actions


'''
newGeneration function takes in the old population from the previous round and:
    - Works out the fitness of each creature (34 in total).
    - Find the fittest agents and puts into the new_population (elitism).
    - Finds 2 parents by using tournament selection.
    - Crosses over the 2 parents chromosomes to create a child.
    - 1 in 10 chance of the child's chromosome to be mutated.
This is done creatures minus the number of elite agents time (so we always have 34 creatures)
in our population.
'''
def newGeneration(old_population):

    # Find the length of the old population
    N = len(old_population)

    # Fitness for all agents
    fitness = []

    # This loop iterates over your agents in the old population and finds the fitness of
    # each creature using the fitness function f. The output of the fitness function for
    # each creature is put into a list and the creature itself.
    all_creatures = []
    for n, creature in enumerate(old_population):
        # creature.alive - boolean, true if creature is alive at the end of the game
        # creature.turn - turn that the creature lived to (last turn if creature survived the entire game)
        # creature.size - size of the creature
        # creature.strawb_eats - how many strawberries the creature ate
        # creature.enemy_eats - how much energy creature gained from eating enemies
        # creature.squares_visited - how many different squares the creature visited
        # creature.bounces - how many times the creature bounced

        turn = creature.turn
        alive = creature.alive
        size = creature.size * 2
        s_eats = creature.strawb_eats * 2
        e_eats = creature.enemy_eats * 2
        squares = creature.squares_visited / 10
        bounces = creature.bounces

        f = turn + alive + size + s_eats + e_eats + squares
        fitness.append(f)
        creature.fitness = f
        all_creatures.append(creature)

    # new_population for the creature is created. This will have a mix of the old_population
    # genes with a chance of mutation.
    new_population = list()

    # Elitism. The best agents from the previous round are automatically
    # put into the next round. This is done using the sort function that
    # sorts a list based on the creatures fitness.
    nElite_agents = 4
    elite_agents = []
    all_creatures.sort(key=lambda creature: creature.fitness, reverse=True)
    for c in all_creatures:
        if len(elite_agents) == nElite_agents:
            break
        else:
            elite_agents.append(c)
    for creatures in elite_agents:
        new_population.append(creatures)

    # Pool tournament that selects a random pool from the list of creatures and
    # selects the best 2 as parents based on fitness.
    size_tournament = 7
    for n in range((N-nElite_agents)):
        pool_tournament = rand.sample(old_population, size_tournament)
        pool_tournament.sort(key=lambda creature: creature.fitness, reverse=True)
        p1 = pool_tournament[0]
        p2 = pool_tournament[1]

        # Cross over parents to create a new child. This is done by using uniform crossover.
        # This means that each gene is is selected randomly from one of the corresponding
        # genes of the parent chromosomes by using a coin-flip. If the coin lands on 1
        # the child gets the gene from parent 1 (p1), if the coin lands on 2 the child gets
        # its gene from parent 2.
        p1 = p1.chromosome
        p2 = p2.chromosome
        child_chrome = np.zeros((nPercepts, nActions))
        for i in range(nPercepts):
            for j in range(nActions):
                rand_parent = np.random.randint(1, 3)
                if rand_parent == 1:
                    child_chrome[i][j] = p1[i][j]
                else:
                    child_chrome[i][j] = p2[i][j]

        # Mutation
        for i in range(nPercepts):
            for j in range(nActions):
                mutation_prop = 0.03
                mutated_gene = np.random.random(1)
                random_num = np.random.random(1)
                if random_num < mutation_prop:
                    child_chrome[i][j] += mutated_gene

        # Inserting the new chromosome into the child then putting the new child into the
        # new population.
        new_creature = MyCreature()
        new_creature.chromosome = child_chrome
        # Add the new agent to the new population
        new_population.append(new_creature)

    # Calculating the average fitness of the generation and putting into the plot.
    av = np.mean(fitness)
    avg_fitness = np.mean(fitness)
    all_fitness.append(avg_fitness)
    if len(all_fitness) == games:
        plt.close('all')
        plt.plot(all_fitness)
        plt.ylabel("Average Fitness")
        plt.xlabel("Games/Generations")
        plt.show()
        graph_mean = np.mean(all_fitness)
        print(graph_mean)

    # Formatting the average fitness to 2dp for better readability.
    av = "{:.2f}".format(av)
    print(av)

    return new_population, avg_fitness
