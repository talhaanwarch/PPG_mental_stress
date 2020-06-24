
from deap import creator, base, tools, algorithms
import random
import numpy as np
from deap import tools
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,scale,MaxAbsScaler

class FeatureSelectionGA:

    def __init__(self, model, x, y, cv_split=27, n_pop=15, n_gen=10):
      
        self.model = model
        self.n_features = x.shape[1]
        self.cv_split = cv_split
        self.x = x
        self.y = y
        self.n_pop = n_pop
        self.n_gen = n_gen
        print("The number of Features: {}".format(self.n_features))
        print("The Shape of Training Data is : {} and Target Data is : {}".format(self.x.shape, self.y.shape))

        individual = [1 for i in range(x.shape[1])]
        print("Accuracy For All the features: " + str(self.fitness_test(individual)) + "\n")

        # Applying Genetic Algorithm
        eg = self.evolutionary_algorithm()
        hof=eg[0]
        log=eg[1]
        plt.plot(log.select("gen"),log.select("avg"))
        plt.title('Support Vector Machine Classifier',fontsize=12)
        plt.xlabel('generation',fontsize=12)
        plt.ylabel('accuracy',fontsize=12)
        plt.savefig('SVM.png')
        #plt.imshow()
        accuracy, individual, header = self.bestIndividual(hof)
        print('Best Accuracy: \t' + str(accuracy[0]))
        print('Number of Features in Subset: \t' + str(individual.count(1)))
        print('Feature Subset: ' + str(header)+'\n')
        #print('\n\nKindly Create a New Classifier with the Above Feature Set')




    def fitness_test(self, individual):
        """
        The Function Analyses Provides the average Cross Val Score Using All the Features
        :param individual:
        :return: Average Cross Val Score
        """
        alpha=0.01 #ranges from 0 to 1
        if (individual.count(0) != len(individual)):
            # Fetched the Index of the Individual
            cols = [index for index in range(len(individual)) if individual[index] == 0]

            # Fetching Feature Subset
            X_parsed = self.x.drop(self.x.columns[cols], axis=1)
            X_subset = pd.get_dummies(X_parsed)

            # Applying the Classification Algorithm
            classifier = self.model
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
            cross_v=cross_val_score(pipe, X_subset, self.y, cv=self.cv_split)
            acc=((sum(cross_v) / float(len(cross_v))),0)
            return acc            
            #return (alpha*(1.0 - acc[0]) + (1.0 - alpha)*(1.0 - (X_subset.shape[1])/10),0)#10 is total feature
        else:
            return (0,)

    def evolutionary_algorithm(self):
        """
        Declaring Global Variables for DEAP
        :return:
        """
        # Creating the Individual Using DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Creating ToolBox For The DEAP Framework
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features) #initialization
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) #initialization
        toolbox.register("evaluate", self.fitness_test) #fitness function
        toolbox.register("mate", tools.cxTwoPoint)#crossover
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) #mutation 
        toolbox.register("select", tools.selTournament, tournsize=5) #selection 

        # Initialize Parameters
        pop = toolbox.population(n=self.n_pop)
        hof = tools.HallOfFame(self.n_pop * self.n_gen)#  best individual that ever lived in the population during the evolution.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean) #avg score from fitness function
        stats.register("min", np.min)  #min score from fitness function
        stats.register("max", np.max)  #max score from fitness function

        # Genetic Algorithm
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=self.n_gen, stats=stats, halloffame=hof,verbose=True)

        # Return Fall Of Home
        return hof,log

    def bestIndividual(self, hof):
        """
        Get the best individual
        """
        maxAccurcy = 0.0
        for individual in hof:
            if (individual.fitness.values[0] > maxAccurcy):
                maxAccurcy = individual.fitness.values[0]
                _individual = individual

        _individualHeader = [list(self.x)[i] for i in range(len(_individual)) if _individual[i] == 1]
        return _individual.fitness.values, _individual, _individualHeader