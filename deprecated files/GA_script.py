# -*- coding: utf-8 -*-
"""
Created on Fri May  8 04:24:56 2020

@author: TAC
"""
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

dataset = pd.read_csv('file.csv')
X, y = dataset.iloc[:,0:-1], le.fit_transform(dataset.iloc[:,-1].values)
features = dataset.columns[0:-1]

from deap import creator, base, tools, algorithms
import random
import numpy as np
from deap import tools
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class FeatureSelectionGA:

    def __init__(self, model, x, y, cv_split=5, n_pop=15, n_gen=10):
      
        self.model = model
        self.n_features = x.shape[1]
        self.cv_split = cv_split
        self.x = x
        self.y = y
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.val=[]
        print("The number of Features: {}".format(self.n_features))
        print("The Shape of Training Data is : {} and Target Data is : {}".format(self.x.shape, self.y.shape))
    def sol(self):
        individual = [1 for i in range(self.x.shape[1])]
        print("Accuracy For All the features: " + str(self.fitness_test(individual)) + "\n")

        # Applying Genetic Algorithm
        eg = self.evolutionary_algorithm()
        hof=eg[0]
        log=eg[1]
        data = [[i for i in item.values()] for item in log]
        df = pd.DataFrame(data, columns=log.header)
        # gen=log.select("gen")
        # avg=log.select("avg")
        # std=log.select("std")
        # #plt.plot(gen,avg)
        # plt.errorbar(gen,avg,std)
        # plt.title('Decision Tree Classifier',fontsize=12)
        # plt.xlabel('generation',fontsize=12)
        # plt.ylabel('accuracy',fontsize=12)
        # plt.savefig('DT_error_bar.png')
        #plt.imshow()
        accuracy, individual, header = self.bestIndividual(hof)
        print('Best Accuracy: \t' + str(accuracy[0]))
        print('Number of Features in Subset: \t' + str(individual.count(1)))
        print('Feature Subset: ' + str(header)+'\n')
        #print('\n\nKindly Create a New Classifier with the Above Feature Set')
        return df,header




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
            cross_v=cross_val_score(classifier, X_subset, self.y, cv=self.cv_split)
            acc=sum(cross_v) / float(len(cross_v))
            #return acc            
            return (alpha*(1.0 - acc) + (1.0 - alpha)*(1.0 - (X_subset.shape[1])/10),0)#10 is total feature
            #return (acc,0)
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
        toolbox.register("mate", tools.	cxOnePoint)#crossover
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.3) #mutation 
        toolbox.register("select", tools.selTournament, tournsize=5) #selection 

        # Initialize Parameters
        pop = toolbox.population(n=self.n_pop)
        hof = tools.HallOfFame(self.n_pop * self.n_gen)#  best individual that ever lived in the population during the evolution.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean) #avg score from fitness function
        stats.register("min", np.min)  #min score from fitness function
        stats.register("max", np.max)  #max score from fitness function
        stats.register("std", np.std)  #max score from fitness function

        # Genetic Algorithm
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=self.n_gen, stats=stats, halloffame=hof,verbose=True)

        # Return hall of fame
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
    
est = RandomForestClassifier()    

df=[]
feat=[]
for i in range(30):
    mod=FeatureSelectionGA(est,X,y,n_gen=20)
    x1,x2=mod.sol()
    df.append(x1.values)
    feat.append(x2)

df1 = np.mean(df,axis=0)


#plt.plot(df1[:,0],df1[:,2])
plt.errorbar(df1[:,0],df1[:,2],df1[:,5])
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Decsion Tree')

