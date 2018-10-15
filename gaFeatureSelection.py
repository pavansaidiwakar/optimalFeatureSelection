import pandas as pd
import numpy as np
import random
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition, preprocessing
from deap import creator, base, tools, algorithms
#from factor_analyzer import FactorAnalyzer
import sys
import math


def avg(l):
    """
    Returns the average between list elements
    """
    return sum(l)/float(len(l))
    '''sum=0
    for i in l:
         sum=sum+numpy.tanh(i)
    return sum'''
    

def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(len(individual)) if individual[index] == 0] #contains the unselected features in feature string	

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1) #dropping the columns whose feature is not present in feature string
        X_subset = pd.get_dummies(X_parsed) 

        # apply classification algorithm
        clf = DecisionTreeClassifier()
        #clf = LogisticRegression()
	#print(cross_val_score(clf, X_subset, y, cv=5))
        return (avg(cross_val_score(clf, X_subset, y, cv=10)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        p1=individual.fitness
        if(p1.values > (maxAccurcy)):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


def getArguments():
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen


if __name__ == '__main__':
    # get dataframe path, population number and generation number from command-line argument
    dataframePath, n_pop, n_gen = getArguments()
    # read dataframe from csv
    df = pd.read_csv(dataframePath, sep=',')

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1]) #last column of dataframe, so it contains the output class(1,2,3)
    y = le.transform(df.iloc[:, -1]) #y contains the label given by the fit
    X = df.iloc[:, :-1] #all table except last column 

    # get accuracy with all features
    individual = [1 for i in range(len(X.columns))]
    print("Accuracy with all features: \t" +
          str(getFitness(individual, X, y)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X, y, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')

    # read dataframe from csv one more time
    df = pd.read_csv(dataframePath, sep=',')

    # with feature subset
    X = df[header]
    print(individual)
    data_normal = preprocessing.scale(X)
    fa = decomposition.FactorAnalysis(n_components=1)
    fa.fit(data_normal)
    print fa.components_ # 
    '''fa= FactorAnalyzer()
    fa.analyze(X, 3 , rotation= None)
    ev, v = fa.get_eigenvalues()
    print(ev)
    print(v)
    '''
    #clf = LogisticRegression()
    clf = DecisionTreeClassifier()

    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")


