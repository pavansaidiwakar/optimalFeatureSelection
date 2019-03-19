import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sys
import math
import time
import warnings
import csv
warnings.filterwarnings("ignore")


def avg(l):
    """
    Returns the average between list elements
    """
    return sum(l)/float(len(l));




def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # apply classification algorithm
        clf = DecisionTreeClassifier()


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
        g = individual.fitness.values
        if(g[0] > maxAccurcy):
            maxAccurcy = g[0]
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
    output=[]

# encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]

    # get accuracy with all features
    t1=time.time()
    individual = [1 for i in range(len(X.columns))]

    totalfeatures=individual.count(1)
    fitval=avg(getFitness(individual, X, y))
    print(("Accuracy with all features: (directly passing dataset)\t" +
          str(fitval)+ "\n"))
    t1=time.time()-t1;

    t2=time.time();
    # apply genetic algorithm
    hof = geneticAlgorithm(X, y, n_pop, n_gen)

    # select the best individual
    #print(hof)
    accuracy, individual, header = bestIndividual(hof, X, y)
    print(('Best Accuracy: \t' + str(accuracy)))
    print(('Number of Features in Datset: \t' + str(individual.count(0)+individual.count(1))))
    print(('Number of Features in Subset: \t' + str(individual.count(1))))
    print(('Individual: \t\t' + str(individual)))
    print(('Feature Subset\t: ' + str(header)))

    genfeatures=individual.count(1)
    print('\n\ncreating a new classifier with the result of genetic algorithm')

    # read dataframe from csv one more time
    df = pd.read_csv(dataframePath, sep=',')

    # with feature subset
    X = df[header]
    #print(header)


    print(individual)
    clf = DecisionTreeClassifier()

    scores = cross_val_score(clf, X, y, cv=10)
    fitval1=avg(scores)
    print(("Accuracy with Feature Subset: \t(after doing geneticAlgoSel)" + str(fitval1) + "\n"))

    t3=time.time()-t2;



    #print((df[header].corr()))
    df1= df[header].corr()
    df1['mean'] = df1.mean(axis=1)
    sorted_df = df1.sort_values(by='mean')


    index_list=list(sorted_df.index.values)

    block1=[]
    block2=[]
    block3=[]
    block4=[]
    block5=[]

    direction=1
    i=0
    nb=4
    while i<len(index_list):
		     if(direction==1 and i%4==0):
		           block1.append(index_list[i])

		     elif(direction==1 and i%4==1):
		           block2.append(index_list[i])

		     elif(direction==1 and i%4==2):
		           block3.append(index_list[i])
		     elif(direction==1 and i%4==3):
		           block4.append(index_list[i])
		           direction=-1
		     elif(direction==-1 and i%4==0):
		           block4.append(index_list[i])
		     elif(direction==-1 and i%4==1):
		           block3.append(index_list[i])

		     elif(direction==-1 and i%4==2):
		           block2.append(index_list[i])

		     elif(direction==-1 and i%4==3):
		           block1.append(index_list[i])

		           direction=1
		     i=i+1
    '''else:
        while i<len(index_list):
             if(direction==1 and i%3==0):
                   block1.append(index_list[i])
             
             elif(direction==1 and i%3==1):
                   block2.append(index_list[i])
                   
             elif(direction==1 and i%3==2):
                   block3.append(index_list[i])
                   
                   direction=-1
             elif(direction==-1 and i%3==0):
                   block3.append(index_list[i])
                   
             elif(direction==-1 and i%3==1):
                   block2.append(index_list[i])
                   
             elif(direction==-1 and i%3==2):
                   block1.append(index_list[i])
                   
                   direction=1
             i=i+1


    '''



    X3=df[block1]
    clf00 = DecisionTreeClassifier()

    scores= cross_val_score(clf00, X3, y,cv=10)


    X4=df[block2]
    clf1 = DecisionTreeClassifier()

    scores= cross_val_score(clf1, X4, y,cv=10)


    X5=df[block3]
    clf2 = DecisionTreeClassifier()

    scores= cross_val_score(clf2, X5, y,cv=10)

    X6=df[block4]
    clf3 = DecisionTreeClassifier()

    scores= cross_val_score(clf3, X6, y,cv=10)


    
    eclf1 = VotingClassifier(estimators=[ ('dt1', clf00), ('dt2', clf1), ('dt3', clf2),('dt4',clf3)], voting='hard')
    
    #eclf1 = VotingClassifier(estimators=[ ('dt1', clf00), ('dt2', clf1), ('dt3', clf2)], voting='hard')    
    eclf1 = eclf1.fit(X, y)

    scores= cross_val_score(eclf1, X, y,cv=10)
    fitval3=avg(scores)
    print(("Accuracy after Majority Voting: \t" +str(fitval3)   + "\n"))
    t4=time.time()-t2;
    print("Time taken when passed entire dataset: "+str(t1));
    print("Time taken for doing genetic algorithm: "+str(t3));
    print("Time taken for majority votin classifer: "+str(t4));
    output.insert(0,dataframePath)
    output.insert(1,totalfeatures)
    output.insert(2,genfeatures)
    output.insert(3,fitval)
    output.insert(4,fitval1)
    output.insert(5,fitval3)
    output.insert(6,nb)
    output.insert(7,t1)
    output.insert(8,t3)
    output.insert(9,t4)
    with open('outputfile.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(output)
    csvFile.close()
