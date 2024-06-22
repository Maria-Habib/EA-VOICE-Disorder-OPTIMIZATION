# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time
from ast import literal_eval
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import math
import csv


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# objective function
def objective(x, train, alpha):
    x_features = x[0:-2].copy()

    X = train.loc[:, train.columns != 'label']
    selectedCols = [x for i, x in enumerate(X.columns) if x_features[i] == 1]

    c = abs(float(x[-1] * 34.9) + 0.1)  # gm_actual = gm_scaled * (max-min)+min
    gama = abs(float(x[-2] * 0.00009999) + 0.00000001)

    if gama == 0.0:
        gama = 1 / len(selectedCols)

    if c == 0.0:
        c = 1

    print(f'At the inner-stage evaluation: Cost = {c}, gamma = {gama}')
    X = X.loc[:, selectedCols]

    X = X.to_numpy()
    y = train['label'].values
    results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svm = SVC(C=c, gamma=gama, random_state=int(time.time()))

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        print(classification_report(y_test, y_pred, digits=3))

        results.append(float(f1_score(y_test, y_pred)))

    f1 = np.mean(results)
    fitness = alpha * (1 - f1) + (1 - alpha) * (sum(x_features) / len(x[0:-2]))

    return fitness


# genetic algorithm
def genetic_algorithm(train, objective, alpha, n_bits, n_iter, n_pop, r_cross, r_mut, dir):
    # initial population of random bitstring
    convergence = []
    lst = []
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    for individual in pop:
        individual[-1] = np.random.uniform(0.0, 1.0)
        individual[-2] = np.random.uniform(0.0, 1.0)

    # keep track of best solution
    best, best_eval = 0, objective(pop[0], train, alpha)

    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c, train, alpha) for c in pop]

        head_individuals = [f'individual_{i}' for i in range(len(scores))]

        lst.append(zip(head_individuals, scores, [list(x) for x in pop]))

        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        convergence.append(best_eval)

        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                c[-1] = np.random.uniform(0.0, 1.0)
                c[-2] = np.random.uniform(0.0, 1.0)
                children.append(c)
        # replace population
        pop = children

    lst = [item for sublist in lst for item in sublist]
    pd.DataFrame(lst).to_csv(
        f'results/GA/{dir}/float-enc/pop/{dir}_GA_population_fitness_alpha={alpha}_GA_po_{n_pop}_iter_{n_iter}_' + time.strftime("%Y-%m-%d-%H-%M-%S" + '.csv'),
        index=False)

    return [convergence, best, best_eval]


def evaluate(individual, train, test):
    cost = abs(float(individual[-1] * 34.9) + 0.1)  # gm_actual = gm_scaled * (max-min)+min
    gamma = abs(float(individual[-2] * 0.00009999) + 0.00000001)  # #[0.00000001 - 0.0001]

    x_features = individual[0:-2].copy()
    X_train = train.loc[:, train.columns != 'label']
    selectedCols = [x for i, x in enumerate(X_train.columns) if x_features[i] == 1]
    X_train = X_train.loc[:, selectedCols]
    print('in optimizer the num of features after FS %s' % len(X_train.columns))
    y_train = train['label'].values.tolist()

    X_test = test.loc[:, test.columns != 'label']
    selectedCols = [x for i, x in enumerate(X_test.columns) if x_features[i] == 1]
    X_test = X_test.loc[:, selectedCols]
    y_test = test['label'].values.tolist()

    if gamma == 0.0:
        gamma = 1 / len(selectedCols)

    if cost == 0.0:
        cost = 1
    print(f'At the inner-stage evaluation: Cost = {cost}, gamma = {gamma}')

    svm = SVC(C=cost, gamma=gamma, random_state=int(time.time()))
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_score1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print("Accuracy: %.3f%%" % (accuracy * 100.0))
    print("F1-score: %.3f%%" % (f1_score1 * 100.0))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    return tn, fp, fn, tp, accuracy, f1_score1, recall, precision, report, selectedCols, cost, gamma, y_test, y_pred


# define the total iterations
train = pd.read_csv('data/svd_train.tsv', sep='\t')
test = pd.read_csv('data/svd_test.tsv', sep='\t')


Runs = 15
n_iter = 30


n_bits = len(train.columns) + 1
n_pop = 30  # the population size
r_cross = 0.5  # crossover rate
r_mut = 1.0 / float(n_bits)  # mutation rate

dir = 'svd'

ALPHA = [0.2, 0.5, 0.8, 0.85, 0.9, 0.95, 0.999]

for value in ALPHA:

    ExportToFile = f"results/GA/{dir}/float-enc/experiment-{dir}-GA-alpha={value}-pop={n_pop}-iter={n_iter}-runs={Runs}_" + time.strftime(
        "%Y-%m-%d-%H-%M-%S") + ".csv"
    ACC = np.zeros(n_iter)
    RECALL = np.zeros(n_iter)

    PREC = np.zeros(n_iter)
    P0 = np.zeros(n_iter)
    P1 = np.zeros(n_iter)
    R0 = np.zeros(n_iter)
    R1 = np.zeros(n_iter)
    F0 = np.zeros(n_iter)
    F1 = np.zeros(n_iter)
    Gm = np.zeros(n_iter)
    FPR = np.zeros(n_iter)
    FNR = np.zeros(n_iter)
    AUC = np.zeros(n_iter)

    TN = np.zeros(n_iter)
    TP = np.zeros(n_iter)
    FP = np.zeros(n_iter)
    FN = np.zeros(n_iter)

    for r in range(0, Runs):
        # perform the genetic algorithm search
        convergence, best, score = genetic_algorithm(train, objective, value, n_bits, n_iter, n_pop, r_cross, r_mut, dir)
        if best == 0:
            continue

        print('Done!')
        print('f(%s) = %f' % (best, score))
        print(convergence)

        # train and evaluate the best individual
        tn, fp, fn, tp, accuracy, f1_score1, recall, precision, report, selectedCols, cost, gamma, y_test, y_pred = evaluate(
            best, train, test)

        TN[0] = tn
        TP[0] = tp
        FP[0] = fp
        FN[0] = fn

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)
        auc = (((1 - fpr) * (1 + tpr)) / 2) + ((fpr * tpr) / 2)

        print(report)
        P0[0] = report['0']['precision']
        P1[0] = report['1']['precision']
        R0[0] = report['0']['recall']
        R1[0] = report['1']['recall']
        F0[0] = report['0']['f1-score']
        F1[0] = report['1']['f1-score']
        Gm[0] = math.sqrt(report['0']['recall'] * report['1']['recall'])
        FPR[0] = fpr
        FNR[0] = fnr
        AUC[0] = auc

        ACC[0] = accuracy
        F1[0] = f1_score1
        RECALL[0] = recall
        PREC[0] = precision

        CnvgHeader = ["Iter" + str(i + 1) for i in range(0, n_iter)]

        try:
            with open(ExportToFile, 'a', newline='\n') as out:
                writer = csv.writer(out, delimiter=',')

                header = np.concatenate([CnvgHeader])
                writer.writerow(header)

                a = np.concatenate([['Convergence'], convergence])
                b = np.concatenate([['Best ACC'], ACC])
                c = np.concatenate(
                    [['Best F1'], F1])
                e = np.concatenate(
                    [['Best Recall'], RECALL])
                d = np.concatenate(
                    [['Best Precision'], PREC])

                tn = np.concatenate([['TN'], TN])
                tp = np.concatenate([['TP'], TP])
                fp = np.concatenate([['FP'], FP])
                fn = np.concatenate([['FN'], FN])

                p0 = np.concatenate([['P0'], P0])
                p1 = np.concatenate([['P1'], P1])
                r0 = np.concatenate([['R0'], R0])
                r1 = np.concatenate([['R1'], R1])
                f0 = np.concatenate([['F0'], F0])
                f1 = np.concatenate([['F1'], F1])
                gm = np.concatenate([['GM'], Gm])
                fprr = np.concatenate([['FPR'], FPR])
                fnrr = np.concatenate([['FNR'], FNR])
                aucc = np.concatenate([['AUC'], AUC])

                writer.writerow(a)
                writer.writerow(b)
                writer.writerow(c)
                writer.writerow(e)
                writer.writerow(d)

                writer.writerow(tn)
                writer.writerow(tp)
                writer.writerow(fp)
                writer.writerow(fn)
                writer.writerow(p0)

                writer.writerow(p1)
                writer.writerow(r0)
                writer.writerow(r1)
                writer.writerow(f0)
                writer.writerow(f1)
                writer.writerow(gm)

                writer.writerow(fprr)
                writer.writerow(fnrr)
                writer.writerow(aucc)

                writer.writerow(['Selected Columns', selectedCols])
                writer.writerow(['len Columns', len(selectedCols)])
                writer.writerow(['feature reduction', (847 - len(selectedCols)) / 847])
                writer.writerow(['cost', cost])
                writer.writerow(['gamma', gamma])
                writer.writerow(['y_test', y_test])
                writer.writerow(['y_pred', list(np.array(y_pred))])
                writer.writerow("")

        except Exception as e:
            print(e)
