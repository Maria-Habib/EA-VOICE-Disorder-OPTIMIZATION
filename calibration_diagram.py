# SVM reliability diagram
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from matplotlib import pyplot
import pandas as pd




train = pd.read_csv('data/svd_train.tsv', sep='\t')
test = pd.read_csv('data/svd_test.tsv', sep='\t')

X_train = train.loc[:, train.columns != 'label']
y_train = train['label'].values.tolist()

X_test = test.loc[:, test.columns != 'label']
y_test = test['label'].values.tolist()


model = SVC()
model.fit(X_train, y_train)
# predict probabilities
probs = model.decision_function(X_test)
# reliability diagram
fop, mpv = calibration_curve(y_test, probs, n_bins=10, normalize=True)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
pyplot.plot(mpv, fop, marker='.')
pyplot.show()