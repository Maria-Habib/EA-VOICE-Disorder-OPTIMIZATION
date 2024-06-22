

import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from collections import Counter

fig = plt.figure(figsize=(6, 9))

df = pd.read_csv('results/GA/sli/float-enc/experiment-sli-GA-alpha=0.85-pop=30-iter=30-runs=15_2024-01-27-16-25-01.csv')

print(df.columns)
df = df.loc[df['Iter1'].isin(['Selected Columns'])]
df.drop('Iter1', axis= 1, inplace=True)

'''
df.drop(['EndTime', 'ExecutionTime'], axis=1, inplace=True)
df = df.loc[df['objfname'].isin(['Selected Columns'])]
df.drop('Measure ', axis=1, inplace=True)
df = df.iloc[0:15, 2]
'''

print(df.head())
print(len(df))

print(df.columns)

data = []
for i in range(len(df)):
    label = df.iloc[i]
    y = list(range(15))
    data.append(label['Iter2'])
    # plt.plot(y,x ,label=f'{label}_run_{i}')

print(data)
print("**********************")

data = [ast.literal_eval(i) for i in data]
data = [item for sub in data for item in sub]

data = Counter(data)

sorted_data = sorted(data.items(), reverse=True, key=lambda x: x[1])

print(sorted_data)
sorted_categories = [item[0] for item in sorted_data]
sorted_values = [item[1] for item in sorted_data]


plt.barh(sorted_categories[0:49], sorted_values[0:49], color='#A7C7E7', )  # pso=#371713  woa=#171337
plt.ylabel("Feature", fontsize=13)
plt.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)

# setting label of x-axis
plt.xlabel("Count", fontsize=13)
# plt.title("Frequency of Features", fontweight='bold', fontsize=13)

plt.grid()
plt.show()
fig.savefig('sli_ga_features.pdf')
