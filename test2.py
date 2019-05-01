import re
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
def plus(f, a, b):
    return f[a] + f[b]


df = pd.DataFrame([[1, 2, 3], [2, 3, 4], [5, 2, 4]], columns=['c1', 'c2', 'c3'])

cnt = df['c2'].value_counts().to_dict()

print(cnt)

df['c4'] = 0
print(df)

df2 = df.ix[df['c2'] == 2].copy()
print(df2)

dfc = df.copy()
df['c1'] += 1

df['c1'] = df['c1'] + dfc['c1']
print(df)

print(df['c1'].mean())

with open('data/HDFS/data_instances.stat.pkl', 'rb') as f:
    d = pickle.load(f)
    print(d['mean'])
    print(d['std'])

with open('data/HDFS/data_instances_100.stat.pkl', 'rb') as f:
    d = pickle.load(f)
    print(d['mean'])
    print(d['std'])

def g(t):
    for i in range(t):
        yield i * 2, i * 3

for i, j in g(3):
    print(i)
    print(j)

'''
s = pd.DataFrame([1, 2], columns=['s'])
s2 = pd.DataFrame(np.arange(7), columns=['s2'])

# Make boxplot for one group only
df = pd.concat([s, s2], axis=1)
print(df)
p = df.boxplot()
p.get_figure().savefig('test2.png')

a = np.array([2, 4, 5, 6])
for i, t in enumerate(a):
    print(i, t)
s2 = s2['s2']
s3 = pd.Series(np.array([1, 2, 3]))

print(s2)
print(s3)
print(s3.values)
print(len(s3.values))
