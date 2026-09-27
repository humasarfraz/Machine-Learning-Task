import sys
import timeit
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt

print("python      ", sys.version.split()[0])
print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("matplotlib  ", matplotlib.__version__)

RANDOM_STATE = 1
rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

plt.rcParams['figure.figsize'] = (8, 4.5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

a = np.array([[1., 2., 3.],
              [4., 5., 6.]])

print('shape       ', a.shape, ' dtype', a.dtype)
print('mean axis=0 ', a.mean(axis=0))
print('mean axis=1 ', a.mean(axis=1))
print('a * 2       ', a[0] * 2)
print('column 1    ', a[:, 1])
print('mask a > 3  ', a[a > 3])

z = (a - a.mean(axis=0)) / a.std(axis=0)
print('standardised:\n', np.round(z, 3))

big = rng.normal(size=(100_000, 20))

def loop_means(arr):
    means = []
    for j in range(arr.shape[1]):
        total = 0.0
        for value in arr[:, j]:
            total += value
        means.append(total / arr.shape[0])
    return np.array(means)

loop_time = min(timeit.repeat(lambda: loop_means(big), repeat=3, number=5)) / 5
vector_time = min(timeit.repeat(lambda: big.mean(axis=0), repeat=3, number=5)) / 5
print(f"Loop time: {loop_time:.6f} s")
print(f"Vectorised time: {vector_time:.6f} s")
print(f"Speed ratio: {loop_time / vector_time:.1f}x")

df = pd.read_csv(Path(__file__).with_name('agriculture_seeds.csv'))
print(df.shape)
print(df.head())
df.info()
print(df.describe().T[['min', 'max', 'mean', 'std']].round(3))
print(df['Class'].value_counts(normalize=True))
print(df.isnull().sum().sum(), 'missing values')

X = df.drop(columns='Class').to_numpy()
y = df['Class'].to_numpy()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].hist(X[:, 0], bins=20, edgecolor='black', label='kernel area')
ax[0].set_xlabel('kernel area [mm²]')
ax[0].set_ylabel('count [samples]')
ax[0].set_title('Histogram — kernel area')
ax[0].legend()

for cl, m in zip(np.unique(y), ('o', 's', '^')):
    ax[1].scatter(X[y == cl, 0], X[y == cl, 1],
                  marker=m, alpha=0.8, edgecolor='k',
                  label=f'class {cl}')
ax[1].set_xlabel('kernel area [mm²]')
ax[1].set_ylabel('perimeter [mm]')
ax[1].set_title('Scatter — seed class separation')
ax[1].legend()

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe.fit(X_train, y_train)

print('train accuracy %.3f' % pipe.score(X_train, y_train))
print('test  accuracy %.3f' % pipe.score(X_test, y_test))

pred = pipe.predict(X_test)
wrong = np.where(pred != y_test)[0]
print("Wrong predictions:", len(wrong))
if len(wrong):
    j = wrong[0]
    print("True class:", y_test[j])
    print("Predicted class:", pred[j])
    print("Test sample:", np.round(X_test[j], 3))

rows = []
for k in (1, 5, 25, 100):
    p = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    p.fit(X_train, y_train)
    rows.append({
        'variant': 'scaled',
        'k': k,
        'train': p.score(X_train, y_train),
        'test': p.score(X_test, y_test)
    })

    raw = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    rows.append({
        'variant': 'unscaled',
        'k': k,
        'train': raw.score(X_train, y_train),
        'test': raw.score(X_test, y_test)
    })

results = pd.DataFrame(rows).pivot(
    index='k', columns='variant', values=['train', 'test']
).round(3)

print(results)
