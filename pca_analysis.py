import numpy as np
from sklearn.decomposition import PCA

from plotly.offline import init_notebook_mode, plot
import plotly.graph_objs as go

roberts =   [100, 83, 65, 43, 52, 60, 42, 52, 70]
kennedy =   [83, 100, 73, 42, 43, 68, 38, 48, 75]
thomas =    [65, 73, 100, 23, 27, 90, 17, 30, 70]
ginsburg =  [43, 42, 23, 100, 83, 18, 93, 82, 27]
breyer =    [52, 43, 27, 83, 100, 22, 88, 92, 35]
alito =     [60, 68, 90, 18, 22, 100, 8, 25, 70]
sotomayor = [42, 38, 17, 93, 88, 8, 100, 85, 22]
kagan =     [52, 48, 30, 82, 92, 25, 85, 100, 37]
gorsuch =   [70, 75, 70, 27, 35, 70, 22, 37, 100]

x = [roberts, kennedy, thomas, ginsburg, breyer, alito, sotomayor, kagan, gorsuch]
names = ['roberts', 'kennedy', 'thomas', 'ginsburg', 'breyer', 'alito', 'sotomayor', 'kagan', 'gorsuch']

n_components = 1
x_reduced = PCA(n_components=n_components).fit_transform(x)

y = [0] * len(names)
if n_components == 2:
    y = x_reduced[:,1]
trace = go.Scatter(
    x = x_reduced[:,0],
    y = y,
    mode='markers+text',
    text=names,
    textposition='bottom center'
)
data = [trace]
plot(data, filename='./graphs/{0}_component_pca.html'.format(n_components))