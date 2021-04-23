# 实验四——k-means

姓名：刘威

学号：PB18010469
$$
\newcommand{\b}{\boldsymbol}
$$


## 实验目的

+ 理解k-means算法的思想
+ 实现k-means算法并将其应用到无监督学习聚类任务上


## 实验原理

给定样本集合 $D=\{x_{1},x_{2},\cdots,x_{m}\}$, “k均值”（k-means) 算法针对聚类所得簇划分 $\mathcal{C}=\{C_{1},C_{2},\cdots,C_{k}\}$ 最小化平方误差
$$
E=\sum_{i=1}^{k}\sum_{\b{x}\in C_{i}}\|\b{x}-\b{\mu}_{i}\|_{2}^{2}
$$
其中 $\b{\mu}_{i}=\frac{1}{|C_{i}|}\sum_{\b{x}\in C_{i}}\b{x}$ 是簇 $C_{i}$ 的均值向量. 直观来看, 上式在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度, $E$ 值越小则簇内样本相似度越高.

最小化上式并不容易. k 均值算法采用了贪心策略, 通过迭代优化来近似求解. 算法流程如下:

![image-20210104211336367](C:\Users\liuwei\AppData\Roaming\Typora\typora-user-images\image-20210104211336367.png)

## 实验结果

### 程序结构

本实验需要实现 k-means 算法，并计算聚类结果的DBI值. k-means算法的程序结构如下：

```python
class KMeans(object):
    def __init__(self,
                 n_clusters: int,
                 *,
                 init: Literal['random', 'from_data'] = 'random',
                 stop: Literal['centroids', 'labels'] = 'centroids',
                 n_init: int = 10,
                 max_iter: int = 300,
                 random_state: int = None):
        """
        n_clusters: number of clusters
        init: method for centroids initialization
            - 'random': generate randomly
            - 'from_data': select from data randomly
        stop: criteria of algorithm termination
            - 'centroids': when centroids do not change
            - 'labels': when labels do not change
        n_init: number of time the k-means algorithm will be run with different centroid seeds.
        max_iter: maximum number of iterations of the k-means algorithm for a single run.
        random_state: seed of random number generator
        """
        pass

    def fit(self, X: np.ndarray):
        pass
    
    def plot_clusters(self):
        """
        plot clusters with different colour
        """
        pass
    
    def __run(self, seed):
        """
        run whole k-means once with random_state=`seed`
        return:
        	n_iter: number of iteration
        	centroids: centers of clusters
        	labels: predicted classes of data
        """
        pass
    
    def __gen_centroids(self, seed):
        """
        generate centroids according to value of self.__init
        	- 'random': generate randomly
            - 'from_data': select from data randomly
        """
    	pass
    
    def __update_labels(self, centroids):
        """
        assign each sample point to its nearest centroid
        return:
        	labels
        """
        pass
    
    def __update_centroids(self, labels):
        """
        calculate centroids of each cluster according to new labels
        return:
        	centroids
        """
        pass
```


### 模型效果

加载数据并输入到模型中：

```python
# load data
data = np.array(list(np.load('./k-means/k-means.npy').item().values()))
data = np.array(data).reshape((-1, data.shape[2]))

# create a kmeans model and fit it
model = KMeans(n_clusters=3, init='from_data', stop='centroids', n_init=5, max_iter=30, random_state=123)
model.fit(data)
```

这里可以看到模型有一些参数可以选择. 关键的几个参数含义如下：

 `init` 参数表示初始化均值向量的方式：设为 `random`表示随机生成; 设为 `from_data` 表示从数据中随机选择.

`stop` 参数表示递归中止的条件：设为 `centroids` 表示 “簇中心向量不再发生变化”；设为 `labels` 表示 “样本分类类别标签不再发生变化.

`n_init` 参数表示随机重启的次数，可以避免偶然较差的初始化点影响聚类结果.

我们将`n_init` 设为 `5` ，表示将用 5 次不同的随机种子运行 5 次k-means 算法，取其中 DBI 最低的结果作为最终聚类结果。

然后对 `init` 参数和 `stop` 参数进行排列组合，共四种情况，分别运行。

由于数据可分性较好，四种组合方式得到的结果没有明显差异。以 `init='random'`, `stop='centroids'` 为例，结果如下：

```powershell
cluster centroids are:
[[31.6182 11.0314]
 [36.782  34.1022]
 [10.444  18.2102]]
DBI = 0.4936301720158927
```

![](F:\Courses\3Fall\IML\exp\实验4\fig\random_centroids.png)


## 实验总结

本次实验原理较为简单，算法流程也容易理解，实现起来困难不大。但有以下几点需要注意：

+ 在随机初始化起始步均值向量时，要注意随机数的范围。如果直接使用默认的[0,1)随机数，这与数据的范围不匹配，将使得初始点相距样本较远，可能导致所有样本点都属于属于同一个簇，而且进一步的迭代不起任何作用。较好的方法是，对每一个维度，选取该维度上数据的范围作为随机数的范围，即将[0,1)随机数做变换，使得生成的均值向量在数据的范围内。

+ 当某一步迭代后，某个簇为空，将会导致这个簇的中心无法计算。这时可以去掉这个簇，这将会使得最终得到的簇比预设的少。如果不想这样做，仍然希望分出预设的簇数，该空簇可以沿用上一步迭代后的簇中心而不更新。

本次实验的数据可分性较好，模型的效果十分完美。