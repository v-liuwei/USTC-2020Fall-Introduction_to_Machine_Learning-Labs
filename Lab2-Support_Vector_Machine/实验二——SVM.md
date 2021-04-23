# 实验二——SVM

姓名：刘威

学号：PB18010469
$$
\newcommand{\b}{\boldsymbol}
\newcommand{\T}{^{\mathrm{T}}}
$$


## 实验目的

+ 熟悉SVM的原理并实现
+ 了解机器学习及模型学习及评估的基本方法

## 实验原理

### SVM模型

**支持向量机**是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器。

间隔可以使用 $ \gamma = \frac{2}{||w||} $ 来表示，这样求解 `SVM` 模型可以变成下面的优化问题：
$$
\mathop{\max}_{\b{w},b} \frac{2}{\| \b{w} \| } \\
\text{s.t.} \, y_i(\b{w}^T \b{x_i} + b) \geq 1  \quad i=1, ..., m
$$
等价于：
$$
\mathop{\min}_{\b{w},b} \frac{1}{2} \| \b{w} \|^{2} \\
\text{s.t.} \, y_i(\b{w}^T \b{x_i} + b) \geq 1  \quad i=1,\dots,m
$$
上面的模型只能解决线性可分的问题，为了解决有部分数据点线性不可分的情况，需要增加软间隔， 软间隔允许某些样本不满足约束 $y_i(\b{w}^T \b{x_i} + b) \ge 1$. 为了使不满足约束的样本尽可能少, 优化目标可以写为
$$
\min_{\b{w},b}\ \frac{1}{2}\|\b{w}\|^{2}+C\sum_{i=1}^{m}l_{0/1}(y_{i}(\b{w}\T\b{x}_{i}+b)-1)
$$
其中 $C>0$ 是一个常数，$l_{0/1}$ 是 “0/1损失函数”
$$
l_{0/1}=\begin{cases}1, & \text{if } z<0;\\0, & \text{otherwise}.\end{cases}
$$
由于 $l_{0/1}$ 非凸、非连续，数学性质不太好，下面使用如下的 hinge loss 函数来替代它：
$$
l_{\text{hinge}}(z)=\max(0, 1-z)
$$
如此， 优化问题变成：
$$
\min_{\b{w},b}\ \frac{1}{2}\|\b{w}\|^{2}+C\sum_{i=1}^{m}\max(0, 1-y_{i}(\b{w}\T\b{x}_{i}+b))
$$

### 模型学习方法

为了求解模型中的参数 $\b{w}$ 和 $b$，我们可以使用**梯度下降法**.

记要优化的式子为 $L$, 记 $\xi_{i}=1-y_{i}(\b{w}\T\b{x}_{i}+b)$, 则
$$
\begin{aligned}
\frac{\partial L}{\partial \b{w}}&=\b{w}-C\sum_{\xi_{i}\ge 0}y_{i}\b{x}_{i}\\
\frac{\partial L}{\partial b}&=-C\sum_{\xi_{i}\ge 0}y_{i}
\end{aligned}
$$
**梯度下降法：**
$$
\begin{aligned}
&\text{while}\, \|  \frac{\partial L}{\partial \b{w}} \| +\|\frac{\partial L}{\partial b} \| >\delta\, \text{do}\\
&\quad \text{for }i=1\text{ to } m \text{ do}\\
&\quad\quad \xi_{i}\leftarrow1-y_{i}(\b{w}\T_{t}\b{x}_{i}+b_{t})\\
&\quad \b{w}_{t+1}\leftarrow \b{w}_{t} - \eta(\b{w}_{t}- C \sum_{\xi_{i}\ge 0}y_{i}\b{x}_{i})  \\
&\quad b_{t+1} \leftarrow b_{t}  -  \eta (-C \sum_{\xi_{i}\ge 0}y_{i}) \\
&\text{end while}
\end{aligned}
$$


## 实验结果

根据上述原理，使用`python`用**梯度下降法**实现了 SVM。


### 程序结构

程序结构如下：

```python
class SVMClassifier(object):
    def __init__(self,
                 learning_rate: float,
                 max_iter: int,
                 C: float,
                 optimizer: Literal['GD', 'SMO', None] = None,
                 seed: Optional[int] = None):
        pass
    	"""
    	这里并未实现SMO算法
    	"""
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        用数据集(X,y)拟合模型，并记录拟合过程中每个epoch的训练错误率及验证错误率(如果提供验证集val_data)
        """
        pass
    def predict(self, X: np.ndarray):
        """
        用训练好的模型预测X的标签
        """
        pass
    def score(self, 
              X: np.ndarray,
              target: np.ndarray,
              metric: Literal['err', 'acc', 'f1'] = 'acc'):
        """
       	用模型预测X的标签并与真实标签target比较，计算评估函数值。
        """
        pass
    def plot_boundary(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      sv: bool = True):
        """
        绘制决策边界，支持超平面，并标出支持向量
        """
        pass
    def plot_learning_curve(self):
        """
        绘制学习曲线
        """
        pass
```


### 实验结果

在训练过程中，每训练一轮都计算出参数在训练集和测试集上的错误率，并且最后在训练集和测试集上绘制训练出的分类决策边界。

参数设置为：

```python
SVMClassifier(learning_rate=0.001, max_iter=500, C=1, optimizer='GD', seed=0)
```

绘制出学习曲线如下：

![](F:\Courses\3Fall\IML\exp\实验2\fig\learning_curve.png)

经实现发现，无论怎么调整参数，学习曲线都会震荡得很厉害，但是最终可以收敛。在上图中，大概训练400轮，模型的错误率就基本不再变化了。但是实际上此时模型并未真正收敛，此时支持平面仍会有较大变化，说明参数还有变化。实践发现，将训练轮数增大到 5000轮后，模型的参数也将基本保持不变，达到真正的收敛。

下面将 max_iter 调整为 5000 绘制决策边界、支持超平面，并标出支持向量。

**训练集：**

![](F:\Courses\3Fall\IML\exp\实验2\fig\train_boundary.png)

**测试集：**

![](F:\Courses\3Fall\IML\exp\实验2\fig\test_boundary.png)

可以看到模型在训练集和测试集上均达到了较好的效果。模型的预测精度 **accuracy** 为：

训练集：0.957

测试集：0.933


## 实验总结

本次实验实现了 SVM 的梯度下降求解方法。在实现过程中发现了如下问题。

助教提供的PPT上给出的梯度下降算法如下：

<img src="C:\Users\liuwei\AppData\Roaming\Typora\typora-user-images\image-20201204184912235.png" alt="image-20201204184912235" style="zoom:80%;" />

差别在于它每次只选择误差最大的样本去更新参数，似乎也是可行的。刚开始我按照这个方法实现，但是不论怎么调整参数，模型都无法收敛，而且效果也很差，还不知道是什么原因。后来改为使用梯度下降，就不存在这样的问题了。