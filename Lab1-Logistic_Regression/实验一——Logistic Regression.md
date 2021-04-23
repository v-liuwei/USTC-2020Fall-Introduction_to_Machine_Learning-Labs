# 实验一——Logistic Regression

姓名：刘威

学号：PB18010469
$$
\newcommand{\b}{\boldsymbol}
\newcommand{\T}{^{\mathrm{T}}}
$$


## 实验目的

+ 熟悉LR的原理并实现
+ 了解机器学习及模型学习及评估的基本方法

## 实验原理


### LR 模型

给定二分类任务数据集 $\{(\b{x}_{i},y_{i})\}_{i=1}^{m}$，其中 $y_{i}\in\{0,1\}$. 

考虑如下线性回归模型：
$$
z=\b{w}\T\b{x}+b
$$
此模型产生的预测值 $z$ 是实值，我们希望将其转换为 $0/1$值.  最理想的是”单位阶跃函数”：
$$
y=\begin{cases}
0, & z<0\\
0.5, & z=0\\
1, & z>0
\end{cases}
$$
即若预测值大于零就判为正例，小于零就判为反例，预测值为零则可以任意判别。但是，单位阶跃函数不连续，不便于优化求解参数 $\b{w}$. 我们采用如下的对数几率函数来近似替代它：
$$
y=\frac{1}{1+e^{-z}}
$$
如下图所示，此函数的形状与单位阶跃函数很接近，它将 $z$ 值转化为一个接近 $0$ 或 $1$ 的 $y$ 值，并且其输出值在 $z=0$ 附近变化很陡。

![](https://chengfeng96.com/blog/2018/12/15/%E5%AF%B9%E6%95%B0%E5%87%A0%E7%8E%87%E5%9B%9E%E5%BD%92%EF%BC%88Logistic-Regression%EF%BC%89%E6%B5%85%E8%B0%88/2.png)

将 $z=\b{w}\T\b{x}+b$ 代入对数几率函数中，得到对率回归模型
$$
y=\frac{1}{1+e^{-(\b{w}\T\b{x}+b)}}
$$
### 模型学习方法

为了求解模型中的参数 $\b{w}$ 和 $b$，我们可以使用**极大似然法**. 

将 $y$ 视为样本 $\b{x}$ 作为正例的可能性，即
$$
p(y=1|\b{x})=\frac{1}{1+e^{-(\b{w}\T\b{x}+b)}}=\frac{e^{1\cdot(\b{w}\T\b{x}+b)}}{1+e^{\b{w}\T\b{x}+b}}
$$
于是
$$
p(y=0|\b{x})=\frac{1}{1+e^{\b{w}\T\b{x}+b}}=\frac{e^{0\cdot(\b{w}\T\b{x}+b)}}{1+e^{\b{w}\T\b{x}+b}}
$$
便于书写，仍记 $\b{w}=(\b{w};b)$, $\b{x}=(\b{x};1)$. $\b{w}$ 的似然函数为
$$
L(\b{w})=\prod_{i=1}^{m}p(y=y_{i}|\b{x}_{i};\b{w})=\prod_{i=1}^{m}\frac{e^{y_{i}\cdot\b{w}\T\b{x}_{i}}}{1+e^{\b{w}\T\b{x}_{i}}}
$$
取负对数得到对数似然函数：
$$
l(\b{w})=-\ln L(\b{w})=\sum_{i=1}^{m}\left(-y_{i}\b{w}\T\b{x}_{i}+\ln\left(1+e^{\b{w}\T\b{x}_{i}}\right)\right)\tag{1}
$$
最大化似然函数等价于最小化 $l(\b{w})$, 可以使用数值优化算法如梯度下降法、牛顿法求得最优解：
$$
\b{w}^{*}=\mathop{\mathrm{arg\,min}}_{w}l(\b{w}).
$$
**梯度下降法：**
$$
\begin{aligned}
&\text{while}\, \|\nabla l(\b{w})\|>\delta\,\text{do}\\
&\quad \b{w}_{t+1}=\b{w}_{t}-\alpha\nabla l(\b{w})\\
&\text{end while}
\end{aligned}
$$
**牛顿法：**
$$
\begin{aligned}
&\text{while}\, \|\nabla l(\b{w})\|>\delta\,\text{do}\\
&\quad \b{w}_{t+1}=\b{w}_{t}-\left(\nabla^{2}l(\b{w})\right)^{-1}\nabla l(\b{w})\\
&\text{end while}
\end{aligned}
$$
由式 $(1)$易求得
$$
\begin{aligned}
\nabla l(\b{w}) &= \sum_{i=1}^{m}\left(p_{1}-y_{i}\right)\b{x}_{i}\\
\nabla^{2}l(\b{w}) &= \sum_{i=1}^{m}p_{1}(1-p_{1})\b{x}_{i}\b{x}_{i}\T
\end{aligned}
$$
其中
$$
p_{1}=\frac{1}{1+e^{-\b{w}\T\b{x}_{i}}}
$$


## 实验结果

根据上述原理，使用`python`编程实现了**梯度下降法**和**牛顿法**。

程序结构如下：

```python
class LogisticRegression(object):
    def __init__(self, 
                 learning_rate: float, 
                 max_iter: int,
                 fit_bias: Optional[bool] = True, 
                 optimizer: Literal['SGD', 'GD', 'mbSGD', 'Newton', None] = None,
                batch_size: Optional[int] = None,
                seed: Optional[int] = None):
        pass
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
    def predict_proba(self, X: np.ndarray):
        """
        用训练好的模型预测X的类别概率，也即对数几率函数的原始输出值
        """
        pass
    def score(self, 
              X: np.ndarray,
              target: np.ndarray,
              metric: Literal['err', 'acc', 'mse', 'rmse', 'f1'] = 'acc'):
        """
       	用模型预测X的标签并与真实标签target比较，计算评估函数值。
        """
        pass
    def plot_boundary(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      type: Literal['scatter'] = 'scatter'):
        """
        绘制决策边界
        """
        pass
    def plot_learning_curve(self):
        """
        绘制学习曲线
        """
        pass
```

在训练过程中，每训练一轮都计算出参数在训练集和测试集上的错误率，并且最后在训练集和测试集中画出训练出的曲线。

其中梯度下降法实现了不同版本的梯度下降法，分为：

- 随机梯度下降法(SGD)，每次使用一个样本进行梯度下降
- 梯度下降法(GD)，每次使用全部的样本进行梯度下降
- 小样本随机梯度下降法(mbSGD)，每次使用 `batch size` 个样本进行梯度下降

### SGD实验结果

参数设置为：

```python
LogisticRegression(learning_rate=0.05, max_iter=50, fit_bias=True, optimizer='SGD', seed=0)
```

绘制学习曲线如下：

![SGDfig1](.\fig\SGDfig1.png)

观察学习曲线，可以发现随着迭代次数的增加，训练集的误差一直下降，但是测试集的误差先下降后又上升，表明随着迭代次数的增加，训练发生了过拟合。训练大概$30$轮可以达到最好的效果。

下面将 **max_iter** 设置为 $30$ 绘制决策边界。

![SGDfig2](.\fig\SGDfig2.png)

![SGDfig3](.\fig\SGDfig3.png)

上面两张图片分别是训练出来的模型在训练集和测试集上的结果，可以发现模型在训练集和测试集上的表现都很好，其中在测试集上的正确率达到了 $100\%$.

### GD实验结果

参数设置为：

```python
LogisticRegression(learning_rate=0.05, max_iter=1000, fit_bias=True, optimizer='GD', seed=0)
```

绘制学习曲线如下：

![GDfig1](.\fig\GDfig1.png)

观察学习曲线，可以发现随着训练的进行，训练集和测试集的的错误率一直下降，直到训练次数足够多，就基本不再变化。每次使用全部的数据进行迭代，几乎没有发生过拟合的现象，但是需要迭代的轮数会增加，计算量较大。大概训练 $500$ 轮可以达到最好效果。

下面将 **max_iter** 设置为 $500$ 绘制决策边界。

![GDfig2](.\fig\GDfig2.png)

![GDfig3](.\fig\GDfig3.png)

上图为使用`GD`算法训练出的模型在训练集和测试集上的表现，可以发现在测试集上表现非常好，正确率达到了$100\%$。

### mbSGD实验结果

参数设置为：

```python
LogisticRegression(learning_rate=0.05, max_iter=300, fit_bias=True, optimizer='mbSGD', batch_size=10, seed=0)
```

绘制学习曲线如下：

![GDfig1](.\fig\mbSGDfig1.png)

设置**batch_size=10**，即每次使用七分之一的数据进行迭代。观察学习曲线，可以发现大概训练 $160$ 轮可以达到最佳效果。

下面将 **max_iter** 设置为 $160$ 绘制决策边界。

![GDfig2](.\fig\mbSGDfig2.png)

![GDfig3](.\fig\mbSGDfig3.png)

上图为使用`mbSGD`算法训练出的模型在训练集和测试集上的表现，可以发现在测试集上表现非常好，正确率达到了$100\%$ 。实验结果与使用`GD`训练出的结果几乎一样，但是迭代次数下降了许多，计算量下降了很多。

### Newton实验结果

参数设置为：

```python
LogisticRegression(learning_rate=1, max_iter=10, fit_bias=True, optimizer='Newton', seed=0)
```

绘制学习曲线如下：

![GDfig1](.\fig\NTfig1.png)

观察学习曲线可以发现牛顿法的效果非常好，只需要两步就收敛了，而且错误率也不高。但是牛顿法需要计算海森矩阵的逆，因此单步的计算复杂度比较高。

下面将 **max_iter** 设置为 $2$ 绘制决策边界。

![GDfig2](.\fig\NTfig2.png)

![GDfig3](.\fig\NTfig3.png)

上图为使用Newton法训练出来的模型在训练集和测试集上的表现，可以发现在训练集和测试集上的表现都可以。

## 实验总结

实现了多种实验方法，下面对这些方法进行总结。

| 实验方法 | 迭代轮数 | 训练集正确率 | 测试集正确率 |                         评价                         |
| :------: | :------: | :----------: | :----------: | :--------------------------------------------------: |
|   SGD    |    30    |    0.971     |    1.000     |            速度快，但是容易过拟合，不稳定            |
|    GD    |   500    |    0.943     |    1.000     |               速度慢，但是稳定，结果好               |
|  mbSGD   |   160    |    0.971     |    1.000     |         速度较快，兼有SGD和GD的优点，结果好          |
|  Newton  |    2     |    0.957     |    0.933     | 迭代轮数较少，但是需要计算海森矩阵的逆，计算复杂度高 |

