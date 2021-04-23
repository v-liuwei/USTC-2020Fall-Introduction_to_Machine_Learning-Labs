# 实验五——Latent Dirichlet Allocation

姓名：刘威

学号：PB18010469
$$
\newcommand{\b}{\boldsymbol}
$$


## 实验目的

+ 理解Latent Dirichlet Allocation模型的思想；
+ 理解Gibbs采样在近似采样中的应用；
+ 实现LDA模型的Gibbs采样近似算法。


## 实验原理

隐狄利克雷分配模型（Latent Dirichlet Allocation, LDA）是话题模型（topic model）的典型代表。

### LDA 模型的基本原理

**LDA 的基本单元：**

+ <font color='green'>词（word）</font>：待处理数据中的基本离散单元
+ <font color='red'>文档（document）</font>：待处理的数据对象，由词组成，词在文档中不计顺序。数据对象只要能用“词袋”（bag-of-words）表示就可以使用话题模型
+ <font color='blue'>话题（topic）</font>：表示一个概念，具体表示为一系列相关的词，以及它们在该概念下出现的概率

**LDA 模型的描述：**

假定数据集中共含 $K$ 个话题和 $M$ 篇文档，词来自含 $V$ 个词的字典：

+ $V$ 个词的字典 $\mathcal{V}=\{w_{1},w_{2},\dots,w_{V}\}$。

+ $M$ 篇文档 $\b{W}=\{\b{w}_{1},\b{w}_{2},\dots,\b{w}_{M}\}$。第 $m$ 篇文档由长度为 $N_{m}$ 的单词序列组成，其单词序列为 $\b{w}_{m}=(w_{m1},\dots,w_{mN_{m}})$ 。

  文档可以表示为话题的分布，即用长度为 $K$ 概率话题向量表示。第 $m$ 篇文档的概率话题向量为 $\b{\theta}_{m}\in[0,1]^{K}$, 其中 $\theta_{mk}=P(z_{k}|\b{w}_{m})$ 表示第 $m$ 篇文档中话题 $z_{k}$ 的概率。

+ $K$ 个话题 $\mathcal{Z}=\{z_{1},z_{2},\dots,z_{K}\}$。

  话题可以表示为词的分布，即用长度为 $V$ 的概率词向量表示。第 $k$ 个话题的概率词向量为 $\b{\varphi}_{k}\in{[0,1]^{V}}$, 其中 $\varphi_{kv}=P(w_{v}|z_{k})$ 表示第 $k$ 个话题中单词 $w_{v}$ 的概率。

+ 隐变量 $\b{Z}=\{\b{z}_{1},\b{z}_{2},\dots,\b{z}_{M}\}$, 与文档 $\b{W}$ 对应，其中 $z_{mn}$ 表示单词 $w_{mn}$ 的话题。 

**LDA模型的生成过程：**

==生成主题 $k$ 的过程：==

+ 从以 $\b{\beta}$ 为参数的狄利克雷分布中随机采样一个词分布 $\b{\varphi}_{k}$.

==生成文档 $d$ 的过程：==

+ 从以 $\b{\alpha}$ 为参数的狄利克雷分布中随机采样一个话题分布 $\b{\theta}_{m}$;
+ 按如下步骤产生文档中的 $N_{m}$ 个词:
  + 根据 $\b{\theta}_{m}$ 进行话题指派，得到文档 $m$ 中的第 $n$ 个词的话题 $z_{mn}$;
  + 根据话题 $z_{mn}$ 所对应的词分布 $\b{\varphi}_{k}$ 随机采样生成词 $w_{mn}$. 

![LDA](LDA.png)

上图描述了LDA的变量关系，其中文档集 $\b{W}$ 是唯一的已观测变量，它依赖于话题指派矩阵 $\b{Z}$，以及话题所对应的词分布矩阵 $\b{\varphi}$; 同时话题指派矩阵 $\b{Z}$ 依赖于话题分布矩阵 $\b{\theta}$。而 $\b{\varphi}$ 依赖于狄利克雷分布的参数 $\b{\beta}$, $\b{\theta}$ 依赖于狄利克雷分布的参数 $\b{\alpha}$.

**LDA模型的形式化表示：**

由上述变量关系，LDA 模型对应的概率分布为
$$
p(\b{W},\b{Z},\b{\varphi},\b{\theta}\mid \b{\alpha},\b{\beta})=\prod_{m=1}^{M}p(\b{\theta}_{m}\mid\b{\alpha})\prod_{k=1}^{K}p(\b{\varphi}_{k}\mid\b{\beta})\left(\prod_{n=1}^{N_{m}}p(w_{mn}\mid z_{mn},\b{\varphi}_{k})p(z_{mn}\mid\b{\theta}_{m})\right)
$$
其中$p(\b{\theta}_{m}\mid\b{\alpha}),p(\b{\varphi}_{k}\mid\b{\beta})$ 分别为以 $\b{\alpha}$ 和 $\b{\beta}$ 为参数的 $K$ 维和 $V$ 维狄利克雷分布，例如
$$
p(\b{\theta}_{m}\mid\b{\alpha})=\frac{\Gamma(\sum_{k}\alpha_{k})}{\prod_{k}\Gamma(\alpha_{k})}\prod_{k}\theta_{mk}^{\alpha_{k}-1},
$$
其中 $\Gamma(\cdot)$ 是 Gamma 函数。显然，$\b{\alpha}$ 和 $\b{\beta}$ 是模型中待确定的参数。

### LDA模型的学习与推断

给定训练数据 $\b{W}=\{\b{w}_{1},\b{w}_{2},\dots,\b{w}_{M}\}$, LDA的模型参数可以通过极大似然法估计，即寻找 $\b{\alpha}$ 和 $\b{\beta}$ 以最大化对数似然
$$
LL(\b{\alpha},\b{\beta})=\sum_{m=1}^{M}\ln p(\b{w}_{m}\mid\b{\alpha},\b{\beta}).
$$
但由于 $p(\b{w}_{m}\mid\b{\alpha},\b{\beta})$ 不易计算，上式难以直接求解，因此实践中常采用变分法来求取近似解.

若模型已知，即参数 $\b{\alpha}$ 和 $\b{\beta}$ 已确定，则根据 $\b{W}$ 来推断文档集所对应的话题结构（即推断 $\b{\theta},\b{\varphi},\b{Z}$）可通过求解
$$
p(\b{Z},\b{\varphi},\b{\theta}\mid \b{W},\b{\alpha},\b{\beta})=\frac{p(\b{W},\b{Z},\b{\varphi},\b{\theta}\mid \b{\alpha},\b{\beta})}{p(\b{W}\mid \b{\alpha,\b{\beta}})}.
$$
然而由于分母难以计算，上式难以直接求解，因此在实践中常采用吉布斯采样或变分法进行近似推断。

本实验使用吉布斯采样完成LDA模型的近似推断。

**吉布斯采样算法思想:**

+ 对隐变量 $\b{\theta},\b{\varphi}$积分，得到边缘概率 $p(\b{Z}\mid\b{W},\b{\alpha},\b{\beta})$;
+ 对后验概率进行吉布斯抽样，得到分布 $p(\b{Z}\mid\b{W},\b{\alpha},\b{\beta})$ 的样本集合;
+ 利用这个样本集合对参数 $\b{\theta}$ 和 $\b{\varphi}$ 进行参数估计.

略去详细推导过程，上述步骤的关键结果如下：

+ 对隐变量 $\b{\theta},\b{\varphi}$积分，得到边缘概率 $p(\b{Z}\mid\b{W},\b{\alpha},\b{\beta})$;
  $$
  p(\b{Z}\mid\b{W},\b{\alpha},\b{\beta})\propto\prod_{m=1}^{M}\frac{\Beta(\b{\alpha}+\b{\sigma}_{m})}{\Beta(\b{\alpha})}\prod_{k=1}^{K}\frac{\Beta(\b{\beta}+\b{\delta}_{k})}{\Beta(\b{\beta})}.
  $$
  其中 $\Beta(\cdot)$ 为 Beta 函数，${\sigma}_{mk}=\sum_{n=1}^{N_{m}}\mathbb{I}(z_{mn}=k)$ 表示第 $m$ 篇文档中第 $k$ 个话题的词的频数, $n_{kv}=\sum_{m=1}^{M}\sum_{n=1}^{N_{m}}\mathbb{I}(w_{mn}=v)\mathbb{I}(z_{mn}=k)$ 表示所有文档中第 $k$ 个话题下词 $w_{v}$ 出现的频数。

+ 对后验概率进行吉布斯抽样，得到分布 $p(\b{Z}\mid\b{W},\b{\alpha},\b{\beta})$ 的样本集合;

  吉布斯采样概率为：
  $$
  p(z_{mn}=k\mid\b{Z}_{\neg mn},\b{W},\b{\alpha},\b{\beta})\propto
  \frac{\sigma_{mk}^{\neg mn}+\alpha_{k}}{\sum_{k=1}^{K}\sigma_{mk}+\alpha_{k}}\cdot\frac{\delta_{kw_{mn}}^{\neg mn}+\beta_{w_{mn}}}{\sum_{v=1}^{V}\delta_{kv}^{\neg mn}+\beta_{v}}
  $$
  其中
  $$
  \sigma_{ji}^{\neg mn}=\begin{cases}
  \sigma_{ji}-1 & j=m\and i=z_{mn},\\
  \sigma_{ji} & \text{otherwise.}
  \end{cases}\\
  \delta_{ir}^{\neg mn}=\begin{cases}
  \delta_{ir}-1 & i=z_{mn} \and r=w_{mn},\\
  \delta_{ir} & \text{otherwise.}
  \end{cases}
  $$

+ 利用这个样本集合对参数 $\b{\theta}$ 和 $\b{\varphi}$ 进行参数估计.
  $$
  \theta_{mk}\approx\frac{\sigma_{mk}+\alpha_{k}}{\sum_{k=1}^{K}\sigma_{mk}+\alpha_{k}}\\
  \varphi_{kv}\approx\frac{\delta_{kv}+\beta_{v}}{\sum_{v=1}^{V}\delta_{kv}+\beta_{v}}
  $$

**吉布斯采样算法：**

1. Initialization

   $\sigma_{mk}=0,\sigma_{m}=0,\delta_{nv}=0,\delta_{n}=0.$

   $\text{for }m\in\{1,2,\dots,M\}\text{ do}:$

   ​	$\text{for }n\in\{1,2,\dots,N_{m}\}\text{ do}:$

   ​		$v=w_{mn},\text{sample }z_{mn}=k\sim\text{Cat}(\frac{1}{K})$

   ​		$\sigma_{mk}=\sigma_{mk}+1,\sigma_{m}=\sigma_{m}+1,$

   ​		$\delta_{kv}=\delta_{kv}+1,\delta_{k}=\delta_{k}+1$

2. Gibbs Sampling

   $\text{while not finished do:}$

   ​	$\text{for }m\in\{1,2,\dots,M\}\text{ do:}$

   ​		$\text{for }n\in\{1,2,\dots,N_{m}\}\text{ do}:$

   ​			$v=w_{mn},k=z_{mn}$

   ​			$\sigma_{mk}=\sigma_{mk}-1,\sigma_{m}=\sigma_{m}-1,$

   ​			$\delta_{kv}=\delta_{kv}-1,\delta_{k}=\delta_{k}-1$

   ​			$\text{sample }z_{mn}=\tilde{k}\sim p(z_{mn}=k\mid\b{Z}_{\neg mn},\b{W},\b{\alpha},\b{\beta})$

   ​			$\sigma_{m\tilde{k}}=\sigma_{m\tilde{k}}-1,\sigma_{m}=\sigma_{m}-1,$

   ​			$\delta_{\tilde{k}v}=\delta_{\tilde{k}v}-1,\delta_{\tilde{k}}=\delta_{\tilde{k}}-1$

3. Parameter estimation

   $\text{for }m\in\{1,2,\dots,M\}\text{ do}:$

   ​	$\text{for }k\in\{1,2,\dots,K\}\text{ do}:$

   ​		$\theta_{mk}=\frac{\sigma_{mk}+\alpha_{k}}{\sum_{k=1}^{K}\sigma_{mk}+\alpha_{k}}$

   $\text{for }k\in\{1,2,\dots,K\}\text{ do}:$

   ​	$\text{for }v\in\{1,2,\dots,V\}\text{ do}:$

   ​		$\varphi_{kv}=\frac{\delta_{kv}+\beta_{v}}{\sum_{v=1}^{V}\delta_{kv}+\beta_{v}}$


## 实验要求

给定包含1000篇随机抽取的英文新闻，在文件`text.npy`中，具体要求如下：

+ 这些新闻采自20个不同的主题，利用LDA模型和吉布斯采样算法，给出这20个主题相关的top10关键词（按照概率大小排序）

+ 文本预处理的操作（删除标点，去除停用词，转化为小写等基本操作可以调用其他库函数）

+ LDA模型和吉布斯采样需要自己实现

+ 提交的实验报告中，写清楚选择的参数，并附上20个主题的top10的词，与下图类似：

  ![image-20210210203355554](C:\Users\liuwei\AppData\Roaming\Typora\typora-user-images\image-20210210203355554.png)

## 实验结果

### 程序结构

程序接口及说明如下：

```python
class LatentDirichletAllocation(object):
    """Latent dirichlet allocation(LDA) with collapsed Gibbs sampling algorithm.

       Symbol description of LDA is as follows:
                 -----------------------
                 |          ---------- |   -------
        alpha ---> theta -> | z -> w | <---| phi | <- beta
                 |          ---------- |   -------
                 -----------------------

    Parameters
    ----------
    n_topics: int, default = 10
        Number of topics.
    doc_topic_prior: float, default = None
        Prior of document topic distribution `alpha`. If the value is None, defaults to 1 / n_topics.
    topic_word_prior: float, default = None
        Prior of topic word distribution `beta`. If the value is None, defaults to 1 / n_topics.
    max_iter: int, default = 10
        The maximum number of iterations.
    random_state: int, default = None
        Random seed.

    Attributes
    ----------
    topic_word_distr: ndarray of shape (n_topics, n_words)
        Topic word distribution.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from model import LatentDirichletAllocation
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> X, _ = make_multilabel_classification(random_state=0)
    >>> X = X.astype(int)
    >>> lda = LatentDirichletAllocation(n_topics=5, random_state=0)
    >>> lda.fit_transform(X)[-2:]
    array([[0.00350877, 0.14385965, 0.31929825, 0.52982456, 0.00350877],       
       [0.00350877, 0.00350877, 0.38947368, 0.00350877, 0.6       ]])
    """
    def __init__(self, n_topics=10, doc_topic_prior=None, topic_word_prior=None, max_iter=10, random_state=None):
        pass
    def fit_transform(self, X):
        """Fit to data, then transform it.

        Parameters
        ----------
        X: array_like of shape (n_docs, n_words)
            Document word matrix.

        Returns
        -------
        doc_topic_distr: ndarray of shape (n_docs, n_topics)
            Document topic distribution for X.
        """
        pass
    def fit(self, X):
        """Learn model for the data X

        Parameters
        ----------
        X : array_like of shape (n_docs, n_words)
            Document word matrix.

        Returns
        -------
        self
        """
        pass
    def transform(self, X):
        """Transform data X according to the fitted model.

        Parameters
        ----------
        X: array_like of shape (n_docs, n_words)
            Document word matrix.

        Returns
        -------
        doc_topic_distr: ndarray of shape (n_docs, n_topics)
            Document topic distribution for X.
        """
        pass
```


### 数据处理

模型要求输入文档词矩阵，即词库中的词在各个文档中出现的频数。因此需要先对数据进行预处理。

**数据加载：**

```python
# load and preprocess data, save result
if not os.path.exists('vocab.npy'):
    docs = np.load('text.npy')
    vocab, doc_word_mat = get_vocab_and_doc_word_mat(docs)
    np.save('vocab.npy', vocab)
    np.save('doc_word_mat.npy', doc_word_mat)
else:
    vocab = np.load('vocab.npy')
    doc_word_mat = np.load('doc_word_mat.npy')
```

其中函数 `get_vocab_and_doc_word_mat` 包含所有的文本预处理过程，返回词库 `vocab` 和 文档词矩阵 `doc_word_mat`. 文本预处理的详细说明如下

**文本预处理：**

文本预处理使用自然语言处理库 `nltk` 完成，具体步骤包括：

+ 转化为小写、去除无关字符、展开缩写

  ```python
  def replace_abbreviations(text):
      text = text.lower()
      # 只保留字母、空格、单引号(所有格及is缩写)
      text = re.sub(r'[^a-z \']+', ' ', text)
      text = re.sub("(it|he|she|that|this|there|here)(\'s)", r"\1 is", text)  # is 缩写
      text = re.sub("(?<=[a-z])\'s", "", text)  # 所有格
      text = re.sub("(?<=s)\'s?", "", text)  # 复数所有格
      text = re.sub("(?<=[a-z])n\'t", " not", text)  # not 缩写
      text = re.sub("(?<=[a-z])\'d", " would", text)  # would 缩写
      text = re.sub("(?<=[a-z])\'ll", " will", text)  # will 缩写
      text = re.sub("(?<=i)\'m", " am", text)  # am 缩写
      text = re.sub("(?<=[a-z])\'re", " are", text)  # are 缩写
      text = re.sub("(?<=[a-z])\'ve", " have", text)  # have 缩写
      text = text.replace('\'', ' ')  # 剩下的单引号去掉
      return text
  ```

+ 词性还原

  ```python
  lmtzr = WordNetLemmatizer()
  def lemmatize(word):
      def get_wordnet_pos(treebank_tag):
          if treebank_tag.startswith('J'):
              return nltk.corpus.wordnet.ADJ
          elif treebank_tag.startswith('V'):
              return nltk.corpus.wordnet.VERB
          elif treebank_tag.startswith('N'):
              return nltk.corpus.wordnet.NOUN
          elif treebank_tag.startswith('R'):
              return nltk.corpus.wordnet.ADV
          else:
              return ''
      tag = nltk.pos_tag(word_tokenize(word))
      pos = get_wordnet_pos(tag[0][1])
      if pos:
          word = lmtzr.lemmatize(word, pos)
      return word
  ```

+ 去除停用词

  ```python
  from nltk.corpus import stopwords
  from string import punctuation
  blacklist = stopwords.words('english') + ['would', 'may', 'might', 'could', 'shall']
  blacklist.extend(list('bcdefghjklmnopqrstuvwxyz'))
  blacklist.extend(list(punctuation))
  if word not in blacklist:  # 去除停用词
  	new_words.append(word)
  ```

+ 词频统计

  ```python
  from collections import Counter
  doc_word_mat = np.zeros((len(doc_words), len(vocab)), dtype=np.int)
  for d in range(len(doc_words)):
      counter = Counter(doc_words[d])
      for t in range(len(vocab)):
          doc_word_mat[d][t] = counter[vocab[t]] if vocab[t] in counter else 0
  ```

+ 去除低频词

  大部分词只出现过一次两次，这使得我们地词库非常大。去除一些低频词可以缩小词库，减小计算量。

  ```python
  # remove low frequent words
  threshold = 1
  selected_words = doc_word_mat.sum(axis=0) > threshold
  vocab = vocab[selected_words]
  doc_word_mat = doc_word_mat[:, selected_words]
  ```

  这里设置了有个词频阈值`threshold`， 频数低于或等于这个阈值的词会被删去。阈值越大，词库越小，计算效率越高，但可能会导致一些话题关键词被移除。因此这个参数也是值得调节的。

### 模型结果

为使得吉布斯采样充分收敛，取 `max_iter=100`，取 `threshold` 分别为 $1,5,20$, 其它参数设置为`n_topics=20,random_state=123`, 得到的各话题Top10词如下：

+ `threshold=1`(耗时351s):

  ```python
  ['graphic', 'image', 'program', 'software', 'also', 'include', 'system', 'format', 'object', 'use']
  ['key', 'government', 'chip', 'encryption', 'gm', 'use', 'clipper', 'system', 'win', 'law']        
  ['weapon', 'section', 'firearm', 'state', 'use', 'gun', 'military', 'person', 'carry', 'license']  
  ['window', 'use', 'help', 'application', 'program', 'file', 'please', 'manager', 'try', 'mail']    
  ['people', 'like', 'think', 'know', 'get', 'one', 'make', 'use', 'right', 'say']
  ['point', 'book', 'line', 'use', 'information', 'find', 'way', 'change', 'plane', 'method']
  ['space', 'april', 'test', 'rocket', 'star', 'technology', 'science', 'th', 'spacecraft', 'km']
  ['magi', 'food', 'know', 'time', 'get', 'well', 'msg', 'period', 'starter', 'win']
  ['israel', 'greek', 'kill', 'israeli', 'attack', 'jew', 'state', 'turkish', 'population', 'arab']
  ['car', 'power', 'get', 'use', 'one', 'like', 'well', 'light', 'two', 'drive']
  ['year', 'car', 'also', 'go', 'insurance', 'us', 'think', 'com', 'course', 'edu']
  ['hiv', 'aid', 'health', 'disease', 'trial', 'care', 'child', 'patient', 'say', 'medical']
  ['edu', 'com', 'file', 'pub', 'mail', 'server', 'send', 'ftp', 'ray', 'archive']
  ['space', 'copy', 'launch', 'earth', 'shuttle', 'flight', 'mission', 'us', 'orbit', 'probe']
  ['get', 'work', 'go', 'know', 'new', 'back', 'thing', 'one', 'use', 'call']
  ['game', 'team', 'go', 'play', 'fan', 'well', 'blue', 'goal', 'score', 'good']
  ['card', 'drive', 'system', 'use', 'monitor', 'xfree', 'work', 'thanks', 'mb', 'support']
  ['go', 'one', 'get', 'year', 'say', 'people', 'come', 'see', 'armenian', 'like']
  ['church', 'billion', 'national', 'president', 'group', 'dollar', 'security', 'general', 'increase', 'paper']
  ['god', 'say', 'make', 'one', 'people', 'think', 'good', 'life', 'also', 'even']
  ```

+ `threshold=5`(耗时305s):

  ```python
  ['support', 'file', 'server', 'edu', 'xfree', 'please', 'mail', 'list', 'include', 'post']    
  ['space', 'earth', 'data', 'launch', 'us', 'mission', 'shuttle', 'use', 'year', 'system']     
  ['system', 'program', 'book', 'point', 'one', 'reference', 'since', 'use', 'know', 'define']  
  ['year', 'think', 'car', 'like', 'get', 'etc', 'make', 'us', 'see', 'money']
  ['israel', 'attack', 'kill', 'people', 'state', 'jew', 'israeli', 'government', 'even', 'say']
  ['window', 'use', 'software', 'time', 'color', 'work', 'display', 'application', 'help', 'also']
  ['make', 'get', 'one', 'good', 'go', 'want', 'right', 'need', 'ca', 'even']
  ['game', 'play', 'well', 'team', 'go', 'fan', 'goal', 'blue', 'score', 'second']
  ['key', 'government', 'use', 'weapon', 'law', 'chip', 'firearm', 'encryption', 'section', 'clipper']
  ['god', 'people', 'say', 'make', 'jesus', 'one', 'church', 'christian', 'life', 'believe']
  ['armenian', 'one', 'th', 'old', 'home', 'come', 'get', 'go', 'group', 'around']
  ['one', 'like', 'use', 'get', 'well', 'go', 'thing', 'know', 'problem', 'good']
  ['edu', 'graphic', 'com', 'pub', 'mail', 'image', 'send', 'file', 'ray', 'object']
  ['gm', 'win', 'bike', 'com', 'helmet', 'motorcycle', 'new', 'main', 'john', 'contact']
  ['greek', 'copy', 'new', 'turkish', 'year', 'state', 'turk', 'sell', 'old', 'cover']
  ['know', 'get', 'one', 'try', 'please', 'time', 'help', 'anyone', 'tell', 'come']
  ['go', 'say', 'people', 'look', 'point', 'get', 'first', 'left', 'like', 'take']
  ['hiv', 'health', 'aid', 'disease', 'new', 'say', 'child', 'trial', 'billion', 'national']
  ['power', 'control', 'woman', 'one', 'want', 'men', 'state', 'signal', 'way', 'wire']
  ['drive', 'card', 'use', 'monitor', 'mb', 'problem', 'bit', 'scsi', 'need', 'system']
  ```

+ `threshold=20`(耗时212s):

  ```python
  ['graphic', 'image', 'pub', 'mail', 'file', 'ray', 'format', 'data', 'program', 'send']
  ['like', 'one', 'get', 'good', 'use', 'well', 'make', 'give', 'look', 'first']
  ['win', 'time', 'system', 'bit', 'mhz', 'one', 'main', 'cache', 'new', 'number']
  ['year', 'two', 'use', 'also', 'make', 'high', 'possible', 'problem', 'last', 'one']
  ['hiv', 'space', 'aid', 'disease', 'health', 'new', 'april', 'study', 'earth', 'number']
  ['window', 'use', 'file', 'card', 'drive', 'driver', 'run', 'problem', 'mb', 'get']
  ['god', 'jesus', 'us', 'woman', 'come', 'make', 'men', 'good', 'life', 'child']
  ['key', 'government', 'use', 'chip', 'encryption', 'clipper', 'system', 'phone', 'make', 'right']
  ['please', 'thanks', 'anyone', 'post', 'know', 'mail', 'edu', 'reply', 'current', 'light']
  ['edu', 'com', 'server', 'xfree', 'list', 'gm', 'faq', 'information', 'support', 'message']
  ['know', 'get', 'like', 'well', 'go', 'one', 'see', 'work', 'people', 'thing']
  ['israel', 'state', 'kill', 'attack', 'people', 'israeli', 'gun', 'control', 'government', 'village']
  ['greek', 'magi', 'turkish', 'first', 'history', 'source', 'turk', 'new', 'jew', 'use']
  ['point', 'go', 'say', 'one', 'get', 'line', 'even', 'make', 'thing', 'back']
  ['use', 'one', 'data', 'read', 'two', 'return', 'true', 'display', 'color', 'software']
  ['people', 'say', 'think', 'see', 'know', 'well', 'time', 'believe', 'like', 'make']
  ['car', 'get', 'copy', 'price', 'one', 'year', 'order', 'think', 'pay', 'new']
  ['go', 'armenian', 'well', 'start', 'first', 'people', 'day', 'time', 'say', 'get']
  ['game', 'play', 'team', 'year', 'make', 'fan', 'goal', 'second', 'go', 'win']
  ['section', 'firearm', 'weapon', 'military', 'person', 'use', 'license', 'carry', 'division', 'issue']
  ```


## 实验分析与总结

本次实验是此课程所有五个实验（前四个实验分别为LR、SVM、K-means、XGBoost）中理论最为复杂的一个，尽管不是代码量最大的。虽然课上老师有仔细讲LDA的原理及算法，但初次接触话题生成模型，很多概念尚不理解，算法更是听的云里雾里。因此本次实验写代码之前看了很多相关的文章和博客，才把LDA及相关的吉布斯采样算法理解清楚，前期准备部分比写代码的时间要长很多。

在实现之前，或者说刚了解LDA的时候，心里总会有这样的疑问: LDA模型结构这么简单，而文档的结构相对复杂得多，使用 LDA 真的能较好地提取到文档中的话题吗？经过此次实验，心中的疑惑也消散了，LDA不仅能够提取出文档中的话题，而且效果很惊人。

在给定的数据中，本实验实现的LDA模型得到的结果看起来就相当不错，大多数话题中Top10词的相关性很大，话题性很明显。比如在 `threshold=1`的结果中：

`['weapon', 'section', 'firearm', 'state', 'use', 'gun', 'military', 'person', 'carry', 'license'] `是关于战争的；

`['space', 'april', 'test', 'rocket', 'star', 'technology', 'science', 'th', 'spacecraft', 'km']`是关于太空航行的；

`['israel', 'greek', 'kill', 'israeli', 'attack', 'jew', 'state', 'turkish', 'population', 'arab']`是关于中东关系的；

`['hiv', 'aid', 'health', 'disease', 'trial', 'care', 'child', 'patient', 'say', 'medical']`是关于医疗与健康的；

`['edu', 'com', 'file', 'pub', 'mail', 'server', 'send', 'ftp', 'ray', 'archive']`是关于计算机网络传输的；

等等。

另外，可以看到当`threshold`增大时，有一些话题的部分top10关键词不见了，因为词频太低被删去了，取而代之的是一些之前排在top10之后的词。

经过本次实验，我对LDA的了解算是从入门到“精通”了，不论是模型层面，还是算法层面都有了较为清晰的认识。但此次实验仅仅是实现了吉布斯采样算法，这是一个近似算法，并且是在已知参数 $\b{\alpha},\b{\beta}$ 的情况下进行的。若想要推断这两个参数，还要使用更为复杂的算法。另外，在某些场合下，文档是在线生成的，还需要一些在线的算法去进行模型的学习与推断，当然，这些已经不是本次实验需要考虑的了，想要真正精通LDA仍有许多内容需要学习。





