import numpy as np
from tqdm import tqdm


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
    def __init__(self,
                 n_topics=10,
                 doc_topic_prior=None,
                 topic_word_prior=None,
                 max_iter=10,
                 random_state=None):
        self.n_topics = n_topics
        self.alpha = doc_topic_prior
        self.beta = topic_word_prior
        self.max_iter = max_iter
        self.seed = random_state

        self.topic_word_distr = None

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
        np.random.seed(self.seed)
        alpha, beta, doc_count, doc_topic_count, topic_count, topic_word_count, z = self.__init(X)
        doc_count, doc_topic_count, topic_count, topic_word_count = self.__gibbs_sampling(X, alpha, beta,
                                                                                          doc_count, doc_topic_count,
                                                                                          topic_count, topic_word_count,
                                                                                          z, fix_phi=False)
        doc_topic_distr, self.topic_word_distr = self.__estimate_params(alpha, beta, doc_count, doc_topic_count,
                                                                        topic_count, topic_word_count)
        return doc_topic_distr

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
        np.random.seed(self.seed)
        alpha, beta, doc_count, doc_topic_count, topic_count, topic_word_count, z = self.__init(X)
        doc_count, doc_topic_count, topic_count, topic_word_count = self.__gibbs_sampling(X, alpha, beta,
                                                                                          doc_count, doc_topic_count,
                                                                                          topic_count, topic_word_count,
                                                                                          z, fix_phi=False)
        _, self.topic_word_distr = self.__estimate_params(alpha, beta, None, None, topic_count, topic_word_count)
        return self

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
        if self.topic_word_distr is None:
            raise Exception("The model should be fitted first.")
        alpha, beta, doc_count, doc_topic_count, topic_count, topic_word_count, z = self.__init(X)
        doc_count, doc_topic_count, topic_count, topic_word_count = self.__gibbs_sampling(X, alpha, beta,
                                                                                          doc_count, doc_topic_count,
                                                                                          topic_count, topic_word_count,
                                                                                          z, phi=self.topic_word_distr,
                                                                                          fix_phi=True)
        doc_topic_distr, _ = self.__estimate_params(alpha, beta, doc_count, doc_topic_count, None, None)
        return doc_topic_distr

    def __init(self, X):
        n_docs, n_words = X.shape
        if self.alpha is not None:
            alpha = np.ones(self.n_topics) * self.alpha
        else:
            alpha = np.ones(self.n_topics) * 1 / self.n_topics
        if self.beta is not None:
            beta = np.ones(n_words) * self.beta
        else:
            beta = np.ones(n_words) * 1 / self.n_topics

        doc_count = np.zeros(n_docs, dtype=np.int)
        doc_topic_count = np.zeros((n_docs, self.n_topics), dtype=np.int)
        topic_count = np.zeros(self.n_topics, dtype=np.int)
        topic_word_count = np.zeros((self.n_topics, n_words), dtype=np.int)
        Nmax = np.max(np.sum(X, axis=1))
        z = np.zeros((n_docs, Nmax), dtype=np.int)
        z_distr = 1./self.n_topics*np.ones(self.n_topics)
        for m in range(n_docs):
            n = 0
            for t in X[m].nonzero()[0]:
                for _ in range(X[m][t]):
                    k = np.random.choice(self.n_topics, p=z_distr)
                    z[m][n] = k
                    doc_count[m] += 1
                    doc_topic_count[m][k] += 1
                    topic_count[k] += 1
                    topic_word_count[k][t] += 1
                    n += 1
        return alpha, beta, doc_count, doc_topic_count, topic_count, topic_word_count, z

    def __gibbs_sampling(self, X, alpha, beta, doc_count, doc_topic_count,
                         topic_count, topic_word_count, z, phi=None, fix_phi=False):
        n_docs, n_words = X.shape
        beta_sum = beta.sum()
        finished = False
        for i in tqdm(range(self.max_iter)):
            for m in range(n_docs):
                n = 0
                for t in X[m].nonzero()[0]:
                    for _ in range(X[m][t]):
                        k = z[m][n]
                        doc_count[m] -= 1
                        doc_topic_count[m][k] -= 1
                        topic_count[k] -= 1
                        topic_word_count[k][t] -= 1
                        if not fix_phi:
                            z_distr = (doc_topic_count[m] + alpha) * \
                                      (topic_word_count[:, t] + beta[t]) / (topic_count + beta_sum)
                        else:
                            z_distr = (doc_topic_count[m] + alpha) * phi
                        z_distr /= np.sum(z_distr)
                        k_new = np.random.choice(self.n_topics, p=z_distr)
                        z[m][n] = k_new
                        doc_count[m] += 1
                        doc_topic_count[m][k_new] += 1
                        topic_count[k_new] += 1
                        topic_word_count[k_new][t] += 1
                        n += 1
            if finished:
                break
        return doc_count, doc_topic_count, topic_count, topic_word_count

    def __estimate_params(self, alpha, beta, doc_count, doc_topic_count, topic_count, topic_word_count):
        theta = phi = None
        if doc_topic_count is not None:
            alpha_sum = alpha.sum()
            theta = ((alpha + doc_topic_count).T / (alpha_sum + doc_count)).T
        if topic_word_count is not None:
            beta_sum = beta.sum()
            phi = ((beta + topic_word_count).T / (beta_sum + topic_count)).T
        return theta, phi
