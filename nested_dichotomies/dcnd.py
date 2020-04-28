import numpy as np 
from .nd import NestedDichotomy, Node 
from scipy.spatial import distance
from copy import deepcopy



class DivisiveClusteringND(NestedDichotomy):
    def __init__(self, base_learner):
        super().__init__(base_learner = base_learner)
        
    def build_structure(self, X, y):
        n = len(np.unique(y))
        centroids = np.zeros((n, X.shape[1]))
        for i in range(n):
            centroids[i] = np.mean(np.array([x for j, x in enumerate(X) if y[j] == i]), axis=0)
        D = distance.squareform(distance.pdist(centroids))
        labels = np.arange(0,n)
        self.root = Node()
        self.root.classes = labels
        def _generate_split(node):
            if len(node.classes) == 1:
                return
            node.left = Node()
            node.right = Node()
            index = np.meshgrid(node.classes, node.classes, indexing = 'ij')
            _D = D[tuple(index)]
            c_max = np.unravel_index(_D.argmax(), _D.shape)
            l_group = [node.classes[c_max[0]]]
            r_group = [node.classes[c_max[1]]]
            for i in range(len(node.classes)):
                if i not in c_max:
                    if _D[c_max[0], i] <= _D[c_max[1], i]:
                        l_group.append(node.classes[i])
                    else:
                        r_group.append(node.classes[i])
            node.left.classes = l_group
            node.right.classes = r_group
        self.root.preorder(_generate_split)
    
    def set_models(self):
        def set_node_model(node):
            if not node.is_leaf():
                node.model = deepcopy(self.base_learner)
        self.root.preorder(set_node_model)
    
    def fit(self, X, y):
        self.build_structure(X, y)
        self.set_models()
        super().fit(X, y)
    
    # def predict_proba(self, X):
    #     super().predict_proba(X)
        
    # def predict(self, X):
    #     super().predict(X) 
        