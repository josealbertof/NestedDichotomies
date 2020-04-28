import numpy as np
from .nd import NestedDichotomy, Node
from copy import deepcopy

def distance(x,y):
    return sum((x-y)**2)

class AglomerativeClusteringND(NestedDichotomy):
    def __init__(self, base_learner):
        super().__init__(base_learner = base_learner)
        
    def build_structure(self, X, y):
        n = len(np.unique(y))
        labels = np.arange(0,n)
        nodes = [Node(classes = [i]) for i in labels]
        centroids = list(np.zeros((n, X.shape[1])))
        group_count = list()
        for i in range(n):
            centroids[i] = np.mean(np.array([x for j, x in enumerate(X) if y[j] == i]), axis=0)
            group_count.append(len(np.where(y==i)))
        D = np.zeros((n,n))+np.iinfo(np.int32).max
        for i in np.arange(n-1):
            for j in np.arange(n-i-1)+i+1:
                D[i,j] = distance(centroids[i], centroids[j])
        while len(nodes)>1:
            min_ind = sorted(np.unravel_index(D.argmin(), D.shape))
            super_group = nodes[min_ind[0]].classes+nodes[min_ind[1]].classes
            super_node = Node(classes = super_group)
            super_node.left = nodes[min_ind[0]]
            super_node.right = nodes[min_ind[1]]
            nodes.append(super_node)
            nodes = nodes[:min_ind[0]]+nodes[(min_ind[0]+1):min_ind[1]]+nodes[(min_ind[1]+1):]
            # The only work left is to update the distance matrix
            D = np.hstack((D, np.ones((D.shape[0],1))*np.iinfo(np.int32).max))
            group_count.append(group_count[min_ind[0]]+group_count[min_ind[1]])
            centroids.append((centroids[min_ind[0]]*group_count[min_ind[0]]+centroids[min_ind[1]]*group_count[min_ind[1]])/group_count[-1])
            for i in range(D.shape[0]):
                if i not in min_ind:
                    D[i,-1] = distance(centroids[i],centroids[-1])
                    
            group_count = group_count[:min_ind[0]]+group_count[(min_ind[0]+1):min_ind[1]]+group_count[(min_ind[1]+1):]
            centroids = centroids[:min_ind[0]]+centroids[(min_ind[0]+1):min_ind[1]]+centroids[(min_ind[1]+1):]
            _labels = list(np.arange(D.shape[1]))
            _labels.remove(min_ind[0])
            _labels.remove(min_ind[1])
            D = D[:,_labels]
            D = D[_labels[:-1],:]
        self.root = nodes[0]
            
    def set_models(self):
        def set_node_model(node):
            if not node.is_leaf():
                node.model = deepcopy(self.base_learner)
        self.root.preorder(set_node_model)
    
    def fit(self, X, y):
        self.build_structure(X, y)
        self.set_models()
        super().fit(X, y)