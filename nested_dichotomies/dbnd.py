import numpy as np 
from .nd import NestedDichotomy, Node 
from copy import deepcopy



class DataBalancedND(NestedDichotomy):
    def __init__(self, base_learner):
        super().__init__(base_learner = base_learner)
        
    def build_structure(self, X, y):
        n = len(np.unique(y))
        labels = np.arange(0,n)
        self.root = Node()
        self.root.classes = list(labels)
        count_group = list()
        for i in labels:
            count_group.append(len(np.where(y==i)))
        def _generate_split(node):
            if len(node.classes) == 1:
                return
            l, r = np.random.choice(node.classes, size = 2, replace = False)
            l_group = [l]
            r_group = [r]
            total_count = sum([count_group[i] for i in node.classes])
            l_count = count_group[l]
            r_count = count_group[r]
            classes_remainding = node.classes[:]
            classes_remainding.remove(r)
            classes_remainding.remove(l)
            while len(classes_remainding)>0 and l_count <= np.floor(total_count/2):
                choice = np.random.choice(classes_remainding, size = 1)[0]
                l_group.append(choice)
                l_count += count_group[choice]
                classes_remainding.remove(choice)
            r_group = r_group + classes_remainding
            node.left = Node()
            node.right = Node()
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