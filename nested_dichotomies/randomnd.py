import numpy as np 
from .nd import NestedDichotomy, Node 
from copy import deepcopy



class RandomND(NestedDichotomy):
    def __init__(self, base_learner):
        super().__init__(base_learner = base_learner)
        
    def build_structure(self, X, y):
        n = len(np.unique(y))
        labels = np.arange(0,n)
        self.root = Node()
        self.root.classes = labels
        def _generate_split(node):
            if len(node.classes) == 1:
                return
            node.left = Node()
            node.right = Node()
            invalid_split = True
            while invalid_split:
                groups = np.random.rand(len(node.classes))
                l_group = node.classes[groups<0.5]
                r_group = node.classes[groups>=0.5]
                if len(l_group)>0 and len(r_group)>0:
                    invalid_split = False
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