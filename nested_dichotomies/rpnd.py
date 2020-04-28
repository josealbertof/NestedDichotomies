import numpy as np 
from .nd import Node, NestedDichotomy
from copy import deepcopy


class RandomPairND(NestedDichotomy):
    def __init__(self, base_learner):
        super().__init__(base_learner = base_learner)
        
    def build_structure(self, X, y):
        n = len(np.unique(y))
        labels = np.arange(0,n)
        self.root = Node(model = deepcopy(self.base_learner))
        self.root.classes = labels
        def _generate_split(node):
            if len(node.classes) == 1:
                return
            l, r = np.random.choice(node.classes, size = 2, replace = False)
            l_group = [l]
            r_group = [r]
            clf = node.model
            _X = X[np.isin(y,[l,r])]
            _y = y[np.isin(y,[l,r])]
            clf.fit(_X, _y)
            preds = clf.predict(X[np.isin(y,node.classes)])
            _y = y[np.isin(y, node.classes)]
            for i in node.classes:
                if i not in [l,r]:
                    proportion_l = sum(preds[_y==i]==0)/sum(_y==i)
                    if proportion_l >= 0.5:
                        l_group.append(i)
                    else:
                        r_group.append(i)
            # We don't want leaf nodes to keep a model in memory
            if len(l_group)>1:
                node.left = Node(model = deepcopy(node.model))
            else:
                node.left = Node()
            if len(r_group)>1:
                node.right = Node(model = deepcopy(node.model))
            else:
                node.right = Node()
            node.left.classes = l_group
            node.right.classes = r_group
        self.root.preorder(_generate_split)
    
    # def set_models(self):
    #     def set_node_model(node):
    #         if not node.is_leaf():
    #             node.model = deepcopy(self.base_learner)
    #     self.root.preorder(set_node_model)
    
    def fit(self, X, y):
        self.build_structure(X, y)
        # self.set_models()
        super().fit(X, y)