import numpy as np 


class NestedDichotomy():
    def __init__(self, base_learner):
        self.root = None 
        self.base_learner = base_learner
    
    def fit(self, X, y):
        """ Fits the model """
        def fit_node(node):
                node.fit(X, y)
        self.root.preorder(fit_node)
    
    def predict_proba(self, X):
        n = len(self.root.classes)
        prob_matrix = np.ones((X.shape[0],n))
        def predict_node(node):
            if node.is_leaf():
                prob_matrix[:,node.classes] = node.p.reshape(-1,1)
            else:
                p = node.predict(X)
                node.left.p = p[:,0]*node.p
                node.right.p = p[:,1]*node.p
        def reset_probabilities(node):
            node.p = 1
        self.root.preorder(predict_node)
        self.root.preorder(reset_probabilities)
        return prob_matrix
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)

class Node():
    def __init__(self, model = None, classes = None):
        self.right = None
        self.left = None 
        self.model = model
        self.classes = classes
        self.p = 1
    
    def is_leaf(self):
        if len(self.classes)==1:
            return True
    
    def preorder(self, visit_fnc):
        if self is None:
            return
        visit_fnc(self)
        Node.preorder(self.left, visit_fnc)
        Node.preorder(self.right, visit_fnc)
    
    def fit(self, X, y):
        if self.is_leaf():
            return
        index_l = np.isin(y,self.left.classes)
        index_r = np.isin(y,self.right.classes)
        y_l = np.zeros(sum(index_l))
        y_r = np.ones(sum(index_r))
        X_l = X[index_l]
        X_r = X[index_r]
        _X = np.concatenate([X_l, X_r], axis = 0)
        _y = np.concatenate([y_l, y_r])
        self.model.fit(_X, _y)
    
    def predict(self, X):
        if self.is_leaf():
            return 
        return self.model.predict_proba(X)

        