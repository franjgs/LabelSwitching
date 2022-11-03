#
import numpy as np
import torch
import torch.nn as nn
from mlp import MLP, AsymmetricMLP1
from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer
from functools import reduce
import statsmodels.api as sm

from sklearn.metrics import log_loss

eps=np.finfo(float).eps

class LBFGSScipy(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now CPU only
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._n_iter = 0
        self._last_loss = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss = loss.data
            flat_grad = self._gather_flat_grad().numpy()
            return loss, flat_grad

        def callback(flat_params):
            self._n_iter += 1

        initial_params = self._gather_flat_params()

        
        fmin_l_bfgs_b(wrapped_closure, initial_params, maxiter=max_iter,
                      maxfun=max_eval,
                      factr=tolerance_change / eps, pgtol=tolerance_grad, epsilon=1e-08,
                      m=history_size,
                      callback=callback)

def label_switching(y, alphasw=0.0, betasw=0.0):

    # alphasw: Label Switching Rate from Majority to Minority class
    # betasw: Label Switching Rate from Minority to Majority class

    ysw=1*y;

    idx1=np.where(y==+1)[0]
    l1=len(idx1)
    bet_1=int(round(l1*betasw))
    idx1_sw=np.random.choice(idx1,bet_1, replace=False)
    ysw[idx1_sw]=-y[idx1_sw]

    idx0=np.where(y==-1)[0]
    l0=len(idx0)
    alph_0=int(round(l0*alphasw))
    idx0_sw=np.random.choice(idx0,alph_0, replace=False)
    ysw[idx0_sw]=-y[idx0_sw]

    return ysw

def compute_weights(targets_train, RB = 1, IR = 1, mode = 'Normal'):
    # RB define la cantidad de reequilibrado final
    # Si RB = IR => No se reequilibra. Si RB = 1, es un reequilibrado full.
    
    weights = np.ones_like(targets_train).astype('float')
    if mode == 'Small':
        weights[np.where(targets_train<=0)[0]] = RB/IR
    else:
        weights[np.where(targets_train>0)[0]] = IR/RB
    
    return torch.from_numpy(weights)

def logsumexp(x):
    y = torch.max(x, dim=1)[0]
    return y+torch.log(torch.exp(torch.sub(x, y[:, None])).sum(dim=1))

def train_LBFGS_scipy(x, y, model, loss_fn, optim, weights, num_epochs=1):
    dataset = torch.utils.data.TensorDataset(x, y) # create your dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                          shuffle=False, num_workers=0)
    #total_step = len(trainloader)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
       for i, data in enumerate(trainloader, 0):
            inputs, labels = data 
            
            def closure():
                optim.zero_grad()
                outputs = model(inputs.double()) # * gates
                loss = loss_fn(outputs.double(), labels.view(len(dataset),1).double(), weights.view(len(dataset),1).double())
                loss.backward()
                return loss

            optim.step(closure)
    return model

def train_LBFGS_gate_scipy(x, y, expert_outputs, gate_layer, loss_fn, optim, weights, num_epochs=1):
    dataset = torch.utils.data.TensorDataset(x, y) # create your dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                          shuffle=False, num_workers=0)
    #total_step = len(trainloader)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
       for i, data in enumerate(trainloader, 0):
            inputs, labels = data 
            
            def closure():
                optim.zero_grad()
                w_layer = gate_layer(inputs.double())
                outputs = torch.sum(expert_outputs*w_layer,dim=1)
                loss = loss_fn(outputs.double(), labels.view(len(dataset),1).double(), weights.view(len(dataset),1).double())
                loss.backward()
                return loss

            optim.step(closure)

    return gate_layer

#LBFGS TRAIN
def train_LBFGS(x, y, model, loss_fn, optim, weights):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    
    def closure():
        optim.zero_grad()
        #y_hat, aux_loss = model(x.float())
        y_hat = model(x.float())
        # calculate prediction loss
        loss = loss_fn(y_hat, torch.reshape(y,(len(y),1)), torch.reshape(weights,(len(weights),1)))
        # combine losses
        loss.backward()
        return loss

    optim.step(closure)
    #print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return 

def train_LBFGS_linear(x, y, model, loss_fn, optim, weights):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    
    def closure():
        optim.zero_grad()
        #y_hat, aux_loss = model(x.float())
        y_hat = model(x.float())
        # calculate prediction loss
        loss = loss_fn(y_hat, y, weights[:,None])
        # combine losses
        loss.backward()
        return loss

    optim.step(closure)
    #print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return 

def train_LBFGS_MoE(x, y, g_p, model, loss_fn, optim, weights):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    
    def closure():
        optim.zero_grad()
        #y_hat, aux_loss = model(x.float())
        y_hat = model(x)*g_p[:,None]
        # calculate prediction loss
        loss = loss_fn(y_hat, y, weights[:,None])
        # combine losses
        loss.backward()
        return loss

    optim.step(closure)
    #print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return 

def train_LBFGS_gate(x, y, cluster_outputs, gate_layer, loss_fn, optim, weights):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    def closure():
        
        optim.zero_grad()
        if cluster_outputs==None:
            w_layer = gate_layer(x.float())
            outputs = w_layer.double()
            """
            M_tr = len(x)
            X_tr_gate = torch.hstack([torch.ones(M_tr).unsqueeze(1), x])
            v_ = torch.hstack([gate_layer[0].bias.data.double().unsqueeze(1),gate_layer[0].weight.data.double()])
            gate = X_tr_gate @ v_.T
            outputs = torch.exp(gate-logsumexp(gate)[:, None]) # Identical to gates_train. g_p Eq 2.
            """
            # calculate prediction loss
            loss = loss_fn(outputs, y.double(), weights.double())
        else:
            w_layer = gate_layer(x.float())
            outputs = torch.sum(cluster_outputs*w_layer,dim=1).double()
            # calculate prediction loss
            loss = loss_fn(torch.reshape(outputs,(len(y),1)), torch.reshape(y.double(),(len(y),1)), torch.reshape(weights.double(),(len(weights),1)))
        # combine losses
        loss.backward()
        # print("Training Results - loss: {:.2f}".format(loss.item()))
        return loss

    optim.step(closure)
        
    # w_layer = gate_layer(x.float())
    # outputs = torch.sum(cluster_outputs*w_layer,dim=1)
    # loss = loss_fn(outputs, torch.reshape(y,(len(y),1)), torch.reshape(weights,(len(weights),1)))
    
    # print("Training Results - loss: {:.2f}".format(loss.item()))
    
    return gate_layer


def train_grad(trainloader, y_aux, model, loss_fn, optim, weights, input_size, epochs, it):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    for epoch in range(0, epochs): # 5 epochs at maximum
        ii = 0
        for Xy in trainloader:
            x = Xy[:, :input_size]
            y = Xy[:, input_size:]
            count = np.floor(len(y_aux[:,0])/50)
            if count == ii:
                weights_aux = weights[(ii*50):]
            else:
                weights_aux = weights[(ii*50):(ii+1)*50]
            
            ii=ii+1
            optim.zero_grad()
            #y_hat, aux_loss = model(x.float())
            y_hat = model(x.float())
            # calculate prediction loss
            loss = loss_fn(y_hat, torch.reshape(y[:,it],(len(y_hat),1)), torch.reshape(weights_aux,(len(weights_aux),1)))
            loss.backward()
            optim.step()

    #print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model

def train_grad_gate(x, y, cluster_outputs, gate_layer, loss_fn, optim, weights, input_size, epochs):
    for epoch in range(0, epochs): # 5 epochs at maximum
        optim.zero_grad()
        
        if cluster_outputs==None:
            w_layer = gate_layer(x.float())
            outputs = w_layer.double()
            """
            M_tr = len(x)
            X_tr_gate = torch.hstack([torch.ones(M_tr).unsqueeze(1), x])
            v_ = torch.hstack([gate_layer[0].bias.data.double().unsqueeze(1),gate_layer[0].weight.data.double()])
            gate = X_tr_gate @ v_.T
            outputs = torch.exp(gate-logsumexp(gate)[:, None]) # Identical to gates_train. g_p Eq 2.
            """
            # calculate prediction loss
            loss = loss_fn(outputs, y.double(), weights.double())
        else:
            #y_hat, aux_loss = model(x.float())
            w_layer = gate_layer(x.float())
            y_hat = torch.sum(cluster_outputs*w_layer,dim=1).double()
            # calculate prediction loss
            loss = loss_fn(torch.reshape(y_hat,(len(y_hat),1)), torch.reshape(y.double(),(len(y_hat),1)), torch.reshape(weights.double(),(len(weights),1)))
        loss.backward()
        optim.step()
        # print("Training Results - loss: {:.2f}".format(loss.item()))
        
    return gate_layer


def weighted_mse_loss(inputs, target, weights=None):
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    if isinstance(inputs,np.ndarray):
        inputs=torch.from_numpy(inputs)
    if weights==None:
        weights = torch.ones_like(inputs)
    return 0.5 * torch.sum((weights*((inputs - target)) ** 2)) # , torch.sum(weights))
    # return 0.5 * torch.div(torch.sum(weights*torch.linalg.norm(inputs - target)**2), torch.sum(weights))
    
def weighted_mse_fit(inputs, target, weights=None):
     return 1 - weighted_mse_loss(inputs, target, weights)
    
def weighted_bce_loss(inputs, target, weights=None):
    
    loss_bce = nn.BCELoss(weight=weights)
    inputs_01 = torch.abs(0.5*(inputs+1)).double() 
    targets_01 = 0.5*(target+1)
  
    """
    loss_bce = nn.BCELoss()
    
    ones = torch.ones_like(inputs)
    zeros = torch.zeros_like(inputs)
    inputs_01 = torch.where(inputs>0, ones, zeros)
    targets_01 = torch.where(target>0, ones, zeros)
    """
    return loss_bce(inputs_01, targets_01)

def weighted_bce_fit(inputs, target, weights=None):
    if not torch.is_tensor(inputs):
        inputs=torch.from_numpy(inputs).double()
    if not torch.is_tensor(target):
        target=torch.from_numpy(target).double()
    if weights != None and not torch.is_tensor(weights):
        weights=torch.from_numpy(weights).double()
    return 1 - weighted_bce_loss(inputs, target, weights)

def weighted_bce_logit_loss(inputs, target, weights):
    
    loss_bce = nn.BCEWithLogitsLoss(weight=weights)
    # inputs_01 = torch.abs(0.5*(inputs+1)) 
    # targets_01 = 0.5*(target+1)
  
    """
    loss_bce = nn.BCELoss()
    
    ones = torch.ones_like(inputs)
    zeros = torch.zeros_like(inputs)
    inputs_01 = torch.where(inputs>0, ones, zeros)
    targets_01 = torch.where(target>0, ones, zeros)
    """
    return loss_bce(inputs, target)

def f1_loss(predict, target, weights=None):
    
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    target = 0.5*(target+1)
    if isinstance(predict,np.ndarray):
        predict=torch.from_numpy(predict)
    predict = torch.clip(0.5*(predict+1),0,1) 
    # predict = torch.sigmoid(0.5*(predict+1))
    
    loss = 0
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += nn.BCEWithLogitsLoss(
            predict[:, lack_cls], target[:, lack_cls], weight=weights)
    # predict = torch.sigmoid(predict)
    
    # predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    
    # precision = tp / (predict.sum(dim=0) + 1e-8)
    # recall = tp / (target.sum(dim=0) + 1e-8)
    # f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    
    # macro_cost = f1.mean()
    
    fp = predict * (1 - target)
    fp = fp.sum(dim=0)
    
    fn = ((1 - predict) * target)
    fn = fn.sum(dim=0)
    
    tn = (1-predict)*(1-target)
    tn = tn.sum(dim=0)
    
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-8)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-8)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = cost.mean() # average on all labels
    
    return (macro_cost + loss)

def f1_fit(predict, target, weights=None):
    return 1 - f1_loss(predict, target, weights=None)

def TPR_fit(predict, target, weights=None):
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    target = 0.5*(target+1)
    if isinstance(predict,np.ndarray):
        predict=torch.from_numpy(predict)
    predict = torch.clip(0.5*(predict+1),0,1) 
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    return precision

def softthrh(z, eta):
    if z > 0 and eta < torch.abs(z):
        s=z-eta
    elif z < 0 and eta < torch.abs(z):
        s=z+eta
    else:
        s=0      
    return s

class BigMoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, num_experts, 
                 alpha=0, beta=0, RB_e=1, IR_e=1, RB_g=1, IR_g=1,
                 clusters=True):
        # arguments
        super(BigMoE, self).__init__()
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.num_experts = num_experts

        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta

        if clusters == True:
            self.num_clusters = len(alpha)
            self.gate_layer = nn.Sequential(nn.Linear(self.input_size, self.num_clusters), nn.Softmax(dim=1))
      
        else:
            self.num_clusters = 1
            # self.gate_layer = nn.Sequential(nn.Linear(self.input_size, self.num_experts), nn.Softmax(dim=1))
            # self.gate_layer[0].weight.data.fill_(0)
            # self.gate_layer[0].bias.data.fill_(0)
            self.gate_layer = nn.Linear(self.input_size, self.num_experts-1)
            # self.gate_layer.weight.data.fill_(0)
            # self.gate_layer.bias.data.fill_(0)
      
        self.experts = nn.ModuleList([AsymmetricMLP1(self.input_size, self.hidden_size, self.alpha, self.beta) for i in range(self.num_experts*self.num_clusters)])
        # Para conseguir expertos con distintas neuronas sustituimos self.experts
        # con el comentario de abajo
        
        self.experts = nn.ModuleList()
        if clusters == True:
            for i in range(self.num_clusters):
                for j in range(self.num_experts):
                    self.experts.add_module(str(i*self.num_experts+j), AsymmetricMLP1(self.input_size, self.hidden_size, self.alpha[i], self.beta))
        else:
            for j in range(self.num_experts):
                self.experts.add_module(str(j), AsymmetricMLP1(self.input_size, self.hidden_size+0*j, self.alpha, self.beta))
        # self.gate_layer = MLP(self.input_size, self.num_clusters,10*self.hidden_size)



        self.loss_fn_e = weighted_mse_loss # f1_loss # weighted_mse_loss # weighted_bce_loss # weighted_bce_logit_loss
        self.loss_fn_g = weighted_mse_loss # f1_loss #  weighted_mse_loss # weighted_bce_loss # weighted_bce_logit_loss
        self.RB_e = RB_e
        self.IR_e = IR_e
        self.RB_g = RB_g
        self.IR_g = IR_g
        
    def fit_experts(self, Xy_train, X_train, y_train, epochs=50, batch_size=50, lbfgs=True):
        trainloader = torch.utils.data.DataLoader(Xy_train, batch_size=batch_size, shuffle=False)
        
        weights = torch.zeros([len(y_train), self.num_experts*self.num_clusters])
        for i in range(self.num_experts*self.num_clusters):
            weights[:,i] = torch.reshape(compute_weights(y_train, RB = self.RB_e, IR = self.IR_e, mode = 'Normal'),(len(y_train),))
        
        # Switching
        y_train_sw = torch.zeros([len(y_train), self.num_experts*self.num_clusters])
        if self.num_clusters == 1:
            for i in range(self.num_clusters):
                for j in range(self.num_experts):
                    if self.alpha==0:
                        y_train_sw[:,i*self.num_experts+j] = y_train
                    else:
                        targets_sw = label_switching(y_train, self.alpha, self.beta)
                        beta_sw = (1-2*self.beta)*torch.ones_like(targets_sw).float()
                        alpha_sw = -(1-2*self.alpha)*torch.ones_like(targets_sw).float()
                        y_train_sw[:,i*self.num_experts+j] = torch.where(targets_sw>0, beta_sw, alpha_sw)
        else:
            for i in range(self.num_clusters):
                for j in range(self.num_experts):
                    if self.alpha[i]==0:
                        y_train_sw[:,i*self.num_experts+j] = y_train
                    else:
                        targets_sw = label_switching(y_train, self.alpha[i], self.beta)
                        beta_sw = (1-2*self.beta)*torch.ones_like(targets_sw).float()
                        alpha_sw = -(1-2*self.alpha[i])*torch.ones_like(targets_sw).float()
                        y_train_sw[:,i*self.num_experts+j] = torch.where(targets_sw>0, beta_sw, alpha_sw)
                
        if lbfgs==True:
            for i in range(self.num_experts*self.num_clusters):
                #optim_LBFGS = torch.optim.LBFGS(self.experts[i].parameters(), lr=0.001, max_iter=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10, line_search_fn='strong_wolfe')
                #self.experts[i] = train_LBFGS(X_train, y_train[:,i], self.experts[i], self.loss_fn_g, optim_LBFGS, weights[:,i])
                optim_LBFGS_scipy = LBFGSScipy(self.experts[i].parameters(), max_iter=150, max_eval=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10)
                train_LBFGS_scipy(X_train, y_train_sw[:,i], self.experts[i], self.loss_fn_e, optim_LBFGS_scipy, weights[:,i])
                
        else:
            for i in range(self.num_experts*self.num_clusters):
                optim_RMS = torch.optim.RMSprop(self.experts[i].parameters(), lr=0.0001)
                train_grad(trainloader, y_train_sw[:,i], self.experts[i], self.loss_fn_e, optim_RMS, weights[:,i], self.input_size, epochs, i)
        return self
    

    def fit_expert_model(self, Xy_train, X_train, y_train, w_train, epochs=50, batch_size=50, lbfgs=True):
        trainloader = torch.utils.data.DataLoader(Xy_train, batch_size=batch_size, shuffle=False)

        if self.num_clusters == 1:
            weights = torch.ones_like(y_train)
            for i in range(self.num_experts*self.num_clusters):
                weights[:,i] = torch.reshape(compute_weights(y_train[:,i], RB = self.RB_e, IR = self.IR_e, mode = 'Normal'),(len(y_train[:,i]),))
            weights *= w_train
            y_train_sw = y_train
        else:
            weights = torch.zeros([len(y_train), self.num_experts*self.num_clusters])
            for i in range(self.num_experts*self.num_clusters):
                weights[:,i] = torch.reshape(compute_weights(y_train, RB = self.RB_e, IR = self.IR_e, mode = 'Normal'),(len(y_train),))
            weights *= w_train
            y_train_sw = torch.zeros([len(y_train), self.num_experts*self.num_clusters])
            for i in range(self.num_clusters):
                for j in range(self.num_experts):
                    # Switching
                    if self.alpha[i]==0:
                        y_train_sw[:,i*self.num_experts+j] = y_train
                    else:
                        targets_sw = label_switching(y_train, self.alpha[i], self.beta)
                        beta_sw = (1-2*self.beta)*torch.ones_like(targets_sw).float()
                        alpha_sw = -(1-2*self.alpha[i])*torch.ones_like(targets_sw).float()
                        y_train_sw[:,i*self.num_experts+j] = torch.where(targets_sw>0, beta_sw, alpha_sw)
                
        if lbfgs==True:
            for i in range(self.num_experts*self.num_clusters):
                #optim_LBFGS = torch.optim.LBFGS(self.experts[i].parameters(), lr=0.001, max_iter=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10, line_search_fn='strong_wolfe')
                #self.experts[i] = train_LBFGS(X_train, y_train[:,i], self.experts[i], self.loss_fn_g, optim_LBFGS, weights[:,i])
                optim_LBFGS_scipy = LBFGSScipy(self.experts[i].parameters(), max_iter=150, max_eval=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10)
                # train_LBFGS_scipy(X_train, torch.div(y_train_sw[:,i],w_train[:,i]), self.experts[i], self.loss_fn_e, optim_LBFGS_scipy, weights[:,i])
                train_LBFGS_scipy(X_train, y_train_sw[:,i], self.experts[i], self.loss_fn_e, optim_LBFGS_scipy, weights[:,i])
        else:
            for i in range(self.num_experts*self.num_clusters):
                optim_RMS = torch.optim.RMSprop(self.experts[i].parameters(), lr=0.0001)
                train_grad(trainloader, y_train_sw[:,i], self.experts[i], self.loss_fn_e, optim_RMS, weights[:,i], self.input_size, epochs, i)
        return self
    
    
    def fit_gate(self, X_train, y_train, cluster_outputs, epochs=50, batch_size=50, lbfgs=True):

        weights = compute_weights(y_train, RB = self.RB_g, IR = self.IR_g, mode = 'Normal')
        # cluster_outputs
        
        if lbfgs==True:
            optim_LBFGS = torch.optim.LBFGS(self.gate_layer.parameters(), lr=1e-2, max_iter=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10, line_search_fn='strong_wolfe')
            train_LBFGS_gate(X_train, y_train, cluster_outputs, self.gate_layer, self.loss_fn_g, optim_LBFGS, weights)
            #optim_LBFGS_scipy = LBFGSScipy(self.gate_layer.parameters(), max_iter=150, max_eval=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10)
            #train_LBFGS_gate_scipy(X_train, y_train, expert_outputs, self.gate_layer, self.loss_fn_g, optim_LBFGS_scipy, weights)
        else:
            # Convendría explorar diferentes parámetros para el optimizador y/o
            # otros optimizadores.
            optim_RMS = torch.optim.RMSprop(self.gate_layer.parameters(), lr=10e-4, alpha=0.9, eps=1e-08, weight_decay=0.5, momentum=1, centered=False)
            train_grad_gate(X_train, y_train, cluster_outputs, self.gate_layer, self.loss_fn_g, optim_RMS, weights, self.input_size, epochs)
        return self
    
    def fit_gate_linear(self, X_train, y_train, weights):

        optim_LBFGS = torch.optim.LBFGS(self.gate_layer.parameters(), lr=1e-2, max_iter=150, tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10, line_search_fn='strong_wolfe')
        train_LBFGS_linear(X_train, y_train, self.gate_layer, self.loss_fn_g, optim_LBFGS, weights)
        """
        Ne_=y_train.shape[1]
        for kk in range(Ne_):
            train_LBFGS(X_train, y_train[:,kk], self.gate_layer, self.loss_fn_g, optim_LBFGS, weights)
        """
        return self
    
    @staticmethod
    def fit_gate_model(x, y, w, theta, lamda, Mode='mlr'):
        
        if Mode=='mlr':
            y_np = y.detach().numpy()
            x_np = x.detach().numpy()
            w_np = w.detach().numpy()
            mod_wls = sm.WLS(y_np,x_np,weights=w_np)
            theta_i = mod_wls.fit().params
            theta_i = torch.from_numpy(theta_i)
            df = x.shape[1]
        else:
            if Mode=='lasso':
                alph = 1.0
            elif Mode=='rr':
                alph =0
            elif Mode=='en':
                alph = 0.5
        
            # weights = compute_weights(y_train, RB = self.RB_g, IR = self.IR_g, mode = 'Normal')
            
            theta_i = theta.clone()
            den = torch.mul(x**2,w[:,None]).sum(dim=0)

            for jj in range(x.shape[1]):
                idx = np.r_[0:jj, jj+1:x.shape[1]]
                tmp = torch.sum(w*x[:,jj]*(y-x[:,idx]@theta_i[idx]))
                theta_i[jj] = softthrh(tmp, lamda*alph)/(eps+den[jj]+lamda*(1-alph))
            df = len(torch.where(theta_i != 0)[0])
        return theta_i, df
 
    
    def get_outputs(self, x):
        expert_outputs = self.experts[0](x)
        aux_outputs = torch.zeros_like(expert_outputs)
        for i in range(self.num_clusters*self.num_experts-1):
            aux_outputs = self.experts[i+1](x)
            expert_outputs = torch.cat((expert_outputs, aux_outputs), 1)
        return expert_outputs
    
    def predict_nogate(self, x):
        with torch.no_grad():
            expert_outputs = self.get_outputs(x.double())
            if self.num_clusters == 1:
                y_hat = expert_outputs
            else:   
                y_hat = torch.mean(expert_outputs[:,0:self.num_experts],dim=1).view(expert_outputs.size()[0],1)
                y_aux = torch.zeros_like(y_hat)
                for i in range(self.num_clusters-1):
                    y_aux = torch.mean(expert_outputs[:,(i+1)*self.num_experts:(i+2)*self.num_experts],dim=1).view(expert_outputs.size()[0],1)
                    y_hat = torch.cat((y_hat, y_aux), 1)
        return y_hat
    
    def get_gates(self, x, y):
        with torch.no_grad():
            gates = self.gate_layer(x)
            g0 = torch.sum(gates,dim=0)/x.shape[0]
            x1 = x[y>0,:]
            x0 = x[y<0,:]
            gates0 = self.gate_layer(x0)
            gates1 = self.gate_layer(x1)
            g00 = torch.sum(gates0,dim=0)/x0.shape[0]
            g01 = torch.sum(gates1,dim=0)/x1.shape[0]
        return gates, g0, g00, g01

    def predict(self, x):
        with torch.no_grad():
            gates = self.gate_layer(x.float())
            expert_outputs = self.predict_nogate(x.double())
            #print(torch.sum(gates,dim=0)/x.shape[0])
            y = torch.sum(expert_outputs*gates,dim=1)
        return y
        
    def forward(self, x):
        gates = self.gate_layer(x.float())
        expert_outputs = self.predict_nogate(x.double())
        #print(torch.sum(gates,dim=0)/x.shape[0])
        y = torch.sum(expert_outputs*gates,dim=1)
        return y