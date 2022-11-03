#

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.Tanh() # nn.ReLU()
        self.soft = nn.Softmax(1) # nn.ReLU() #nn.Linear(output_size, output_size) #  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

class AsymmetricMLP0(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, beta):
        super(AsymmetricMLP0, self).__init__()
        
        n_out = 1
        self.alpha = alpha
        self.beta = beta
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size))
        
        self.hidden0.apply(init_weights)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, n_out))
        self.out.apply(init_weights)
    
    def forward(self, x):
        o = torch.tanh(self.hidden0(x))
        o = torch.tanh(self.out(o))
        return o

class AsymmetricMLP1(nn.Module):

    def __init__(self, input_size, hidden_size, alpha, beta):
        super(AsymmetricMLP1, self).__init__()
        
        n_out = 1
        self.alpha = alpha
        self.beta = beta
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                #nn.init.zeros_(m.weight)

        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size))
        
        self.hidden0.apply(init_weights)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, n_out))
        self.out.apply(init_weights)
        
    def forward(self, x):
        o = torch.tanh(self.hidden0(x))
        z = self.out(o)
        return torch.where(z < 0, torch.tanh(z)*(1-2*self.alpha), torch.tanh(z)*(1-2*self.beta))
    
    
class AsymmetricMLP2(nn.Module):

    def __init__(self, input_size, hidden_size, alpha, beta):
        super(AsymmetricMLP2, self).__init__()
        
        n_out = 1
        self.alpha = alpha
        self.beta = beta
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size))
        
        self.hidden0.apply(init_weights)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, n_out))
        self.out.apply(init_weights)
        
    def forward(self, x):
        o = torch.tanh(self.hidden0(x))
        z = self.out(o)
        return torch.where(z < 0, torch.tanh(z/(1-2*self.alpha))*(1-2*self.alpha), torch.tanh(z/(1-2*self.beta))*(1-2*self.beta))
    
"""
class AsymmetricMLP(): # (nn.Module):

    __name__ = "AsymmetricMLP"
    def __init__(self, hidden_layer_sizes, input_size=None, alpha=None, beta=None, activ=None, RB=None, IR=None, weight_mode=None):
        self.model = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.RB = RB
        self.IR = IR
        self.alpha = alpha
        self.beta=beta
        self.activ = activ
        self.weight_mode = weight_mode

                                
    def fit(self, X_train, y_train):
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(X_train, y_train)
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y_train)
        
        # AsymmetricMLP1(input_size=input_size, hidden_size=hidden_layer_sizes, alpha=alpha, beta=beta).to(device)
        
        input_size = X_train.shape[1]
        
        if self.activ == 'act1':  
            self.model = AsymmetricMLP1(input_size=input_size, 
                                        hidden_size=self.hidden_layer_sizes, 
                                        alpha=self.alpha, 
                                        beta=self.beta).to(device)
        elif self.activ == 'act2':
            self.model = AsymmetricMLP2(input_size=input_size, 
                                        hidden_size=self.hidden_layer_sizes, 
                                        alpha=self.alpha, 
                                        beta=self.beta).to(device)
        else:
            self.model = AsymmetricMLP0(input_size=input_size, 
                                        hidden_size=self.hidden_layer_sizes, 
                                        alpha=self.alpha, beta=self.beta).to(device)
        
        tensor_x_tr = torch.from_numpy(X_train)
        tensor_y_tr = torch.from_numpy(y_train)

        dataset = torch.utils.data.TensorDataset(tensor_x_tr, tensor_y_tr) # create your dataset
        lbfgs_optim = LBFGSScipy(self.model.parameters(), max_iter=150, max_eval=150, 
                                                      tolerance_grad=1e-04, tolerance_change=10e6*eps, 
                                                      history_size=10)
                            
        loss = weighted_mse_loss
                            
        weights = compute_weights(y_train, RB = self.RB, IR = self.IR, mode = self.weight_mode)
                            
        train_discriminator(model=self.model, optimizer=lbfgs_optim, criterion=loss, 
                            train_weights=weights, num_epochs=1, dataset=dataset)
        
        self.weights_in_ = self.model.hidden0.state_dict()['0.weight']
        self.bias_in_ = self.model.hidden0.state_dict()['0.bias']

        self.weights_out_ = self.model.out.state_dict()['0.weight']
        self.bias_out_ = self.model.out.state_dict()['0.bias']
        
        # Return the classifier
        return self
                            
    def predict(self, X_test):
        
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X_test = check_array(X_test)
        
        tensor_x_tst = torch.from_numpy(X_test)
        y_pred = self.model(tensor_x_tst.double()).detach().numpy().flatten()
        
        return y_pred
    
    def predict_proba(self, X):
        o_x = self.predict(X)
        
        pyX=0.5*(o_x+1)
        pyX = np.array([1-pyX, pyX]).transpose()
        
        return pyX
        
"""