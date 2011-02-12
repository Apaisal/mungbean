'''
Created on Feb 12, 2011

@author: anol
'''
import PyML
import pybrain

def SVM_clas(data):
    '''
    '''
#def SVM_clas(X1, y1, X2, y2, tol, C, sigma):
#    options = svmsmoset('TolKKT', tol, 'Display', 'iter', 'MaxIter', 20000, 'KernelCacheLimit', 10000);
##    % Training and Ploting parameters
#    [SVMstruct, svIndex] = svmtrain(X1, y1, 'KERNEL_FUNCTION', 'rbf', 'RBF_SIGMA', sigma, 'BOXCONSTRAINT', C, 'showplot', true, ...
#    'Method', 'SMO', 'SMO_Opts', options);
##    % Computation of the error probability
#    train_res = svmclassify(SVMstruct, X1);
#    pe_tr = sum(y1 ~= train_res) / length(y1);
#    test_res = svmclassify(SVMstruct, X2);
#    pe_te = sum(y2 ~= test_res) / length(y2);
#    return [SVMstruct, svIndex, pe_tr, pe_te]

#def NN_training(x, y, k, code, iter, par_vec)
##    % Initialization of the random number
#    rand('seed', 0)
##% generators
##    randn('seed', 0)
##% for reproducibility of net initial
##% conditions
##% List of training methods
#    methods_list = {'traingd'; 'traingdm'; 'traingda'};
##    % Limits of the region where data lie
#    limit = [min(x(:, 1)) max(x(:, 1)); min(x(:, 2)) max(x(:, 2))];
##    % Neural network definition
#    net = newff(limit, [k 1], {'tansig', 'tansig'}, ...
#    methods_list{code, 1});
##    % Neural network initialization
#    net = init(net);
##    % Setting parameters
#    net.trainParam.epochs = iter;
#    net.trainParam.lr = par_vec(1);
#    if(code == 2):
#        net.trainParam.mc = par_vec(2);
#    elif(code == 3):
#        net.trainParam.lr_inc = par_vec(3);
#        net.trainParam.lr_dec = par_vec(4);
#        net.trainParam.max_perf_inc = par_vec(5);
#
#    #% Neural network training
#    return train(net, x, y);
#    #%NOTE: During training, the MATLAB shows a plot of the
#    #% MSE vs the number of iterations.
