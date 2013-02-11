'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import theano
from theano import function, scan
from theano.tensor import TensorType, add, sum, grad,  flatten, arange, concatenate, constant

import numpy as np 
import time 
from history import NpHistory, MultiHistory

import itertools as itools
import multiprocessing as mp
import collections

# TODO Can we change this to just 'Variable'? 
def Variable(name, shape, dtype='float64'):
    """
    Creates a TensorVariable of the given shape and type
    
    Parameters
    ----------
    
    shape : int or vector of ints        
    dtype : str
    
    Examples
    --------
    
    
    """
    shape = np.atleast_1d(shape)
    var = TensorType(str(dtype), shape == 1)(name)
    var.dshape = tuple(shape)
    var.dsize = int(np.prod(shape))
    return var

class Model(object):
    """
    Base class for encapsulation of the variables and 
    likelihood factors of a model.
    """
    
    def __init__(self, test_point):
       self.vars = []
       self.factors = [] 
       self.test_point = clean_point(test_point)
       if test_point is not None:
           theano.config.compute_test_value = 'raise'
       else:
           theano.config.compute_test_value = 'off'

    """
    these functions add random variables
    it is totally appropriate to add new functions to Model
    """
    def Data(model, data, distribution):
        args = map(constant, as_iterargs(data))
        model.factors.append(distribution(*args))

    def Var(model, name, distribution, shape = 1, dtype = 'float64'):
        var = Variable(name, shape, dtype)
        model.vars.append(var)
        if model.test_point is not None: 
            var.tag.test_value = model.test_point[name]
        model.factors.append(distribution(var))
        return var
        
    def VarIndirectElemewise(model, name,proximate_calc, distribution, shape = 1):
        var = Variable(name, shape)
        model.vars.append(var)
        prox_var = proximate_calc(var)
        
        model.factors.append(distribution(prox_var) + log_jacobian_determinant(prox_var, var))
        return var
     

def as_iterargs(data):
    if isinstance(data, tuple): 
        return data
    if hasattr(data, 'columns'): #data frames
        return [np.asarray(data[c]) for c in data.columns] 
    else:
        return [data]

       
def continuous_vars(model):
    return [ var for var in model.vars if var.dtype in continuous_types] 


"""
these functions compile log-posterior functions (and derivatives)
"""
def model_func(model, calcs, mode = None):
    f = function(model.vars, 
             calcs,
             allow_input_downcast = True, mode = mode)
    return KWArgFunc(f)

class KWArgFunc(object): 
    def __init__(self, f):
        self.f = f
    def __call__(self,state):
        return self.f(**state)

def model_logp(model, mode = None):
    return model_func(model, logp_calc(model), mode)

def model_dlogp(model, dvars = None, mode = None ):    
    return model_func(model, dlogp_calc(model, dvars), mode)
    
def model_logp_dlogp(model, dvars = None, mode = None ):    
    return model_func(model, [logp_calc(model), dlogp_calc(model, dvars)], mode)


"""
The functions build graphs from other graphs
"""
def flatgrad(f, v):
    return flatten(grad(f, v))

def gradient(f, dvars):
    return concatenate([flatgrad(f, v) for v in dvars])

def jacobian(f, dvars):
    def jac(v):
        def grad_i(i, f1, v): 
            return flatgrad(f1[i], v)
        
        return scan(grad_i, sequences=arange(f.shape[0]), non_sequences=[f,v])[0]

    return concatenate(map(jac, dvars))

def hessian(f, dvars):
    return jacobian(gradient(f, dvars), dvars)



def hessian_diag(f, dvars):
    def hess(v):
        df = flatten(grad(f, v))
        def grad_i (i, df1, v): 
            return flatgrad(df1[i], v)[i]
        
        return scan(grad_i, sequences = arange(f.shape[0]), non_sequences = [df,v])[0]

    return concatenate(map(hess, dvars))


"""
These functions build log-posterior graphs (and derivatives)
""" 
def logp_calc(model):
    """
    Calculates the log-probability of a specified model
        
    Parameters
    ----------
        
    model : Model  
        
    Examples
    --------
        
    >>> an example
        
    """
    return add(*map(sum, model.factors))

def dercalc(d_calc):
    """
    Returns a function for calculating the derivative of the output 
    of another function.
    
    Parameters
    ----------
    d_calc : function
    
    Returns
    -------
    der_calc : function
    """
    def der_calc(model, dvars = None):
        if dvars is None:
            dvars = continuous_vars(model)
        
        return d_calc(logp_calc(model), dvars)
    return der_calc

dlogp_calc = dercalc(gradient)
hess_calc = dercalc(jacobian)
hess_diag_calc = dercalc(hessian_diag)

def clean_point(d) : 
    return dict([(k,np.atleast_1d(v)) for (k,v) in d.iteritems()]) 
    
VarMap = collections.namedtuple('VarMap', 'var, slc, shp')

class IdxMap(object):
    def __init__(self, vars):
        self.vmap = []
        dim = 0

        for var in vars:       
            slc = slice(dim, dim + var.dsize)
            self.vmap.append( VarMap(str(var), slc, var.dshape)  )
            dim += var.dsize

        self.dimensions = dim
            

class DictArrBij(object):
    def __init__(self, idxmap, dpoint):
        self.idxmap = idxmap
        self.dpt = dpoint

    def map(self, dpt):
        """
        Maps value from dict space to array space
        
        Parameters
        ----------
        dpt : dict 
        """
        apt = np.empty(self.idxmap.dimensions)
        for var, slc, _ in self.idxmap.vmap:
                apt[slc] = np.ravel(dpt[var])
        return apt

    def rmap(self, apt): 
        """
        Maps value from array space to dict space 

        Parameters
        ----------
        apt : array
        """
        dpt = self.dpt.copy()
            
        for var, slc, shp in self.idxmap.vmap:
            dpt[var] = np.reshape(apt[slc], shp)
                
            
        return dpt

    def mapf(self, f):
        """
        Maps function f : DictSpace -> T to ArraySpace -> T
        
        Parameters
        ---------

        f : dict -> T 
        """
        return BijWrapIn(self,f)

class DictElemBij(object):
    def __init__(self, var, idx, dpoint):
        self.var = str(var)
        self.idx = idx
        self.dpt = dpoint

    def map(self, dpt):
        return dpt[self.var][self.idx]

    def rmap(self, apt):
        dpt = self.dpt.copy()
        if self.idx:   
            dvar = dpt[self.var].copy()
            dvar[self.idx] = apt
        else:
            dvar = apt
        dpt[self.var] = dvar
        
        return dpt 
    def mapf(self, f):
        return BijWrapIn(self, f)


class BijWrapIn(object):
    def __init__(self, bij, fn): 
        self.bij = bij
        self.fn = fn

    def __call__(self, d):
        return self.fn(self.bij.rmap(d))


# TODO Can we change `sample_history` to `trace`?
def sample(draws, step, point, sample_history = None, state = None): 
    """
    Draw a number of samples using the given step method. 
    Multiple step methods supported via compound step method 
    returns the amount of time taken.
        
    Parameters
    ----------
        
    draws : int  
        The number of samples to draw
    step : function
        A step function
    point : float or vector
        The current sample index
    sample_history : NpHistory
        A trace of past values (defaults to None)
    state : 
        The current state of the sampler (defaults to None)
        
    Examples
    --------
        
    >>> an example
        
    """

    point = clean_point(point)
    if sample_history is None: 
        sample_history = NpHistory()
    # Keep track of sampling time  
    tstart = time.time() 
    for _ in xrange(int(draws)):
        state, point = step.step(state, point)
        sample_history = sample_history + point

    return sample_history, state, time.time() - tstart

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)
  
def psample(draws, step, point, msample_history = None, state = None, threads = None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(point, dict) :
        point = threads * [point]

    if not msample_history:
        msample_history = MultiHistory([NpHistory() for _ in xrange(threads)])

    if not state: 
        state = threads*[None]

    p = mp.Pool(threads)

    argset = zip([draws]*threads, [step]*threads, point, msample_history.histories, state)
    
    # Keep track of sampling time  
    tstart = time.time() 

    res = p.map(argsample, argset)
    states, hist, _ = zip(*res)
        
    return msample_history, state, (time.time() - tstart)

# Sets of dtypes

bool_types = set(['int8'])
   
int_types = set(['int8',
            'int16' ,   
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64'])
float_types = set(['float32',
              'float64'])
complex_types = set(['complex64',
                'complex128'])
continuous_types = float_types | complex_types
discrete_types = bool_types | int_types

#theano stuff 
theano.config.warn.sum_div_dimshuffle_bug = False