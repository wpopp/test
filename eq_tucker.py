from time import *
from functools import partial
import gc
import os
from multiprocessing import Process,Queue
import multiprocessing as mp
import ast
import math as m
import sys
import timeit
import numpy as np
from pprint import pprint


#import matplotlib.pyplot as plt


#Givens rotation subfunction
def givens_rot_mat_entries(a,b):
    r = hypot(a,b)
    c = a/r
    s = -b/r

    return (c,s)

#Givens rotation to generate unit vector in first column of matrix

def givens(inmat,matdim):

    conv = 1.0e-24
    #initialise transformation matrix
    outmat = np.identity(matdim)

    #run over rows
    for i in range(1,matdim):
        if abs(inmat[i,0]) > conv:
            (c,s)= givens_rot_mat_entries(inmat[0,0],inmat[i,0])
            G=np.identity(matdim)
            G[[0,i],[0,i]] = c
            G[i,0] = s
            G[0,i] = -s
            inmat = np.dot(G,inmat)
            outmat = np.dot(G,outmat)

    return outmat,inmat


#vector projection for gram schmidt
def proj(u,v):
    return u * np.dot(v,u) / np.dot(u,u)

#gram schmidt
def gs(T):
    T = 1.0*T
    U = np.copy(T)
    for i in range(1,T.shape[1]):
        for j in range(i):
            U[:,i] -= proj(U[:,j],T[:,i])

    den=(U**2).sum(axis=0)**0.5
    E = U/den

    return E



def pause():
    programPause = input("Press the <ENTER> key to continue...")



#tensor algebra from tensorly, don't use it for tensor-vector multiplication!
def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1),order='C')


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape,order='C'), 0, mode)



def mode_dot(tensor, matrix_or_vector, mode):
        """n-mode product of a tensor and a matrix or vector at the specified mode

        Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


        Parameters
        ----------
        tensor : ndarray
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        ndarray
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        multi_mode_dot : chaining several mode_dot in one call
        """
        # the mode along which to fold might decrease if we take product with a vector
        fold_mode = mode
        new_shape = list(tensor.shape)

        if np.ndim(matrix_or_vector) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                    ))
            new_shape[mode] = matrix_or_vector.shape[0]

        elif np.ndim(matrix_or_vector) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                    ))
            if len(new_shape) > 1:
                new_shape.pop(mode)
                fold_mode -= 1
            else:
                new_shape = [1]

        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                             'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))
        res = np.dot(matrix_or_vector, unfold(tensor, mode))
        return fold(res, fold_mode, new_shape)

def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of lengh ``tensor.ndim``

    skip : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`

    See also
    --------
    mode_dot
    """
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    for i, (matrix_or_vec, mode) in enumerate(zip(matrix_or_vec_list, modes)):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot(res, np.transpose(matrix_or_vec), mode - decrement)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement)

        if np.ndim(matrix_or_vec) == 1:
            decrement = 1

    return res

def mult_kronecker(matrices,skip_matrix=None):
    if skip_matrix is not None:
        matrices = [matrices[i,:,:] for i in range(len(matrices)) if i != skip_matrix]
    for i, matrix in enumerate(matrices[::1]):
        if not i:
            res = matrix
        else:
            res = np.kron(res,matrix)
    return res



#loss function to minimize

def loss_func(sig_tens,nstates,core,A,indices,gloss=0):

#Calculate the loss function after reconstruction of the signal tensor from
#the updated factor matrices and core tensor
    A_el = np.eye(nstates)
    loss = 0.0

#reconstruct signal tensor from new core and factor
    reconstruction = multi_mode_dot(core,[A_el,A_el,A[:,:],A[:,:]],[0,1,2,3],transpose=False)
    for i,tup in indices.items():
        loss += (sig_tens[tup] - reconstruction[tup])**2

    if(gloss == 0):
        loss += reg_par_fac*((np.linalg.norm(A[:,:])**2)+(np.linalg.norm(A[:,:])**2))

    return loss

#find correct root of cubic polynomial
def check_roots(row,col,root_arr,sig_tens,factor,alpha_dict,reg_par_fac):
    L_arr = np.zeros(len(root_arr))
    factor_old = factor

    #if only complex numbers result, convert the real value
    if( any(np.iscomplex(root_arr))==True ):
        for i in root_arr:
            if( np.imag(i)==0.0 ):
                return np.real(i)
    else:
        for i in range(len(root_arr)):
            factor = factor_old
            #put root into factor matrix
            factor[row,col] = root_arr[i]

            #calculate loss function

            L_arr[i] = 0.0
            for key,tup in alpha_dict.items():

                reconst = 0.0
                for j in range(nbasis_fin):
                    reconst += factor[tup[2],j]*np.dot(G_mat_dict[tup[0:2]][j,:],factor[tup[3],:])

                L_arr[i] += (sig_tens[tup] - reconst)**2

            L_arr[i] += 2*reg_par_fac*(np.linalg.norm(factor)**2)

        global_min = np.argmin(L_arr)

        return root_arr[global_min]

#calculate third order prefactor
def calc_third_order(G_mat_dict,gamma_dict,row,col):
    third_order = 0.0
    for key,tup in gamma_dict.items():
        if( tup[2] == row ):
            third_order += 4*G_mat_dict[tup[0:2]][col,col]**2

    #l1.put(third_order)
    return third_order

#calculate second order prefactor

def calc_second_order(G_mat_dict,gamma_dict,factor,row,nbasis_fin,col):
    second_order = 0.0
    for key,tup in gamma_dict.items():
        if( tup[2] == row ):
            #g_isum = 0.0
            g_vec = (G_mat_dict[tup[0:2]][col,:]+G_mat_dict[tup[0:2]][:,col])

            #for j in range(nbasis_fin):
            #    if(j!=col):
            #        g_isum += (G_mat_dict[tup[0:2]][col,j] + G_mat_dict[tup[0:2]][j,col]\
            #                )*factor[row,j]
            #second_order += 6*G_mat_dict[tup[0:2]][col,col]*g_isum
            second_order += 6*G_mat_dict[tup[0:2]][col,col]*(np.dot(g_vec,factor[row,:])-g_vec[col]*factor[row,col])

    #l2.put(second_order)
    return second_order

#calculate first order prefactor

def calc_first_order(beta_dict,gamma_dict,G_mat_dict,factor,row,col,sig_tens,reg_par,nbasis_fin):

    first_order = 0.0
    #beta_i2 = i
    b_i2 = 0.0
    for key,tup in beta_dict.items():
        if( tup[2] == row ):
            b_i2 += 2*(np.dot(G_mat_dict[tup[0:2]][col,:],factor[tup[3],:])**2)
    #beta_i3 = i
    b_i3 = 0.0
    for key,tup in beta_dict.items():
        if( tup[3] == row ):
            b_i3 += 2*(np.dot(G_mat_dict[tup[0:2]][:,col],factor[tup[2],:])**2)
    #first gamma summand
    g_1 = 0.0
    for key,tup in gamma_dict.items():
        if( tup[2] == row ):
            vec = np.dot(G_mat_dict[tup[0:2]],factor[row,:]) - G_mat_dict[tup[0:2]][:,col]*factor[row,col]
            g_1 += 4*(G_mat_dict[tup[0:2]][col,col]*( np.dot(factor[row,:],vec) - factor[row,col]*vec[col] )\
                    -sig_tens[tup]*G_mat_dict[tup[0:2]][col,col])

    #second gamma summand
    g_2 = 0.0
    for key,tup in gamma_dict.items():
        if( tup[2] == row ):
            g_isum = 0.0
            for j in range(nbasis_fin):
                if(j!=col):#j.eq.l in eq 67
                    g_isum += factor[row,j]*(G_mat_dict[tup[0:2]][col,j]+G_mat_dict[tup[0:2]][j,col])**2
            g_2 += 2*g_isum


    first_order = b_i2 + b_i3 + g_1 + g_2 + 4*reg_par_fac

    #l3.put(first_order)
    return(first_order)

#calculate zeroth order prefactor

def calc_zeroth_order(beta_dict,gamma_dict,G_mat_dict,factor,sig_tens,nbasis_fin,row,col):
    zeroth_order = 0.0
    b_i2 = 0.0
    for key,tup in beta_dict.items():
        if( tup[2] == row ):
            b_insum = 0.0

            vec = np.dot(G_mat_dict[tup[0:2]],factor[tup[3],:])

            b_i2 += np.dot(G_mat_dict[tup[0:2]][col,:],factor[tup[3],:])*\
                        ( np.dot(factor[row,:],vec)-factor[row,col]*vec[col]   )-\
                            sig_tens[tup]*np.dot(G_mat_dict[tup[0:2]][col,:],factor[tup[3],:])
    b_i2 = 0.0
    for key,tup in beta_dict.items():
        if( tup[2] == row ):
            b_insum = 0.0

            for j in range(nbasis_fin):#j.eq.l in eq 67
                if(j!=col):
                    b_insum += factor[row,j]*np.dot(G_mat_dict[tup[0:2]][j,:],factor[tup[3],:])
            b_i2 += np.dot(G_mat_dict[tup[0:2]][col,:],factor[tup[3],:])*b_insum - \
                    sig_tens[tup]*np.dot(G_mat_dict[tup[0:2]][col,:],factor[tup[3],:])


    b_i3 = 0.0
    for key,tup in beta_dict.items():
        if( tup[3] == row):
            b_insum = 0.0
            vec = np.dot(G_mat_dict[tup[0:2]],factor[row,:]) - G_mat_dict[tup[0:2]][:,col]*factor[row,col]
            b_i3 += np.dot(G_mat_dict[tup[0:2]][:,col],factor[tup[2],:])*np.dot(factor[tup[2],:],vec)\
                - sig_tens[tup]*(np.dot(G_mat_dict[tup[0:2]][:,col],factor[tup[2],:]))


    #g_term
    g_term = 0.0
    for key,tup in gamma_dict.items():
        if( tup[2] == row ):
            sf_1 = 0.0
            vec = G_mat_dict[tup[0:2]][col,:]+G_mat_dict[tup[0:2]][:,col]
            sf_1 = np.dot(vec,factor[row,:]) - vec[col]*factor[row,col]

            sf_2 = 0.0
            vec = np.dot(G_mat_dict[tup[0:2]],factor[row,:]) - G_mat_dict[tup[0:2]][:,col]*factor[row,col]
            sf_2 = np.dot(factor[row,:],vec)-factor[row,col]*vec[col]

            g_term += sf_1*sf_2-sig_tens[tup]*sf_1


    zeroth_order =   2*(b_i2 + b_i3 + g_term)

    #l4.put(zeroth_order)
    return zeroth_order

#get new factor matrix row from pool process
def refresh_results(res,row,factor):
    factor[row,:] = res

#updating one row of the factor matrix
def update_factor_row(row,sig_tens,factor,G_mat_dict,nbasis_fin,beta_dict\
        ,gamma_dict,alpha_dict,reg_par_fac):

    new_factors = np.zeros(nbasis_fin)

    #iterate over columns of factor row vector (j)
    for i in range(1,nbasis_fin):

        third_order = calc_third_order(G_mat_dict,gamma_dict,row,i)

        second_order = calc_second_order(G_mat_dict,gamma_dict,factor,row,nbasis_fin,i)

        first_order = calc_first_order(beta_dict,gamma_dict,G_mat_dict,factor,row,\
                i,sig_tens,reg_par_fac,nbasis_fin)
        zeroth_order = calc_zeroth_order(beta_dict,gamma_dict,G_mat_dict,factor,\
                sig_tens,nbasis_fin,row,i)

        #calculate roots of the polynomial
        root_arr = np.roots([third_order, second_order, first_order, zeroth_order])

        factor[row,i] = check_roots(row,i,root_arr,sig_tens,factor,alpha_dict,reg_par_fac)

    return factor[row,:]


def find_alpha_tuple(tensor):
    #find and store all tupels, for which the tensor has
    #non-zero entries
    counter = 1
    index_dict = {}
    for i in range(np.size(tensor,0)):
        for j in range(np.size(tensor,1)):
            for k in range(0,np.size(tensor,2)):
                for l in range(0,np.size(tensor,3)):
                    if( abs(X[i,j,k,l]) > 1e-30):
                        #store explicit index as tupel
                        tmp_indx = (i,j,k,l)
                        #and update the dictionary containing all indices for which
                        #the signal matrix entry is not zero
                        index_dict.update({counter:tmp_indx})
                        counter += 1

    return index_dict




def find_beta_tuple(tensor,row):
    #find tupels for the beta set. I.e. the second index is #
    #not equal to the third one (i_2 != i_3)
    counter = 0
    index_dict = {}
    for i in range(np.size(tensor,0)):
        for j in range(np.size(tensor,1)):
            for k in range(np.size(tensor,2)):
                for l in range(np.size(tensor,3)):
                    if( (k != l) and (abs(tensor[i,j,k,l]) > 1e-30) ):
                    #store explicit index as tupel
                        tmp_indx = (i,j,k,l)
                    #and update the dictionary containing all indices for which
                    #the signal matrix entry is not zero
                        index_dict.update({counter:tmp_indx})
                        counter += 1

    return index_dict



def find_gamma_tuple(tensor,row):
    #find tupels for the beta set. I.e. the second index is #
    #equal to the third one (i_2 = i_3)
    counter = 1
    index_dict = {}
    for i in range(np.size(tensor,0)):
        for j in range(np.size(tensor,1)):
            for k in range(np.size(tensor,3)):
                if( (k == row) and (abs(tensor[i,j,row,k]) > 1e-30) ):
                    #store explicit index as tupel
                    tmp_indx = (i,j,row,k)
                    #and update the dictionary containing all indices for which
                    #the signal matrix entry is not zero
                    index_dict.update({counter:tmp_indx})
                    counter += 1

    return index_dict

def find_gamma_tuple_full(tensor,row):
    #find tupels for the beta set. I.e. the second index is #
    #equal to the third one (i_2 = i_3)
    counter = 1
    index_dict = {}
    for i in range(np.size(tensor,0)):
        for j in range(np.size(tensor,1)):
            for k in range(np.size(tensor,2)):
                for l in range(np.size(tensor,3)):
                    if( (k == l) and (abs(tensor[i,j,k,l]) > 1e-30) ):
                    #store explicit index as tupel
                        tmp_indx = (i,j,k,l)
                    #and update the dictionary containing all indices for which
                    #the signal matrix entry is not zero
                        index_dict.update({counter:tmp_indx})
                        counter += 1

    return index_dict

def lfunc_a00(row,col,X,factor,gdict,adict,bdict,gamdict,xvals,nbasis_fin,reg_par_fac):


    y = 0.0
    for key,tup in bdict.items():

        if( tup[2] == row ):
            insum = 0.0
            for i in range(nbasis_fin):
                for j in range(nbasis_fin):
                    if(i!=col):
                        insum += factor[row,i]*gdict[tup[0:2]][i,j]*factor[tup[3],j]

            y += (X[tup] - xvals*np.dot(gdict[tup[0:2]][col,:],factor[tup[3],:])\
                    - insum)**2

        if( tup[3] == row ):
            insum = 0.0
            for i in range(nbasis_fin):
                for j in range(nbasis_fin):
                    if(j!=col):
                        insum += factor[tup[2],i]*gdict[tup[0:2]][i,j]*factor[row,j]

            y += (X[tup] - xvals*np.dot(gdict[tup[0:2]][:,col],factor[tup[2],:])\
                    - insum)**2


        if( tup[2] != row and tup[3] != row):
            y += (X[tup] - np.dot(factor[tup[2],:],np.dot(gdict[tup[0:2]],factor[tup[3],:])))**2

    for key,tup in gamdict.items():

        if( tup[2] == row ):
            insum = 0.0
            for i in range(nbasis_fin):
                if(i!=col):
                    insum += factor[row,i]*(gdict[tup[0:2]][col,i]+gdict[tup[0:2]][i,col])

            insum2 = 0.0
            for i in range(nbasis_fin):
                for j in range(nbasis_fin):
                    if(i!=col and j!=col):
                        insum2 += factor[row,i]*gdict[tup[0:2]][i,j]*factor[row,j]

            y += (X[tup] - (xvals**2)*gdict[tup[0:2]][col,col] -xvals*insum \
                    - insum2)**2

        if( tup[2] != row):
            y += (X[tup] - np.dot(factor[tup[2],:],np.dot(gdict[tup[0:2]],factor[tup[3],:])))**2


    y += 2*reg_par_fac*(np.linalg.norm(factor))**2

    #plt.plot(xvals,y)
   # plt.show()

    pause()


def part_rec_err(sig_tens,factor,g_tens,nstates,nbasis,nbasis_fin,alpha_dict):

    el_fac = np.eye(nstates)
    counter = 0
    g_tuple_dict = {}
    for i in range(np.size(g_tens,0)):
        for j in range(np.size(g_tens,1)):
            for k in range(np.size(g_tens,2)):
                for l in range(np.size(g_tens,3)):
                    g_tuple_dict.update({counter : (i,j,k,l) })
                    counter += 1


    r_err_vec = np.zeros(len(g_tuple_dict))
    counter = 0
    for b_key,b_tup in g_tuple_dict.items():

        for a_key,a_tup in alpha_dict.items():
           #calculate prefactor
            prefac = g_tens[b_tup]*el_fac[a_tup[0],b_tup[0]]*el_fac[a_tup[1],b_tup[1]]*\
                    factor[a_tup[2],b_tup[2]]*factor[a_tup[3],b_tup[3]]

            #calculate element without beta entry:
            ex_sum = 0.0
            for g_key,g_tup in g_tuple_dict.items():
                if( g_tup != b_tup ):
                    ex_sum += 2*g_tens[g_tup]*el_fac[a_tup[0],g_tup[0]]*\
                            el_fac[a_tup[1],g_tup[1]]*\
                    factor[a_tup[2],g_tup[2]]*factor[a_tup[3],g_tup[3]]

            r_err_vec[counter] += prefac*(prefac - 2*sig_tens[a_tup] + ex_sum)
        counter += 1


    minval = min(r_err_vec,key=abs)
    maxval = max(r_err_vec,key=abs)
    for i in range(len(r_err_vec)):
        if((abs(r_err_vec[i]) > abs(0.9*maxval))):
            if( (g_tuple_dict[i][0] == g_tuple_dict[i][1]) and \
                    g_tuple_dict[i][2:4] == (0,0)):

                continue
            g_tens[g_tuple_dict[i]] = 0.0
            print(g_tuple_dict[i])

    return g_tens


def calc_nom(tup,sig_tens,frametens,ytens):
    res = (sig_tens[tup] * framtens[tup]\
        - ytens[tup] * frametens[tup])

    return res



def collect_nom(result):

    nom_res.append(result)


def calc_denom(tup,frametens):

    return frametens[tup]**2

def collect_denom(result_denom):

    denom_res.append(result_denom)


def calc_g_mat(G,nstates,A_el):
    G_mat_dict = {}
    for i in range(nstates):
        for j in range(nstates):
            tmp = np.tensordot(G,A_el[i,:],(0,0))
            tmp2 = np.tensordot(tmp,A_el[j,:],(0,0))
            G_mat_dict.update({(i,j) : tmp2})

    return G_mat_dict


# MAIN #


approx = False
maxiter = 20
nstates = 5
nmodes=237
nmodes_fin=100
nbasis=nmodes+1
nbasis_fin=nmodes_fin+1
reg_par_fac = 0.01
reg_par_core = 15.0
reg_par_core_diag = 2.0

global factor
global core
global nom_res
global denom_res


elem = 1
outfile = open("OUTPUT","wb")

#number of distinctive electronic operators
n_eloper = nstates*(nstates+1)/2

#size of electronic submatrices [n,n,:,:]
n_elsub = ((nbasis*(nbasis+1))/2.0)*nstates

#number of total entries in signal tensor
sig_tens_dim = int( nstates*nstates*nbasis*nbasis )

#initialise signal tensor
X = np.zeros((nstates,nstates,nbasis,nbasis))

filename="output_blub"
#filename="ifile_9modes"

#-----------------------------------------#
#Relict from using f2py3 which behaved like a little bitch
#read in Taylor factors into signal tensor
#X = fort_utils_tucker.read_data(X,nstates,nmodes,filename)
#-----------------------------------------#,

with open(filename) as infile:
    for num,line in enumerate(infile,1):
        if "zeroth-order" in line:
            for i in range(nstates):
                X[i,i,0,0] = float(infile.readline().strip().replace('d','e'))

    infile.seek(0)
    for num,line in enumerate(infile,1):
        if "first-order diagonal s1" in line:
            for i in range(nstates):
                for j in range(nmodes):
                    X[i,i,0,j+1] = float(infile.readline().strip().replace('d','e'))/2.0
                    X[i,i,j+1,0] = X[i,i,0,j+1]
                infile.readline()

    infile.seek(0)
    for num,line in enumerate(infile,1):
        if "first-order off-diagonal s1s" in line:
            for i in range(nstates-1):
                for j in range(i+1,nstates):
                    for k in range(nmodes):
                        X[i,j,0,k+1] = float(infile.readline().strip().replace('d','e'))/2.0
                        X[j,i,0,k+1] = X[i,j,0,k+1]
                        X[i,j,k+1,0] = X[i,j,0,k+1]
                        X[j,i,k+1,0] = X[i,j,0,k+1]
                    infile.readline()

    infile.seek(0)
    for num,line in enumerate(infile,1):
        if "second-order diagonal s1" in line:
            for i in range(nstates):
                for j in range(1,nbasis):
                    for k in range(j,nbasis):
                        X[i,i,j,k] = float(infile.readline().strip().replace('d','e'))/2.0
                        X[i,i,k,j] = X[i,i,j,k]
                infile.readline()

    infile.seek(0)
    for num,line in enumerate(infile,1):
        if "second-order off-diagonal s1s2" in line:
            for i in range(nstates-1):
                for j in range(i+1,nstates):
                    for k in range(1,nbasis):
                        for l in range(k,nbasis):
                            X[i,j,k,l] = float(infile.readline().strip().replace('d','e'))/2.00
                            X[i,j,l,k] = X[i,j,k,l]
                    infile.readline()
                    X[j,i,:,:] = X[i,j,:,:]

#write X to check, wether everything worked:
outfile.write(b"Signal tensor\n")
outfile.write(b"\n")

for i in range(nstates):
    for j in range(nstates):
        outfile.write(b"\n")
        ws = "X |{}><{}|\n".format(i,j)
        ws = ws.encode('ASCII')

        outfile.write(ws)
        np.savetxt(outfile,X[i,j,:,:],fmt='%.5E')
outfile.write(b"\n")

#initialise core tensor
G = np.random.uniform(-1,1,(nstates,nstates,nbasis_fin,nbasis_fin))

#initialise factor matrix (only A^(2) and A^(3) are needed, since we do not want
#to touch the electronic dimensions AND they are identical, since the tensor is
#symmteric in these modes)
A = np.random.uniform(-1,1,(nbasis,nbasis_fin))
A[0,:] = 0.0
A[:,0] = 0.0
A[0,0] = 1.0


#symmetrise G
for i in range(np.size(G,0)):
    for j in range(i,np.size(G,1)):
        G[i,j,0,0] = X[i,j,0,0]
        for k in range(np.size(G,2)):
            for l in range(k,np.size(G,3)):
                #we do not want cross couplings between new modes:
                if( k > 0 and k != l ):
                    G[i,j,k,l] = 0.0
                if( i==j and k>0 and k==l):
                    if( G[i,j,k,l] < 0 ):
                        G[i,j,k,l] *= -1


                G[i,j,l,k] = G[i,j,k,l]
        G[j,i,:,:] = G[i,j,:,:]

#to test fr simplest case of unit transformation, set guess as signal tensor
#and factor matrix as unit matrix

#G = X
#A = np.eye(nbasis)

core = G.copy()
A_el = np.eye(nstates)
err_old = 300.0
err_new = 0.0
iteration = 1
#calculate the different vector products between
#the core tensor and the electronic factor matrices
#which are per definition unit matrices
global G_mat_dict
G_mat_dict = {}

G_mat_dict = calc_g_mat(G,nstates,A_el)

#set up dictionaries with relevant tuple
alpha_dict = {0:(0,0,0,0)}
alpha_dict.update(find_alpha_tuple(X))
beta_dict = find_beta_tuple(X,0)
gamma_dict = {0:(0,0,0,0)}
gamma_dict.update(find_gamma_tuple_full(X,j))
counter = len(gamma_dict)+1

for m in range(nstates):
    for n in range(nstates):
        if(m!=n):
            gamma_dict.update({counter:(m,n,0,0)})
            counter += 1

#--------------------------#
#IF YOU WOULD LIKE TO LOOK AT THE LOSS FUNCTION FOR A SPECIFIC ELEMENT a_ij#

#xvals = np.arange(-5,5,0.1)
#lfunc_a00(1,2,X,A,G_mat_dict,alpha_dict,beta_dict,gamma_dict,xvals,nbasis_fin,reg_par_fac)
#pause()
#--------------------------#


#insert tuple for which zero entry is hard
counter = len(alpha_dict)+1
for i in range(nstates):
    for j in range(nstates):
        if(i!=j):
            alpha_dict.update({counter:(i,j,0,0)})
            counter += 1

factor = A.copy()
core = G.copy()

#while( (abs(np.linalg.norm(G_temp)-np.linalg.norm(X)) > 1e-25 ) ):
for macro in range(10):

#while( abs(err_new-err_old) > 1e-10):

    print('Macroiteration ',macro)
    err_new = 0.0
    err_old = 300.0
    while( (abs(err_new-err_old) > 1e-25) ):
        err_old = err_new
        intime = time()
        print('Factor Iteration: ',iteration)

        #updating the factor matrices will be done only for the mode n=2
        #since mode 2 and 3 are symmetric and we do not want to decompose along
        #the electronic modes 0 and 1
        pool = mp.Pool(80)

        #iterate over rows of A matrix
        for j in range(1,nbasis):

            #update relevant tuple dictionaries for current row
            beta_dict = find_beta_tuple(X,j)
            gamma_dict = {0:(0,0,0,0)}
            gamma_dict.update(find_gamma_tuple(X,j))
            counter = len(gamma_dict)+1
            for m in range(nstates):
                for n in range(nstates):
                    if(m!=n):
                        gamma_dict.update({counter:(m,n,0,0)})
                        counter += 1

            #update row
            factor_time = time()

        #PARALLEL:
            new_callback_function=partial(refresh_results,row=j,factor=factor)
            pool.apply_async(update_factor_row,args = (j,X,factor,G_mat_dict,nbasis_fin,beta_dict\
                        ,gamma_dict,alpha_dict,reg_par_fac),callback=new_callback_function)
        #SERIAL:
            #factor[j,:] = update_factor_row(j,X,factor,G_mat_dict,nbasis_fin,beta_dict\
            #        ,gamma_dict,alpha_dict,reg_par_fac)

        pool.close()
        pool.join()

        print('Iteration took: ',time()-intime,' s')
        #calculate error
        err_new = loss_func(X,nstates,core,factor,alpha_dict)
        if( approx == True):
            G = part_rec_err(X,A,G,nstates,nbasis,nbasis_fin,alpha_dict)

        print('New Error: ',err_new)
        iteration += 1

    #start iterations for updating G tensor
    err_old = 300.0
    err_new = 0.0
    iteration = 1
    #Orthonormalise factor matrix
    Q,R = np.linalg.qr(factor)
    factor = Q
    #refresh core
    core = multi_mode_dot(core,[R,R],[2,3])
    G_mat_dict = calc_g_mat(core,nstates,A_el)
    while( (abs(err_new-err_old) > 1e-15) ):
        err_old = err_new
        print('Core Iteration: ',iteration)
        gdict = {}
        for i in range(np.size(core,0)):
            for j in range(np.size(core,1)):
                for k in range(np.size(core,2)):
                    for l in range(np.size(core,3)):
                        if(k == 0 and l == 0):
                            continue
                        else:
                            nom_res = []
                            denom_res = []

                            #calculate Y tensor
                            tensor = np.einsum('i,j,k,l',A_el[:,i],A_el[:,j],factor[:,k],factor[:,l])
                            #substract exception
                            Y = multi_mode_dot(core,[A_el,A_el,factor,factor],[0,1,2,3])\
                                    - core[i,j,k,l]*tensor

                            nominator = 0.0
                            denominator = 0.0
                            #open pools to calculate nominator and denominator simultaneous
                            pool_nom = mp.Pool(40)
                            pool_denom = mp.Pool(40)
                            #iterate over all relevant tuple
                            for key,tup in alpha_dict.items():
                                pool_nom.apply_async(calc_nom, args = (tup,\
                                        X,tensor\
                                        ,Y),callback = collect_nom)

                                pool_denom.apply_async(calc_denom, args = (tup,\
                                        tensor),callback = collect_denom)

                          #   for key,tup in alpha_dict.items():
                          #       #insum = calc_insum(i,j,k,l,gdict,nstates,nbasis_fin,tup)
                          #       nominator += sig_tens[tup] * gdict[i,j,k,l][1][tup]\
                          #               - Y[tup] * gdict[i,j,k,l][1][tup]
                            pool_nom.close()
                            pool_denom.close()
                            pool_nom.join()
                            pool_denom.close()

                            nominator = sum(nom_res)
                            denominator = sum(denom_res)

                            if( not ((i,j,k,l) in alpha_dict.values()) ):
                                denominator += reg_par_core
                            if(nominator == 0.0 and denominator == 0.0):
                                core[i,j,k,l] = 0.0
                            else:
                                core[i,j,k,l] = nominator / denominator
                                if( (core[i,j,k,l] < 0.0 ) and (k>0) and (k == l) ):
                                    core[i,j,k,l] = 1e-15


        err_new = loss_func(X,nstates,core,factor,alpha_dict,1)

        print('New Error: ',err_new)
        iteration += 1

    G_mat_dict = calc_g_mat(core,nstates,A_el)


#REDUNDANT IF ONE STARTS FACTOR UPDATE AT ROW AND COLUMN 1 INSTEAD OF 0:
#rotate first column to unit vector:
giv_mat,riv_mat = givens(factor,np.size(factor,0))

#gram schmidt orthonormalisation
A_ortho = gs(riv_mat)
#get matrix, that transforms A into A_ortho
base_change = np.dot(np.transpose(A),A_ortho)

outfile.write(b"Final Factor matrix\n")
np.savetxt(outfile,factor,fmt='%.5E')
outfile.write(b"\n")

outfile.write(b"Orthonormalised Factor matrix\n")
np.savetxt(outfile,A_ortho,fmt='%.5E')
outfile.write(b"\n")

G_new = core.copy()

G_fin = multi_mode_dot(G_new,[base_change,base_change],[2,3])
reconst = multi_mode_dot(G_fin,[A_ortho,A_ortho],[2,3])

#write reconstruction
outfile.write(b"Reconstruction from A matrix\n")
for i in range(nstates):
    for j in range(nstates):
        outfile.write(b"\n")
        ws = "X' |{}><{}|\n".format(i,j)
        ws = ws.encode('ASCII')

        outfile.write(ws)
        np.savetxt(outfile,reconst[i,j,:,:],fmt='%.5E')
outfile.write(b"\n")

G_orig = G.copy()
#write final core tensor
for i in range(nstates):
    for j in range(nstates):
        outfile.write(b"\n")
        ws = "G_orig |{}><{}|\n".format(i,j)
        ws = ws.encode('ASCII')

        outfile.write(ws)
        np.savetxt(outfile,G_orig[i,j,:,:],fmt='%.5E')
outfile.write(b"\n")



#write final core tensor
for i in range(nstates):
    for j in range(nstates):
        outfile.write(b"\n")
        ws = "G_fin |{}><{}|\n".format(i,j)
        ws = ws.encode('ASCII')

        outfile.write(ws)
        np.savetxt(outfile,G_fin[i,j,:,:],fmt='%.5E')
outfile.write(b"\n")

#write mctdh parameters and operator file
oper = open("mctdh_operator","wb")

oper.write(b"#PARAMETERS\n\n")

oper.write(b"#zeroth-order \n")
for i in range(nstates):
    oper.write(b"e_%i = %15.11e\n" % ( i, G_fin[i,i,0,0] ))

oper.write(b"\n\n")

for i in range(nstates):

    oper.write(b"#first-order diagonal S%i\n" % i)

    for j in range(nmodes_fin):

        oper.write(b"k_s%i_m%i = %15.11e\n" % ( i,j+1, G_fin[i,i,0,j+1]*2.0 ))

    oper.write(b"\n")

oper.write(b"\n\n")

for i in range(nstates-1):
    for j in range(i+1,nstates):
        oper.write(b"#first order off-diagonal S%i%i\n" % (i,j))
        for k in range(nmodes_fin):

            oper.write(b"l_s%i%i_m%i = %15.11e\n" % (i,j,k+1, G_fin[i,j,0,k+1]*2.0) )

        oper.write(b"\n")

oper.write(b"\n\n")

for i in range(nstates):
    oper.write(b"#second order diagonal S%i\n" % i)
    for j in range(nmodes_fin):
        for k in range(j,nmodes_fin):

            oper.write(b"m_s%i_m%i%i = %15.11e\n" % (i,j+1,k+1, G_fin[i,i,j+1,k+1]*2.0) )
    oper.write(b"\n")

oper.write(b"\n\n")

for i in range(nstates-1):
    for j in range(i+1,nstates):
        oper.write(b"#second order off-diagonal S%i%i\n" % (i,j))
        for k in range(nmodes_fin):
            for l in range(k,nmodes_fin):

                oper.write(b"n_s%i%i_m%i%i = %15.11e\n" % (i,j,k+1,l+1, G_fin[i,j,k+1,l+1]*2.0) )


oper.write(b"\n\n\n")

oper.write(b"#OPERATOR\n\n")

for i in range(nstates):
    oper.write(b"e_%i\t|1\t S%i&%i\n" % (i,i+1,i+1))

oper.write(b"\n")

for i in range(nmodes_fin):
    oper.write(b"1.0\t|%i\t KE\n" % (i+2))

oper.write(b"\n")

for i in range(nstates):
    for j in range(nmodes_fin):
        oper.write(b"k_s%i_m%i\t|1\tS%i&%i\t|%i\tq\n" % (i,j+1,i+1,i+1,j+2) )

oper.write(b"\n")

for i in range(nstates-1):
    for j in range(i+1,nstates):
        for k in range(nmodes_fin):
            oper.write(b"l_s%i%i_m%i\t|1\tS%i&%i\t|%i\tq\n" % (i,j,k+1,i+1,j+1,k+2) )

oper.write(b"\n")

for i in range(nstates):
    for j in range(nmodes_fin):
        for k in range(j,nmodes_fin):
            if(j == k):
                oper.write(b"0.5*m_s%i_m%i%i\t|1\tS%i&%i\t|%i\tq^2\n" % (i,j+1,k+1,i+1,i+1,j+2) )
            else:
                oper.write(b"m_s%i_m%i%i\t|1\tS%i&%i\t|%i\tq\t|%i\tq\n" % (i,j+1,k+1,i+1,i+1,j+2,k+2) )

oper.write(b"\n")

for i in range(nstates-1):
    for j in range(i+1,nstates):
        for k in range(nmodes_fin):
            for l in range(nmodes_fin):
                if(k == l):
                    oper.write(b"0.5*n_s%i%i_m%i%i\t|1\tS%i&%i\t|%i\tq^2\n" % (i,j,k+1,l+1,i+1,j+1,k+2))
                else:
                    oper.write(b"n_s%i%i_m%i%i\t|1\tS%i&%i\t|%i\tq\t|%i\tq\n" % (i,j,k+1,l+1,i+1,j+1,k+2,l+2))


oper.close()








