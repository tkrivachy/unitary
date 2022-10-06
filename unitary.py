import numpy as np
import tensorflow as tf

def UC(lambda_ = None):
    """
    %
    %   COMPOSITE PARAMETERIZATION of U(d) /// version 1.5 /// 12.04.2011
    %
    %   Original Matlab script (retrieved from
    %   https://www.mathworks.com/matlabcentral/fileexchange/30990-composite-parameterization-of-unitary-groups
    %   on 30.09.2022):
    %
    %   Python translation by Tamás Kriváchy and Flavien Hirsch /// 30.09.2022
    %
    %   Usage : UC(lambda)
    %
    %   lambda - dxd real matrix
    %   lambda(a,b) diagonal components a=b - absolute phases for a in [0,2*pi]
    %   lambda(a,b) upper right components a<b - rotations in a-b plane in [0,pi/2]
    %   lambda(a,b) lower left components a>b - relative phases between a-b in [0,2*pi]
    %
    %   References: --- PLEASE CITE THESE PAPERS WHEN USING THIS FILE ---
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'A composite parameterization of unitary groups, density matrices and subspaces'
    %   arXiv:1004.5252 // J. Phys. A: Math. Theor. 43, 385306 (2010); https://doi.org/10.1088/1751-8113/43/38/385306; https://doi.org/10.1088/1751-8113/43/38/385306
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'Composite parameterization and Haar measure for all unitary and special unitary groups'
    %   arXiv:1103.3408 // J. Math. Phys. 53, 013501 (2012); https://doi.org/10.1063/1.3672064
    %
    """
    d = np.shape(lambda_)[0]
    unitary = np.array([1], dtype=complex)
    for m in np.arange(d-1, 0,- 1).reshape(-1):
        ex1 = np.zeros(d-m)
        ex2 = np.zeros(d-m+1)
        ex2[0] = 1
        if len(unitary.shape)==1:
            unitary = np.hstack([ex1,unitary])
        else:
            unitary = np.hstack([ex1.reshape(-1,1),unitary])
        unitary = np.vstack([ex2,unitary])
        for n in np.arange((d-m+1),1,- 1).reshape(-1):
            A = np.eye(d-m+1)
            A = A.astype(complex)
            A[0,0] = np.cos(lambda_[m-1,n+m-2])
            A[n-1,n-1] = np.exp(1j * lambda_[n + m - 2,m-1]) * np.cos(lambda_[m-1,n + m - 2])
            A[n-1,0] = - np.exp(1j * lambda_[n + m - 2,m-1]) * np.sin(lambda_[m-1,n + m - 2])
            A[0,n-1] = np.sin(lambda_[m-1,n + m - 2])
            unitary = A @ unitary
    for k in np.arange(0,d).reshape(-1):
        unitary[:,k] = unitary[:,k] * np.exp(1j * lambda_[k,k])
    return unitary

def UC_tensorflow(lambda_ = None):
    """
    %
    %   COMPOSITE PARAMETERIZATION of U(d) /// version 1.5 /// 12.04.2011
    %
    %   Original Matlab script (retrieved from
    %   https://www.mathworks.com/matlabcentral/fileexchange/30990-composite-parameterization-of-unitary-groups
    %   on 30.09.2022):
    %   (c) Christoph Spengler 2011, Faculty of Physics, University of Vienna
    %       Contact: christoph.spengler@univie.ac.at
    %
    %   Python translation by Tamás Kriváchy and Flavien Hirsch /// 30.09.2022
    %
    %   Usage : UC_tensorflow(lambda)
    %
    %   lambda - dxd real matrix
    %   lambda(a,b) diagonal components a=b - absolute phases for a in [0,2*pi]
    %   lambda(a,b) upper right components a<b - rotations in a-b plane in [0,pi/2]
    %   lambda(a,b) lower left components a>b - relative phases between a-b in [0,2*pi]
    %
    %   References: --- PLEASE CITE THESE PAPERS WHEN USING THIS FILE ---
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'A composite parameterization of unitary groups, density matrices and subspaces'
    %   arXiv:1004.5252 // J. Phys. A: Math. Theor. 43, 385306 (2010); https://doi.org/10.1088/1751-8113/43/38/385306
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'Composite parameterization and Haar measure for all unitary and special unitary groups'
    %   arXiv:1103.3408 // J. Math. Phys. 53, 013501 (2012); https://doi.org/10.1063/1.3672064
    %
    """
    d = np.shape(lambda_)[0]
    lambda_ = tf.cast(lambda_,tf.dtypes.complex64)
    unitary = tf.ones((1,1), dtype=tf.dtypes.complex64)
    for m in np.arange(d-1, 0,- 1).reshape(-1):
        ex1 = tf.zeros((d-m,1),dtype=tf.dtypes.complex64)
        ex2 = tf.zeros((1,d-m),dtype=tf.dtypes.complex64)
        ex2_prepend = tf.ones((1,1),dtype=tf.dtypes.complex64)
        ex2 = tf.concat((ex2_prepend,ex2),axis=1)
        unitary = tf.concat([ex1,unitary],axis=1)
        unitary = tf.concat([ex2,unitary],axis=0)
        for n in np.arange((d-m+1),1,- 1).reshape(-1):
            A = tf.Variable(tf.eye(d-m+1, dtype = tf.dtypes.complex64))
            A = A[0,0].assign(tf.math.cos(lambda_[m-1,n+m-2]))
            A = A[n-1,n-1].assign(tf.math.exp(1j * lambda_[n + m - 2,m-1]) * tf.math.cos(lambda_[m-1,n + m - 2]))
            A = A[n-1,0].assign(-1* tf.math.exp(1j * lambda_[n + m - 2,m-1]) * tf.math.sin(lambda_[m-1,n + m - 2]))
            A = A[0,n-1].assign(tf.math.sin(lambda_[m-1,n + m - 2]))
            unitary = tf.linalg.matmul(A, unitary)
    multipland = tf.Variable(tf.ones_like(unitary))
    for k in np.arange(0,d).reshape(-1):
        multipland = multipland[:,k].assign(tf.math.exp(1j * lambda_[k,k]))
    unitary = tf.math.multiply(unitary,multipland)
    return unitary


def UCS(lambda_ = None):
    """
    %
    %   COMPOSITE PARAMETERIZATION of SU(d) /// version 1.5 /// 12.04.2011
    %
    %   Original Matlab script (retrieved from
    %   https://www.mathworks.com/matlabcentral/fileexchange/30990-composite-parameterization-of-unitary-groups
    %   on 30.09.2022):
    %   (c) Christoph Spengler 2011, Faculty of Physics, University of Vienna
    %       Contact: christoph.spengler@univie.ac.at
    %
    %   Python translation by Tamás Kriváchy and Flavien Hirsch /// 30.09.2022
    %
    %   Usage : UCS(lambda)
    %
    %   lambda - dxd real matrix (lambda(d,d) is ignored)
    %   lambda(a,b) diagonal components a=b with a,b in {1,..,d-1} - absolute phases for a in [0,2*pi]
    %   lambda(a,b) upper right components a<b - rotations in a-b plane in [0,pi/2]
    %   lambda(a,b) lower left components a>b - relative phases between a-b in [0,pi]
    %
    %   References: --- PLEASE CITE THESE PAPERS WHEN USING THIS FILE ---
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'A composite parameterization of unitary groups, density matrices and subspaces'
    %   arXiv:1004.5252 // J. Phys. A: Math. Theor. 43, 385306 (2010); https://doi.org/10.1088/1751-8113/43/38/385306
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'Composite parameterization and Haar measure for all unitary and special unitary groups'
    %   arXiv:1103.3408 // J. Math. Phys. 53, 013501 (2012); https://doi.org/10.1063/1.3672064
    %
    """
    d = np.shape(lambda_)[0]
    unitary = np.array([1], dtype=complex)
    for m in np.arange(d-1, 0,- 1).reshape(-1):
        ex1 = np.zeros(d-m)
        ex2 = np.zeros(d-m+1)
        ex2[0] = 1
        if len(unitary.shape)==1:
            unitary = np.hstack([ex1,unitary])
        else:
            unitary = np.hstack([ex1.reshape(-1,1),unitary])
        unitary = np.vstack([ex2,unitary])
        for n in np.arange((d-m+1),1,- 1).reshape(-1):
            A = np.eye(d-m+1)
            A = A.astype(complex)
            A[0,0] = np.exp(1j * lambda_[n+m-2, m-1]) * np.cos(lambda_[m-1,n + m - 2])
            A[n-1,n-1] = np.exp(-1j * lambda_[n+m-2, m-1]) * np.cos(lambda_[m-1,n + m - 2])
            A[n-1,0] = - np.exp(-1j * lambda_[n+m-2, m-1]) * np.sin(lambda_[m-1,n + m - 2])
            A[0,n-1] = np.exp(1j * lambda_[n+m-2, m-1]) * np.sin(lambda_[m-1,n + m - 2])
            unitary = A @ unitary
    for k in np.arange(0,d-1).reshape(-1):
        unitary[:,k] = unitary[:,k] * np.exp(1j * lambda_[k,k])
        unitary[:,d-1] = unitary[:,d-1] * np.exp(-1j * lambda_[k,k])
    return unitary

def UCS_tensorflow(lambda_ = None):
    """
    %
    %   COMPOSITE PARAMETERIZATION of SU(d) /// version 1.5 /// 12.04.2011
    %
    %   Original Matlab script (retrieved from
    %   https://www.mathworks.com/matlabcentral/fileexchange/30990-composite-parameterization-of-unitary-groups
    %   on 30.09.2022):
    %   (c) Christoph Spengler 2011, Faculty of Physics, University of Vienna
    %       Contact: christoph.spengler@univie.ac.at
    %
    %   Python translation by Tamás Kriváchy and Flavien Hirsch /// 30.09.2022
    %
    %   Usage : UCS_tensorflow(lambda)
    %
    %   lambda - dxd real matrix (lambda(d,d) is ignored)
    %   lambda(a,b) diagonal components a=b with a,b in {1,..,d-1} - absolute phases for a in [0,2*pi]
    %   lambda(a,b) upper right components a<b - rotations in a-b plane in [0,pi/2]
    %   lambda(a,b) lower left components a>b - relative phases between a-b in [0,pi]
    %
    %   References: --- PLEASE CITE THESE PAPERS WHEN USING THIS FILE ---
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'A composite parameterization of unitary groups, density matrices and subspaces'
    %   arXiv:1004.5252 // J. Phys. A: Math. Theor. 43, 385306 (2010); https://doi.org/10.1088/1751-8113/43/38/385306
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'Composite parameterization and Haar measure for all unitary and special unitary groups'
    %   arXiv:1103.3408 // J. Math. Phys. 53, 013501 (2012); https://doi.org/10.1063/1.3672064
    %
    """
    d = np.shape(lambda_)[0]
    lambda_ = tf.cast(lambda_,tf.dtypes.complex64)
    unitary = tf.ones((1,1), dtype=tf.dtypes.complex64)
    for m in np.arange(d-1, 0,- 1).reshape(-1):
        ex1 = tf.zeros((d-m,1),dtype=tf.dtypes.complex64)
        ex2 = tf.zeros((1,d-m),dtype=tf.dtypes.complex64)
        ex2_prepend = tf.ones((1,1),dtype=tf.dtypes.complex64)
        ex2 = tf.concat((ex2_prepend,ex2),axis=1)
        unitary = tf.concat([ex1,unitary],axis=1)
        unitary = tf.concat([ex2,unitary],axis=0)
        for n in np.arange((d-m+1),1,- 1).reshape(-1):
            A = tf.Variable(tf.eye(d-m+1, dtype = tf.dtypes.complex64))
            A = A[0,0].assign(tf.math.exp(1j * lambda_[n+m-2, m-1]) * tf.math.cos(lambda_[m-1,n + m - 2]))
            A = A[n-1,n-1].assign(tf.math.exp(-1j * lambda_[n+m-2, m-1]) * tf.math.cos(lambda_[m-1,n + m - 2]))
            A = A[n-1,0].assign(-1* tf.math.exp(-1j * lambda_[n+m-2, m-1]) * tf.math.sin(lambda_[m-1,n + m - 2]))
            A = A[0,n-1].assign(tf.math.exp(1j * lambda_[n+m-2, m-1]) * tf.math.sin(lambda_[m-1,n + m - 2]))
            unitary = tf.linalg.matmul(A, unitary)
    multipland = tf.Variable(tf.ones_like(unitary))
    for k in np.arange(0,d-1).reshape(-1):
        multipland = multipland[:,k].assign(tf.math.exp(1j * lambda_[k,k]))
    unitary = tf.math.multiply(unitary,multipland)
    for k in np.arange(0,d-1).reshape(-1):
        multipland = tf.Variable(tf.ones_like(unitary))
        multipland = multipland[:,d-1].assign(tf.math.exp(-1j * lambda_[k,k]))
        unitary = tf.math.multiply(unitary,multipland)
    return unitary

""" Testing """
# lambda_ = np.array([[0,0],[0,0]])
# lambda_ = np.array([[1,0,0],[1,1,1],[1,1,1]])
# lambda_ = np.array([[1,0.5,1,0],[0.3,0.7,1,0],[1.2,1.1,0.4,0.6],[0,0,0,0.45]])
# print(lambda_)
#
# U = UC(lambda_)
# print(U)
#
# U_tf = UC_tensorflow(lambda_)
# print(U_tf)
#
# U = UCS(lambda_)
# print(U)
#
# U = UCS_tensorflow(lambda_)
# print(U)
