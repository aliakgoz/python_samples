import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import math


def acovf2D(x,y, nlag): 

    # demean=True
    xo = x - x.mean() 
    yo = y - y.mean() 

    n = len(x)   
    lag_len = nlag 
    
    acov = np.empty(lag_len + 1)   
    acov[0] = xo.dot(yo)   

    *******************        
        ********************************

    **************

    return acov 

def osmos_oc(y,ni):

    m = y.shape[1]
    Rcalc = np.zeros((ni,y.shape[1],y.shape[1]))

    for i in range(m):
        for j in range(m):
            Rcalc[:,i,j]= acovf2D(y[:,i],y[:,j],ni-1)

    ******************************
    R = R.transpose()

    #print(R[:,2])
    #print(R.shape)
    Y = []
    ************************
        ******************************

    return Y
# Y > Y[0] - Y[198] 199 değer?

def osmos_hankel(Y):
        
    m = Y[1].shape[1]
    nc = len(Y) #199
    al = math.ceil(((nc+1)/2)) #100
    be = al
    H = np.empty((0,(al)*m),float)
    count=0
    for j in range(al):
        ******************************
        ******************************  
        ******************************
    
        for jj in range(j,indl):
            Ybind = np.concatenate((Ybind,Y[jj]),axis=1)
        H=np.concatenate((H,Ybind),axis=0)
    hdim = H.shape[1]
    H0 = H[:,:-m]
    H1 = H[:,m:]
    return [H0,H1]
def osmos_covERA(y,ni,nL):

    nS=1
    flag=0 

    if 'nS' in locals():
        x8=1
    else:
        x8=0
    if 'nL' in locals():
        x9=1
    else:
        x9=0
    ******************************
    #x9=(len(nL)==0)
    x10 = x8 + x9
    if x10 == 2:
        flag=1
    elif x10 == 1:
        print('error in the nL specification')
    
    t9=nL-nS
    if t9 <= 0:
        print('error = the value of nL must be greater than 1')
    p1 = y.shape[0]
    p2 = y.shape[1]
    #Orient the output so that time runs along the columns
    if p2 >= p1:
        ny = p2
        m = p1
    else:
        m=p2
        ny=p1
        y=y.transpose()
    #m=y.shape[1]
    y = np.conjugate(y.transpose())
    Y = osmos_oc(y,ni)
    print(Y)
    nmk = len(Y) +1
    Y9=Y
    Hs=osmos_hankel(Y9)
    H0=Hs[0]
    H1=Hs[1]
    print("H0")
    print(H0[0:9,0:9])
    #print(Hs[1])
    svdH0 = np.linalg.svd(H0,full_matrices=False)
    R=svdH0[0] #u
    s=svdH0[1] #d
    St=svdH0[2] #v
    St = St.transpose()
    ###################################
    NL=nL-nS+1
    ******************************
    
    C1=[0]*NL
    A1=[0]*NL

    for j in np.arange(NL): #değişiklik
        n = nn[j]
        Rn = R[:,0:n]
        Sig=s[0:n]
        Sn=St[:,0:n]

        if n==1:
            ******************************
        else:
            Ob = np.array(np.dot(Rn,np.diag(np.sqrt(Sig))))
        C1[j] = Ob[0:m,:]
        if n==1:
            ******************************
        else:
            ******************************
        A1[j] = np.dot(np.dot(np.dot(Si,np.conjugate(Rn.transpose())),H1),np.dot(Sn,Si))
        print(A1[0])
    return [A1,C1]
def osmos_n4sid(y,i,nL):
    nS=1
    flag=0
    ###########################################
    # nS ile nL in varlığı kontrol ediliyor sanırım şimdilik geçtim
    ###########################################
    #if 'nS' in locals():
    #    x8=1
    #else:
    #    x8=0
    #if nL in locals():
    #    x9=1
    #else:
    #    x9=0
    #x10 = x8 + x9
    #if x10==2:
    #    flag=1
    #elif x10 == 1:
    #    print('error in the nS nL specification')
    #    #A=c()
    #    #B=c()
    t9 = nL - nS
    
    if t9<=0:
        print('error = the value of nL must be greater than nS')
        #A=c()
        #B=c()
    p1 = y.shape[0]
    p2 = y.shape[1]
    # Orient the output so that time runs along the columns
    if p2 >= p1:
        ny = p2
        m = p1
    else:
        ny=p1
        m=p2
        y=y.transpose()
    # Number of columns in the data matrix
    j = ny-2*i+1  #################### Bunu anlamadım???????
    # Form the four required partitions
    Yp = np.zeros((p2*i,p1-2*i+1))
    for p in np.arange(j):
        in1 = p
        in2 = in1 + i 
     

    Yf = np.zeros((p2*i,p1-2*i+1))
    for p in np.arange(j):
        in1 = p + i
        in2 = in1 + i
        block = y[:,in1:in2]

        vb=block.T.flatten() 
        Yf[:,p]=vb
    Ypp = np.concatenate((Yp,Yf[0:m,:]),axis=0)
    Yfm = Yf
    Yfm = Yfm[m:,:] 
    ******************************
    Oim=np.dot(np.dot(np.dot(Yfm,np.conjugate(Ypp.transpose())),np.linalg.pinv(np.dot(Ypp,np.conjugate(Ypp.transpose())))),Ypp)

    Oisvd=np.linalg.svd(Oi,full_matrices=False)

    U=Oisvd[0] #u
    s4=Oisvd[1] #d
    s = np.diag(s4)
    if flag==1:
        ******************************
        plt.show()
        nL=int(input("max system order to be consider\ "))
        nS=int(input("min system order to be consider\ "))
    NL=nL-nS+1
    ******************************
    
    C1=[0]*NL
    A1=[0]*NL
    for j in range(NL):
        n=nn[j]
        U1=U[:,0:n]
        S1=s[0:n,0:n]
        if n==1:
            ******************************
        else:
            ******************************
        Tau1m = Tau1
        taudim = Tau1.shape[0]
        ******************************

        X1 = np.dot(np.linalg.pinv(Tau1),Oi)
        X1p=np.dot(np.linalg.pinv(Tau1m),Oim)
        Yii=Yf[0:m,:]
        R = np.dot(np.dot(np.concatenate((X1p,Yii),axis=0),np.conjugate(X1.transpose())),np.linalg.pinv(np.dot(X1,np.conjugate(X1.transpose()))))
        A1[j]=R[0:n,:]
        rdim = R.shape[0]
        C1[j]=R[n:rdim,:]
    return [A1,C1]
def OrderPoles4(poles):
    tol = 1e-8
    n=len(poles)
        
    #eliminate any real poles
    Y1=np.where(np.imag(poles) <= tol)[0]
    ******************************
    ******************************
    # eliminate poles with negative imaginary parts
    ******************************
    #poles=[i for i in poles if np.imag(i) > 0]
    ******************************
    # Eliminate unstable poles
    Y3 =  np.where(np.real(poles) >= 0)[0]
    #poles=[i for i in poles if np.real(i) < 0]
    poles = np.delete(poles, Y3)
    #Sort remaining poles by imaginary
    Y = np.array(poles)
    Y = Y[np.argsort(np.imag(poles))]
    Yix = np.argsort(np.imag(poles))
    Opoles=np.take(poles,Yix)
    return [Opoles,Y1,Y2,Y3,Y,Yix]
def rot_mod(psi):
    q=np.array(psi).shape[0]
    ******************************
    ******************************
    ******************************
    eta = (syy-sxx) / (2*sxy)
    term0 = np.sqrt((eta*eta)+1)
    beta = eta + (np.sign(sxy)*term0)
    term1 = (sxx+syy)/2
    term2 = sxy*term0
    ev1 = term1+term2
    ev2 = term1-term2
    eratio = (ev1-ev2)/(ev1+ev2)
    tau = np.arctan(beta)
    mpcw0 = eratio*eratio
    mpcw = 100*np.conjugate(mpcw0.transpose())
    return mpcw

def ceig(bb):
    a9 = -np.real(bb)
    test = np.sign(a9)
    b9 = np.imag(bb)
    r = (b9/a9)**2
    zai = 1/np.sqrt(1+r)
    zai = test*zai
    w=a9/zai
    return [w,zai]

def osmos_StabD(A1,C1,Z,MCI,dt,nL):
    orders = np.arange(1,nL+1)  ## 
    n = len(A1)
    #figure1=np.zeros((n,))
    #figure2=np.zeros((n,)) #kullanılmıyor
    #Poles = np.zeros((n,),dtype=complex)
    #Mosh = np.zeros((n,),dtype=complex)

    figure1=[0]*n
    figure2=[0]*n #kullanılmıyor
    Poles = [0]*n
    Mosh =[0]*n

    for j in np.arange(n):
        ###
        print(A1[j])
        if A1[j].shape==():
            continue
        polD, aa = np.linalg.eig(A1[j])
        
        MS = np.dot(C1[j],aa)
        #take to continuos time
        polC=np.log(polD)/dt
        Opoles4 = OrderPoles4(polC)
        polC1=Opoles4[0]
        #if len(polC1)<3:
        #    continue
        Y1 = Opoles4[1]
        Y2 = Opoles4[2]
        Y3 = Opoles4[3]
        Y = Opoles4[4]
        Yix = Opoles4[5]
        print(MS)
        print(Y1)
        print(Y)
        # Eliminate the corresponding mode shapes
        if len(Y1) > 0:
            MS = np.delete(MS, Y1, axis=1) # axis=1 kolon, axis=0 satır
        if len(Y2) > 0:
            MS = np.delete(MS, Y2, axis=1)
        if len(Y3) > 0:
            MS = np.delete(MS, Y3, axis=1)
        # Order the remaining mode shapes

        MS = MS[:,Yix]
        # Eliminate the poles that have damping higher than Z
        test=np.abs(np.real(polC1)/np.imag(polC1))
        Y4=np.where(test>=Z)
        if len(Y4)>0:
            polC1=np.delete(polC1,Y4)
            MS=np.array(MS)
            MS=np.delete(MS,Y4,1)
        # Eliminate the poles for which the Modal colinearity index is less than
        # MCI
        mpcw=rot_mod(MS)
        Y5=np.where(mpcw<=MCI)
        if len(Y5[0])>0:
            polC1=np.delete(polC1,Y5[0])
            if len(polC1) == 0:
                continue
            MS=np.array(MS)
            MS=np.delete(MS,Y5[0],1)
        # Compute the frequencies and the damping ratios
        w=ceig(polC1)[0]
        freq=w/2/np.pi
        #plotting
        ******************************
        figure1j = []
        for fr in freq:
            figure1j.append(np.array([fr,orders[j]]))
        figure1[j] = figure1j
        Poles[j]=polC1
        Mosh[j]=MS # Mosh da bazı değerler R dakinin -1 katı çıkmış... ör. Mosh[10]
    #Figure 1
    #print(len(figure1))
    for j in range(len(figure1)):
        if figure1[j] == 0:
            continue
        else:
            for i in range(len(figure1[j])):
                plt.plot(figure1[j][i][0],figure1[j][i][1],marker='x')
    
    plt.xlabel="modal frequency (Hz)"
    plt.ylabel="order"
    plt.xlim=[0,6]
    plt.ylim=[0,n]
    plt.grid(True)
    plt.show()
    return [Poles,Mosh]

def osmos_optR(tt):
    R = np.real(tt)
    I = np.imag(tt)
    rtr =  np.conj( np.dot(R.T,R)-np.conj(np.dot(I.T,I)) )
    if isinstance(rtr, np.ndarray):
        H = -2 *np.dot( np.conj(R.T) , np.dot( I , np.linalg.inv( rtr ) ) )
    else:
        H = -2 *np.dot( np.conj(R.T) , np.dot( I , rtr**(-1) ) )
    TZ = np.arctan(H)
    Z1 = TZ/2
    Z2 = Z1 - np.pi/2
    ******************************
    T2 = np.dot( -np.sin(2*Z2), np.conj(np.dot(R.T,I)))

    if T1 > T2:
        Z = Z1
    else:
        Z = Z2

    rt = np.exp(Z*1j)
    t = np.dot(tt,rt)
    t = np.real(t)
    return t

def osmos_select(Poles,Mosh,SM,SPM):
    npp = SM.shape[0]
    freq = [0]*npp 
    Damp = [0]*npp
    Rv = []
    
    for j in range(npp):
        order = SM[j,1]-1
        polN = SM[j,0]-1

        if len(Poles[order])<polN:
            print("This order ("+ str(order) + ") doesn't include required number of poles("+ str(polN) +")\n")
            print("Please select a new order that includes higher poles.")
        else:
            p = Poles[order][polN]
            w = ceig(p)[0]
            freq[j] = w/(2*np.pi)
            Damp[j] = ceig(p)[1]
            tt = Mosh[order][:,polN]
            # optimize the normalization to real
            t = osmos_optR(tt)
            Y = np.where(abs(t)==max(abs(t)))
            Rv.append(t/t[Y])
    Rv = np.array(Rv)
    lws = np.unique(SPM[:,0])
    print(SPM[:,0])
    nls = len(lws)
    lws.sort()
    kosul = 0
    for j in range(nls):
        lc = lws[j]
        Y = np.where(SPM[:,0]==lc)[0]
        x1 = SPM[Y,1]
        y1 = SPM[Y,2]
        z1 = SPM[Y,3]
        Q = np.concatenate((np.cos(z1*np.pi/180)[np.newaxis].T,np.sin(z1*np.pi/180)[np.newaxis].T,(x1*np.sin(z1*np.pi/180)-y1*np.cos(z1*np.pi/180))[np.newaxis].T),axis=1) # a = np.array([1,2,3]) a[np.newaxis].T = [[1],[2],[3]]
        nr = Q.shape[0]
        if nr<3:
            print('Warning:') 
            print('The Mode Shape Expressed at the Origin Specified in SPM')
            print('Is Not Unique in Level');
            print(lc)
        print(Y)
        print(Rv)
        rr = np.dot(np.linalg.pinv(Q),Rv[:,Y].T)
        if kosul == 0:
            RVT = rr
            kosul = 1
        else:
            RVT = np.concatenate((RVT,rr),axis=0)
    
    return [freq,Damp,RVT]


******************************
******************************

print(Measured_data.shape)

dt=0.02
ni=200
i=60
nL=60
Z=0.07
MCI= 90

#AC1=osmos_covERA(Measured_data,ni,nL)

AC1=osmos_n4sid(Measured_data,i,nL)
A1=AC1[0]
C1=AC1[1]

Stbdout=osmos_StabD(A1,C1,Z,MCI,dt,nL)
Poles=Stbdout[0]
Mosh=Stbdout[1]
print(Poles)
******************************
SPM = np.array(SPM['SPM'])
SM=np.array([[1,50],[2,50],[3,50],[4,50],[5,50]])
Slctout=osmos_select(Poles,Mosh,SM,SPM)
RVT = Slctout[2]
for j in range(SM.shape[0]):
    RVT[:,j] = RVT[:,j]/np.linalg.norm(RVT[:,j], ord=2)

print(RVT)
