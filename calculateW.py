from math import *
import numpy as np
import tqdm

sliceNumber = 1000
epsilon = 2*pi / sliceNumber

tr = 60
t0=45
t1=20
t2=80
t3=30
tR=0

w=cos(2*pi/3)+sin(2*pi/3)*1j
tpp=15
tmm=tpp
tzz=10
tpm=12
tzm=0
tzp=tzm
kz=5
l=30

def H(kx, ky):
    k1 = kx
    k2 = -kx + sqrt(3)*ky
    k2 *= 0.5
    k3 = -kx - sqrt(3)*ky
    k3 *= 0.5
    kk1 = (k1-k3)/3 - kz
    kk2 = (k2-k1)/3 - kz
    kk3 = (k3-k2)/3 - kz

    h11 = -2*t1*( cos(kx) + cos(k2) + cos(k3)) + 2*tR*( sin(kx) + sin(k2) + sin(k3) ) - 0.5*l
    h12 = 2*t2*( cos(kx) + w*cos(k2) + w*w*cos(k3) )
    h13 = -tpp*(cos(kk1)+cos(kk2)+cos(kk3)+sin(kk1)*1j+sin(kk2)*1j+sin(kk3)*1j)
    h14 = -tpm*(cos(kk1)+w*cos(kk2)+w*w*cos(kk3) + sin(kk1)*1j+w*sin(kk2)*1j+w*w*sin(kk3)*1j)
    h21 = 2*t2*( cos(kx) + w*w*cos(k2) + w*cos(k3) )
    h22 = -2*t1*(cos(k1)+cos(k2)+cos(k3))-2*tR*(sin(k1)+sin(k2)+sin(k3))+l/2
    h23 = -tpm*( cos(kk1)+w*w*cos(kk2)+w*cos(kk3) + sin(kk1)*1j+w*w*sin(kk2)*1j+w*sin(kk3)*1j)
    h24 = -tpp*(cos(kk1)+cos(kk2)+cos(kk3)+sin(kk1)*1j+sin(kk2)*1j+sin(kk3)*1j)
    h31 = -tpp*(cos(kk1)+cos(kk2)+cos(kk3)-sin(kk1)*1j-sin(kk2)*1j-sin(kk3)*1j)
    h32 = -tpm*( cos(kk1)+w*cos(kk2)+w*w*cos(kk3) - sin(kk1)*1j-w*sin(kk2)*1j-w*w*sin(kk3)*1j)
    h33 = -2*t1*(cos(k1)+cos(k2)+cos(k3))-2*tR*(sin(k1)+sin(k2)+sin(k3))-l/2
    h34 = 2*t2*(cos(k1)+w*cos(k2)+w*w*cos(k3))
    h41 = -tpm*(cos(kk1)+w*w*cos(kk2)+w*cos(kk3) - sin(kk1)*1j-w*w*sin(kk2)*1j-w*sin(kk3)*1j)
    h42 = -tpp*(cos(kk1)+cos(kk2)+cos(kk3)-sin(kk1)*1j-sin(kk2)*1j-sin(kk3)*1j)
    h43 = 2*t2*(cos(k1)+w*w*cos(k2)+w*cos(k3))
    h44 = -2*t1*( cos(kx) + cos(k2) + cos(k3)) + 2*tR*( sin(kx) + sin(k2) + sin(k3) ) + 0.5*l

    ans = np.array([[h11,h12,h13,h14],[h21,h22,h23,h24],[h31,h32,h33,h34],[h41,h42,h43,h44]])
    return ans

Hlib = []
for x in range(0,sliceNumber):
    Htmp = []
    for y in range(0,sliceNumber):
        kx = 2*pi*x/sliceNumber
        ky = 2*pi*y/sliceNumber
        
        Htmp.append(H(kx,ky))
    Hlib.append(Htmp)
 
def W(n, kx, ky):
    H1 = Hlib(kx, ky)
    Msize = int(sqrt(H1.size))
    H1x = Hlib(kx+epsilon, ky)
    H1y = Hlib(kx, ky+epsilon)
    dHdx = (H1x - H1) / epsilon
    dHdy = (H1y - H1) / epsilon
    E = np.linalg.eig(H1)
    Es = np.sort(E[0])

    if n<1 or n>Msize:
        raise "Unappropriate n"

    for nr in range(0,Msize):
        if Es[n-1] == E[0][nr]:
            break
    if nr >= Msize:
        nr = Msize-1
    # nr은 eigenvalue가 정렬되지 않았기 때문에 n 번째에 해당되는 인덱스가 nr
    #print("E : ",E[0])
    #print("nr : ",nr)
    #print("Ev : ",E[1][:,nr])

    ans = 0
    for npr in range(0,Msize):
        if nr == npr:
            continue

        EVnconj = np.conjugate(E[1][:,nr])
        EVnpconj = np.conjugate(E[1][:,npr])
        EVn = E[1][:,nr]
        EVnp = E[1][:,npr]

        t1 = np.dot(np.dot(EVnconj, dHdx), EVnp)
        t2 = np.dot(np.dot(EVnpconj, dHdy), EVn)
        t3 = np.dot(np.dot(EVnconj, dHdy), EVnp)
        t4 = np.dot(np.dot(EVnpconj, dHdx), EVn)
        
        ans += (t1*t2 - t3*t4) / (E[0][nr] - E[0][npr] + 10**(-15))**2
    
    #print("실수부를 취하면 : ",ans.real)
    # -> returns around e-19, etc
    return ans.imag * -1

#test code
"""
H1 = H(2,2)
H2 = H(-2,-2)
E1 = np.linalg.eig(H1)
E2 = np.linalg.eig(H2)
print("E1 : ",E1[0])
print("E2 : ",E2[0])
print("W_1(1,1) : ",W(1,1,1))
print("W_1(-1,-1) : ",W(1,-1,-1))
print("W_1(2,2) : ",W(1,2,2))
print("W_1(-2,-2) : ",W(1,-2,-2))
print("W_1(3,3) : ",W(1,3,3))
print("W_1(-3,-3) : ",W(1,-3,-3))
print("W_1(4,4) : ",W(1,4,4))
print("W_1(-4,-4) : ",W(1,-4,-4))
print("W_1(5,5) : ",W(1,5,5))
print("W_1(-5,-5) : ",W(1,-5,-5))
print("Evsize : ",E1[1][0].size)

Econj = np.conjugate(E1[1][0])
tmp = np.dot(Econj, E1[1][0])
print("Evec size ",tmp)
#-> returns exactly 1
"""

#W1 = []
W2 = []
W3 = []
#W4 = []
for y in tqdm.tqdm(range(sliceNumber)):
    #W1r = []
    W2r = []
    W3r = []
    #W4r = []
    for x in range(sliceNumber):
        kx = 2*pi*x/sliceNumber
        ky = 2*pi*y/sliceNumber
        #W1r.append(W(1,kx,ky))
        W2r.append(W(2,kx,ky))
        W3r.append(W(3,kx,ky))
        #W4r.append(W(4,kx,ky))
    #W1.append(W1r)
    W2.append(W2r)
    W3.append(W3r)
    #W4.append(W4r)

#np.savetxt("W1.csv",W1,delimiter=", ")
np.savetxt("W2.csv",W2,delimiter=", ")
np.savetxt("W3.csv",W3,delimiter=", ")
#np.savetxt("W4.csv",W4,delimiter=", ")
