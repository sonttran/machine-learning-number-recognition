import numpy as np
import numpy.linalg as LA



X = np.array([[72.,101.,94.],[50.,96.,70.],[14.,79.,10.],[8.,70.,1.]], np.float64);
print(X)


μ=np.mean(X,axis=0);print(μ)

print(type(μ))
print((μ.shape))

Z=X-μ;print(Z)

print(np.mean(Z,axis=0))

C=np.cov(Z,rowvar=False);print(C)


[λ,V]=LA.eigh(C);print(λ,'\n\n',V)

row=V[0,:];col=V[:,0]; # col is the eigenvectors

rowTest = np.dot(C,row)-(λ[0]*row) #If the matrix product C.row is the same as λ[0]*row, this should evaluate to [0,0,0]
print(rowTest)
colTest = np.dot(C,col)-(λ[0]*col) #If the matrix product C.col is the same as λ[0]*col, this should evaluate to [0,0,0]
print(colTest)

λ=np.flipud(λ);V=np.flipud(V.T); # flip so the most significant vectors are now the first ones and now the eigenvectors are in rows
print(λ)
print(V)

row=V[0,:]
rowTest = np.dot(C,row)-(λ[0]*row) #If the matrix product C.row is the same as λ[0]*row, this should evaluate to [0,0,0]
print(rowTest) # all zeros, good to go

rowTest = np.allclose(np.dot(C,row),λ[0]*row) # another way to test
print(rowTest) # true, good to go


P=np.dot(Z,V.T);print(P) #Principal components
R=np.dot(P,V);print(R-Z) #Z is recovered since R-Z is seen to contain very small values

Xrec=R+μ;print(Xrec-X) #X is recovered since Xrec-X is seen to contain very small values


Xrec1=(np.dot(P[:,0:1],V[0:1,:]))+μ;print(Xrec1) #Reconstruction using 1 component
Xrec2=(np.dot(P[:,0:2],V[0:2,:]))+μ;print(Xrec2) #Reconstruction using 2 components
Xrec3=(np.dot(P[:,0:3],V[0:3,:]))+μ;print(Xrec3) #Reconstruction using 3 components (X is recovered)
#
#print(V)
#print(V[0:1,:])
#
#print(P)
#print(P[:,0:1])