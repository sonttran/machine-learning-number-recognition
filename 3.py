import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA



def load_mnist(dataset="training", selecteddigits=range(10), path='./MNIST_data'):
    #Check training/testing specification. Must be "training" (default) or "testing"
    if dataset == "training":
        fname_digits = path + '/' + 'train-images-idx3-ubyte'
        fname_labels = path + '/' + 'train-labels-idx1-ubyte'
    elif dataset == "testing":
        fname_digits = path + '\\' + 't10k-images.idx3-ubyte'
        fname_labels = path + '\\' + 't10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
        
    #Import digits data
    digitsfileobject = open(fname_digits, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", digitsfileobject.read(16))
    digitsdata = array("B", digitsfileobject.read())
    digitsfileobject.close()

    #Import label data
    labelsfileobject = open(fname_labels, 'rb')
    magic_nr, size = struct.unpack(">II", labelsfileobject.read(8))
    labelsdata=array("B",labelsfileobject.read())
    labelsfileobject.close()
    
    #Find indices of selected digits
    indices=[k for k in range(size) if labelsdata[k] in selecteddigits]
    N=len(indices)
    
    #Create empty arrays for X and T
    X = np.zeros((N, rows*cols), dtype=np.uint8)
    T = np.zeros((N, 1), dtype=np.uint8)
    
    #Fill X from digitsdata
    #Fill T from labelsdata
    for i in range(N):
        X[i] = digitsdata[indices[i]*rows*cols:(indices[i]+1)*rows*cols]
        T[i] = labelsdata[indices[i]]
    return X,T

def vectortoimg(v,show=True):
    plt.imshow(v.reshape(28, 28),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()   
    
    



def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]

def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data
    
def writeExcelData(x,excelfile,sheetname,startrow,startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()
    
def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names







excelfile='./Assignment_3_ Submission_Template.xlsx'
sheets=getSheetNames(excelfile)

        
X, T = load_mnist(dataset="training",selecteddigits=[1,9])
print("Checking multiple training vectors by plotting images.\nBe patient:")
plt.close('all')
fig = plt.figure()
nrows=10
ncols=10
for row in range(nrows):
    for col in range(ncols):
        plt.subplot(nrows, ncols, row*ncols+col + 1)
        vectortoimg(X[np.random.randint(len(T))],show=False)
plt.show()

mu=np.mean(X, axis=0) # get mean vector d feature
Z = X - mu # get mean matrix N sample d feature
C=np.cov(Z,rowvar=False) # get covariance matrix d x d
[λ,V]=LA.eigh(C) # λ is vector of eigenvalues, V is matrix of eigenvectors
λ=np.flipud(λ);V=np.flipud(V.T); # check if eigenvectors are row or column vector
row=V[0,:]; #Check once again 


P=np.dot(Z,V.T) #Principal components
v1 = V[0] # most significant eigenvector
v2 = V[1] # second most significant eigenvector


Nn=len(T[T==1])
Np=len(T[T==9])


TT = np.zeros(len(T)) # pseudo T for looping through XX[TT==1]
for i,b in enumerate(T):
    TT[i] = b

XX = P[:,0:2]

Nn = len(XX[TT==1]) # 1: neg
Np = len(XX[TT==9]) # 9: pos

#1: neg; 9:pos
mup=np.mean(XX[TT==9], axis=0)
mun=np.mean(XX[TT==1], axis=0)


# plot reduced dimension dataset
def plotP1P2(XX, TT):
    from matplotlib  import cm
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_title("Son Tran - 1 VS 9, Reduced data 784 -> 2",fontsize=14)
    ax.set_xlabel("1: Blue          9: Red",fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    zz = np.random.random(len(XX[TT==1][:,0]))
    ax.scatter(XX[TT==1][:,0],XX[TT==1][:,1],s=3,c='blue', marker = 'o');
    ax.scatter(XX[TT==9][:,0],XX[TT==9][:,1],s=3,c='red', marker = 'o' );
    ax.scatter(np.mean(XX[:,0]),np.mean(XX[:,1]),s=100,c='green', marker = 'o' );
    #ax.scatter(XX[:,0],XX[:,1],s=20,c=zz, marker = 'o', cmap = cm.jet );
    plt.show()
plotP1P2(XX,TT)



cp = np.cov(XX[TT==9,].T)
cn = np.cov(XX[TT==1,].T)



p1min = np.amin(XX[:,0])
p1max = np.amax(XX[:,0])
p2min = np.amin(XX[:,1])
p2max = np.amax(XX[:,1])





def Build2DHistogramClassifier(B, X, htmin, htmax, hsmin, hsmax, T):
    HF = np.zeros(B*B).reshape((B,B))
    HM = np.zeros(B*B).reshape((B,B))
    heightBinIndices=(np.round(((B-1)*(X[:,0]-htmin)/(htmax-htmin)))).astype('int32')
    handSpanBinIndices=(np.round(((B-1)*(X[:,1]-hsmin)/(hsmax-hsmin)))).astype('int32')
    for i,b in enumerate(heightBinIndices):
        if T[i]==1: HF[heightBinIndices[i], handSpanBinIndices[i]]+=1
        else : HM[heightBinIndices[i], handSpanBinIndices[i]]+=1
    return HF, HM



# F is 1 == negative, M is 9 == positive
Hn, Hp = Build2DHistogramClassifier(25, XX, p1min,p1max,p2min,p2max,T)


truthp = TT[100]
truthn = TT[109]
#print(TT[100]) # 9 positive
#print(TT[109]) # 1 negative
xp = X[100] # selected a feature vector # 100
zp = xp - mu # substract mean
pp=np.dot(zp,V.T)[0:2] # get 2 first Principal components
rp=np.dot(np.dot(zp,V.T),V)
xrecp=(np.dot(pp,V[0:2,:]))+mu #Reconstruction using 2 components
#print(xp)
#print(zp)
#print(pp)
#print(rp)
#print(xrecp)

xn = X[109] # selected a feature vector # 109
zn = xn - mu # substract mean
pn=np.dot(zn,V.T)[0:2] # get 2 first Principal components
rn=np.dot(np.dot(zn,V.T),V)
xrecn=(np.dot(pn,V[0:2,:]))+mu #Reconstruction using 2 components
#print(xn)
#print(zn)
#print(pn)
#print(rn)
#print(xrecn)



def getFemaleOrMaleGivenHeightAndHandspan(B,x,htmin,htmax,hsmin,hsmax,HF,HM):
    heightBinIndex=(np.round(((B-1)*(x[0]-htmin)/(htmax-htmin)))).astype('int32')
    handSpanBinIndex=(np.round(((B-1)*(x[1]-hsmin)/(hsmax-hsmin)))).astype('int32')
    negCount = HF[int(heightBinIndex),int(handSpanBinIndex)]
    posCount = HM[int(heightBinIndex),int(handSpanBinIndex)]
#    print ('given height & handspan', x)
#    print ('binIndex', heightBinIndex, handSpanBinIndex)
#    print ('F count', negCount)
#    print ('M count', posCount)
    
    if(negCount > posCount):
        label = 1
        prob = negCount/(negCount + posCount)
    else:
        label = 9
        prob = posCount/(negCount + posCount)
    return label, prob



def BayesianProbFemale(x, sigmaF, muF, NF, sigmaM, muM, NM):
    negCount = 1/(2*np.pi*math.sqrt(np.linalg.det(sigmaF)))*np.exp(-0.5*(x-muF).dot((np.linalg.inv(sigmaF)).dot((x-muF).T)))*NF
    posCount = 1/(2*np.pi*math.sqrt(np.linalg.det(sigmaM)))*np.exp(-0.5*(x-muM).dot((np.linalg.inv(sigmaM)).dot((x-muM).T)))*NM
    
    if(negCount > posCount):
        label = 1
        prob = negCount/(negCount + posCount)
    else:
        label = 9
        prob = posCount/(negCount + posCount)
    return label, prob






resultlabelHp, resultprobHp = getFemaleOrMaleGivenHeightAndHandspan(25,pp,p1min,p1max,p2min,p2max,Hn ,Hp)
#print(resultlabelHp, resultprobHp)
resultlabelBp, resultprobBp = BayesianProbFemale(pp, cn, mun, Nn, cp, mup, Np)
#print(resultlabelBp, resultprobBp)
    
resultlabelHn, resultprobHn = getFemaleOrMaleGivenHeightAndHandspan(25,pn,p1min,p1max,p2min,p2max,Hn ,Hp)
#print(resultlabelHn, resultprobHn)
resultlabelBn, resultprobBn = BayesianProbFemale(pn, cn, mun, Nn, cp, mup, Np)
#print(resultlabelBn, resultprobBn)

accuracyH = (resultprobHp + resultprobHn) / 2
accuracyB = (resultprobBp + resultprobBn) / 2


hResults = np.zeros(len(TT))
for i,b in enumerate(XX):
    label, prob = getFemaleOrMaleGivenHeightAndHandspan(25,b,p1min,p1max,p2min,p2max,Hn ,Hp)
    hResults[i] = label


bResults = np.zeros(len(TT))
for i,b in enumerate(XX):
    label, prob = BayesianProbFemale(b, cn, mun, Nn, cp, mup, Np)
    bResults[i] = label
    
print(bResults)

#hResultsOne = hResults[TT==1]
#TTone = TT[TT==1]
#
#hResultsNine = hResults[TT==9]
#TTnine = TT[TT==9]
#print((hResultsNine))
#print((TTnine))

def calAccuracy(cRes, TT):
    count = 0
    for i,b in enumerate(cRes):
        if cRes[i] == TT[i]: count +=1
    return count/len(TT)

accuracyH = calAccuracy(hResults, TT)
accuracyB = calAccuracy(bResults, TT)
print(accuracyH)
print(accuracyB)


#
#
#
#for i,b in enumerate(hResults):
#    if hResults[i] == TT[]:
#        fname_digits = path + '/' + 'train-images-idx3-ubyte'
#        fname_labels = path + '/' + 'train-labels-idx1-ubyte'
#        elif dataset == "testing":
#            fname_digits = path + '\\' + 't10k-images.idx3-ubyte'
#            fname_labels = path + '\\' + 't10k-labels.idx1-ubyte'
#            else:
#                raise ValueError("dataset must be 'testing' or 'training'")
    
#print(TT)
#print((hResults))
#print(pp)


def f(x): return x**2 
print(f(8))

g = lambda x: x**2 
print(g(8))

    


#writeExcelData([mu],excelfile,'Results',2,2)
#writeExcelData([v1],excelfile,'Results',3,2)
#writeExcelData([v2],excelfile,'Results',4,2)
#print("Written mu, v1, v2")
##
#writeExcelData([Np,Nn],excelfile,'Results',6,2)
#print("Written Np, Nn")
#
#writeExcelData([mup,mun],excelfile,'Results',9,2)
#print("Written mup, mun")
#
#writeExcelData(cp,excelfile,'Results',12,2)
#writeExcelData(cn,excelfile,'Results',14,2)
#print("Written cp, cn")
#
#writeExcelData([[p1min,p1max]],excelfile,'Results',17,2)
#writeExcelData([[p2min,p2max]],excelfile,'Results',18,2)
#print("Written p1min, p1max, p2min, p2max")
#
#writeExcelData(Hp,excelfile,'Results',20,2)
#writeExcelData(Hn,excelfile,'Results',46,2)
#print("Written Hp, Hn")
#
#writeExcelData([xp],excelfile,'Results',74,2)
#writeExcelData([zp],excelfile,'Results',75,2)
#writeExcelData([pp],excelfile,'Results',76,2)
#writeExcelData([rp],excelfile,'Results',77,2)
#writeExcelData([xrecp],excelfile,'Results',78,2)
#print("Written xp, zp, rp, pp, xrecp")
#
#writeExcelData([xn],excelfile,'Results',80,2)
#writeExcelData([zn],excelfile,'Results',81,2)
#writeExcelData([pn],excelfile,'Results',82,2)
#writeExcelData([rn],excelfile,'Results',83,2)
#writeExcelData([xrecn],excelfile,'Results',84,2)
#print("Written xn, zn, rn, pn, xrecn")
#
#writeExcelData([truthp],excelfile,'Results',88,2)
#writeExcelData([resultlabelHp],excelfile,'Results',89,2)
#writeExcelData([resultprobHp],excelfile,'Results',89,3)
#writeExcelData([resultlabelBp],excelfile,'Results',90,2)
#writeExcelData([resultprobBp],excelfile,'Results',90,3)
#print("Written truthp, resultlabelHp, resultprobHp, resultlabelBp, resultprobBp")
#
#writeExcelData([truthn],excelfile,'Results',92,2)
#writeExcelData([resultlabelHn],excelfile,'Results',93,2)
#writeExcelData([resultprobHn],excelfile,'Results',93,3)
#writeExcelData([resultlabelBn],excelfile,'Results',94,2)
#writeExcelData([resultprobBn],excelfile,'Results',94,3)
#print("Written truthn, resultlabelHn, resultprobHn, resultlabelBn, resultprobBn")
#
#writeExcelData([accuracyH,accuracyB],excelfile,'Results',97,2)
#print("Written accuracyH, accuracyB")
#
#print("Written everything!")