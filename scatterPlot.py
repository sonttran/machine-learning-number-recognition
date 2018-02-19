import matplotlib.pyplot as plt




# install the dependencies, plug in your X and T and you are good to go

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

    plt.show()

plotP1P2(XX,TT)