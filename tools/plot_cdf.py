import matplotlib.pyplot as plt
import numpy as np
import csv

def main():
    ll = []
    cl = []

    with open("stats/client_times_ll.csv", 'r') as csv_file:
        csv_r = csv.reader(csv_file)
        for row in csv_r:
            for elmt in row:
                ll.append(float(elmt))
    with open("stats/client_times_cl.csv", 'r') as csv_file:
        csv_r = csv.reader(csv_file)
        for row in csv_r:
            for elmt in row:
                cl.append(float(elmt))        

    Xll = np.array( sorted(ll) )
    Yll = np.exp( -np.power(Xll,2) )
    CYll = np.cumsum(Yll) / sum(Yll)
    
    Xcl = np.array( sorted(cl) )
    Ycl = np.exp( -np.power(Xcl,2) )
    CYcl = np.cumsum(Ycl) / sum(Ycl)

    Xll = np.insert(Xll,0,0)
    Xcl = np.insert(Xcl,0,0)
    Xll = np.append(Xll,max([max(Xll), max(Xcl)]))
    Xcl = np.append(Xcl,max([max(Xll), max(Xcl)]))

    CYll = np.insert(CYll,0,0)
    CYcl = np.insert(CYcl,0,0)
    CYll = np.append(CYll,1)
    CYcl = np.append(CYcl,1)

    my_dpi=100
    plt.figure(figsize=(800.0/my_dpi, 400.0/my_dpi), dpi=my_dpi)
    plt.plot(Xll, CYll, label="MLP")
    plt.plot(Xcl, CYcl, label="Conv2d")

    plt.xlabel("Seconds")
    plt.ylabel("CDF")
    plt.title("Client Computation Duration")

    plt.legend()

    # plt.show()
    plt.savefig("client_times_gen.png", transparent=True, dpi=100)
    #plot_cdf(ll, cl)
        
def plot_cdf(x1, x2):
    for x in [x1,x2]:
        X = np.array( sorted(x) )
        Y = np.exp( -np.power(X,2) )
        CY = np.cumsum(Y) / sum(Y)
        plt.plot(X, CY)
        # plt.plot(X,Y)
    plt.show()


if __name__=="__main__":
    main()