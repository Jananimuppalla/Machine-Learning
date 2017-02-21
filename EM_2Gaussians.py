import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

# getting data into numpy array




filename = "./data2.txt"
data = np.genfromtxt(filename)
# setting randomly 3 means.
mean_1 = random.choice(data)
mean_2 = random.choice(data)
# mean_3 = random.choice(data)
std_1 = random.randint(1,5)
std_2 = random.randint(1,5)
# std_3 = random.randint(1,5)

max_iter = 0
tol = 0.5
ll_old = 0
logobj = []
w_1 = 0.50
w_2 = 0.50
# w_3 = 0.33
ll_1=np.zeros(len(data))
ll_2=np.zeros(len(data))
ll_3=np.zeros(len(data))

dnom=np.zeros(len(data))


#weights
log_old=0
log_new=1

while(log_old!=log_new ):
    log_old=log_new
    max_iter+=1
    for x in range(0,len(data)):
        #E STEP
        P_1 = w_1*stats.norm(mean_1,std_1).pdf(data[x])
        P_2 = w_2*stats.norm(mean_2,std_2).pdf(data[x])
        # P_3 = w_3*stats.norm(mean_3,std_3).pdf(data[x])
        # P_1 = w_1 * P_1
        # P_2 = w_2 * P_2
        # P_3 = w_3 * P_3
        # denominator
        denominator = (P_1 + P_2 )
        dnom[x]=denominator
        # print denominator
        ll_1[x] = P_1 / denominator
        ll_2[x] = P_2 / denominator
        # ll_3[x] = P_3 / denominator
    N_1=sum(ll_1)
    N_2=sum(ll_2)
    # N_3=sum(ll_3)
    # print N_1
    # print N_2
    # print N_3

    # M STEP

    # MEANS
    mean_1=sum(ll_1 * data)/N_1
    mean_2=sum(ll_2 * data)/N_2
    # mean_3=sum(ll_3 * data)/N_3

    #Standard deviation
    std_1= math.sqrt((sum(ll_1 * data**2)/N_1) - (mean_1**2))
    std_2= math.sqrt((sum(ll_2 * data**2)/N_2) - (mean_2**2))
    # std_3= math.sqrt((sum(ll_3 * data**2)/N_3) - (mean_3**2))

    # weights
    w_1=N_1/len(data)
    w_2=N_2/len(data)
    # w_3=N_3/len(data)

    # log likelihood
    log_new=round(sum(np.log(dnom)),9)
    logobj.append(log_new)
print logobj
print max_iter
print "mean_1 = %.2f, mean_2 = %.2f" % (mean_1, mean_2)
print "weight_1 = %.2f, weight_2 = %.2f" % (w_1, w_2)
print "standard_1 = %.2f, standard2 = %.2f" % (std_1, std_2)
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.scatter(range(max_iter),logobj)
plt.show()
