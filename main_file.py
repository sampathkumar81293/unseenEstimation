import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson
import entropy_est
import warnings
warnings.filterwarnings("ignore")


# min_i find minimum of all the elements in the f where f>0
def find_min_pos(arr):
    for i in range(len(arr)):
        if arr[i] > 0:
            return i;


def find_max_pos(arr):
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] > 0:
            return i;


def make_finger(sample):
    #print(sample)
    diff=np.max(sample)-np.min(sample);

    #create histogram of sample with diff bins
    h=np.histogram(sample,int(diff)+1);

    count_hist=h[0];

    count_hist=np.insert(count_hist,0,0); #to tackle zero case in numpy

    final_hist=np.histogram(count_hist,range(np.max(count_hist)+2))
    return final_hist[0][1:]





def unseen_largescale(f, cutoff):
    f = np.transpose(f);
    size = f.shape[1]

    k = np.matmul(f, np.arange(1, size + 1))  # sample size coming wrong. Just check once
    print("sample_size is:", k, f)

    k = np.sum(k);
    cutoff = 500;
    # -----algorithm parameters -----
    gridFactor = 1.05;  # the grid of probabilities  will be geometric,with this ratio. setting this smaller may slightly increase accuracy, at the cost of speed
    alpha = .5;

    #     #the allowable discrepancy between the returned solution and the "best"(overfit).
    #     #0.5 worked well in all examples we tried, though the results were nearly indistinguishable
    #     #for any alpha between 0.25 and 1.  Decreasing alpha increases the chances of overfitting.

    # xLPmin = 1 / (k * min(10000, max(10, k)));

    xLPmin = 1.0 / (k * max(10, k));



    # min_i find minimum of all the elements in the f where f>0
    min_array = f[0]  # to mask 2D array


    min_i = find_min_pos(min_array);

    if min_i > 0:
        #print("First element is not min")
        xLPmin = min_i * 1.0 / k

    #print(min_i, xLPmin)

    maxLPIters = 1000;  # the 'MaxIter' parameter for Matlab's 'linprog' LP solver.

    x = [0];
    histx = [0];
    fLP = [0] * np.max(np.shape(f));


    f = f[0]
    for i in range(np.max(np.shape(f))):
        if f[i] > 0:
            # print([np.max([1,i-int(np.ceil(np.sqrt(i)))]),np.min([i+int(np.ceil(np.sqrt(i))),np.max(np.shape(f))])])
            wind = [np.max([0, i - int(np.ceil(np.sqrt(i)))]),
                    np.min([i + int(np.ceil(np.sqrt(i))), np.max(np.shape(f))])];

            if sum(f[wind[0]:wind[1]]) < np.sqrt(i):
                x.append(i / k * 1.0);
                histx.append(f[i]);
                fLP[i] = 0;
            else:
                fLP[i] = f[i];



    #     #% If no LP portion, return the empirical histogram

    fmax = find_max_pos(fLP) + 1;

    if min(np.array([fmax]).shape) == 0:
        print("Returning x and h from here");
        x = x[1:];
        histx = histx[2:];
        return x,histx;

    LPMass = 1 - np.matmul(x, np.transpose(histx));


    fLP = fLP[:fmax]
    ceil = int(np.ceil(np.sqrt(fmax)));
    for i in range(ceil):
        fLP.append(0);



    szLPf = len(fLP);


    xLPmax = fmax / float(k);


    find_ceil = int(np.ceil(np.log(xLPmax / xLPmin) / np.log(gridFactor))) + 1;

    xLP = [];
    for i in range(find_ceil):
        xLP.append(xLPmin * np.power(gridFactor, i));


    szLPx = np.max(np.array(xLP).shape);


    A = np.zeros((2 * szLPf, szLPx + 2 * szLPf))
    b = np.zeros((2 * szLPf, 1))
    for i in range(0, szLPf):
        A[2 * i][:szLPx] = poisson.pmf(i + 1, k * np.array(xLP))
        A[2 * i + 1][:szLPx] = (-1) * A[2 * i][:szLPx]

    for i in range(szLPf):
        A[2 * i][szLPx + 2 * i] = -1
        A[2 * i + 1][szLPx + 2 * i + 1] = -1

    for i in range(szLPf):
        b[2 * i] = fLP[i]
        b[2 * i + 1] = -fLP[i]

    Aeq = np.zeros((1, szLPx + 2 * szLPf))
    for i in range(szLPx):
        Aeq[0][i] = xLP[i]

    for i in range(szLPx):
        Aeq[0][i] = Aeq[0][i] / xLP[i] * 1.0

    for i in range(2 * szLPf):
        for j in range(szLPx):
            A[i][j] = A[i][j] / xLP[j] * 1.0

    beq = np.array([LPMass]).reshape(1,1)


    opts = {"maxiter": maxLPIters, "disp": False};

    eq_var1 = [[0 for i in range(1)] for j in range(szLPx + 2 * szLPf)];
    eq_var2 = [[float('Inf') for i in range(1)] for j in range(szLPx + 2 * szLPf)];
    bounds = [(0, None) for i in range(1) for j in range(szLPx + 2 * szLPf)];


    objf = np.zeros((szLPx + 2 * szLPf, 1))

    t_fLP = [0] * len(fLP)
    for i in range(len(fLP)):
        t_fLP[i] = 1.0 / np.sqrt(fLP[i] + 1)


    count = 0;
    for i in range(szLPx, np.shape(objf)[0], 2):
        objf[i] = t_fLP[count];
        count = count + 1;


    count = 0;
    for i in range(szLPx + 1, np.shape(objf)[0], 2):
        objf[i] = t_fLP[count];
        count = count + 1;

    A = np.round(A, 4)
    LP1_output = linprog(np.squeeze(objf), A, b, Aeq, beq, options=opts)
    fval1 = LP1_output.fun
    sol1 = LP1_output.x

    exitflag1=LP1_output['status']
    if exitflag1>1:
        print("LP1 solution was not found.Trying to solve  LP2 anyway.")
        print(LP1_output['message'])

    if exitflag1==1:
        print("Maximum number of iterations reached. try increasing maxiter variable in options.")
        print(LP1_output['message'])
    else:
        print(LP1_output['message'])


    objf2 = np.zeros((1, szLPx + 2 * szLPf));

    for i in range(szLPx):
        objf2[0][i] = 1

    for i in range(szLPx):
        objf2[0][i] = objf2[0][i] / xLP[i];

    #print("Objective value 2:",objf2)

    A2 = [[0 for i in range(len(A[0]))] for j in range(len(A) + 1)];
    A2 = np.zeros((A.shape[0] + 1, A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A2[i][j] = A[i][j]

    for i in range(A.shape[1]):
        A2[-1][i] = objf[i];



    b2 = b;
    b2 = np.append(b, [fval1 + alpha])

    A2=np.round(A2, 4)

    LP2_output = linprog(np.squeeze(objf2), A2, b2, Aeq, beq, options=opts)
    fval2 = LP2_output.fun
    sol2 = LP2_output.x
    exitflag2=LP2_output['message']

    if exitflag2>1:
        print("LP2 solution was not found.")
        print(LP1_output['message'])

    print(xLP)
    print(sol2)
    for i in range(szLPx):
        sol2[i] = sol2[i] / xLP[i];

    for i in range(len(xLP)):
        x.append(xLP[i])

    for i in range(sol2.shape[0]):
        histx.append(sol2[i])

    indices = np.argsort(x);
    sort_x = np.sort(x);


    out_hist = []

    for i in indices:
        out_hist.append(histx[i])



    out_indices = [];
    final_hist = [];
    for i in range(len(out_hist)):
        if out_hist[i] > 0:
            out_indices.append(i);
            final_hist.append(out_hist[i]);

    final_x = [];
    for i in out_indices:
        final_x.append(sort_x[i])
    print(szLPx)
    return final_x,final_hist






def generate_sample(max_value,n):
    #max_value = 10;  # max_value of an element in array
    #n = 20;  # generates an array of size(nxn)
    sample = np.random.randint(1, max_value, size=(n, 1));
    return sample


max_value=100000
count=10000

sample=generate_sample(max_value,count)


#print("Sample is:",sample)
hist = make_finger(sample);
hist = np.reshape(hist, (hist.shape[0], 1))
#print(hist.shape)
f=hist;
#print(f)

x,hist=unseen_largescale(f, 500);
#estimating the histogram of distribution from which it is drawn
print(x,hist)
print("The estimate of histogram of distribution from which sample is drawn is :",hist)


true_entropy=np.log(max_value);

#empiricalEntropy = -f'*(((1:max(size(f)))/k).*log(((1:max(size(f)))/k)))'

temp=np.arange(1,len(f)+1)
temp=np.divide(temp,count*1.0);
empirical_entropy=np.matmul(np.transpose(f),np.multiply(temp,np.log(temp)))
print("Empirical entropy is:",empirical_entropy)

#output entropy of the recovered histogram, [h,x]
estimated_entropy=-np.matmul(hist,np.multiply(x,np.log(x)))
print("Estimated entropy is:",estimated_entropy)

# #output entropy using entropy_estC.m (should be almost the same as above):
estimatedEntropy2 = entropy_est.entropy_estimate(f)
print("The estimated entropy is:",estimatedEntropy2)