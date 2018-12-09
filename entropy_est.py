import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

def find_min_pos(arr):
    for i in range(len(arr)):
        if arr[i] > 0:
            return i;
    return -1;


def find_max_pos(arr):
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] > 0:
            return i;
    return -1;

# def ent(x,histx,k):
#
#     np_x=np.array(x);
#     np_histx=np.array(histx)
#
#     x_mul=np.multiply(np_x, np.log(np_x))
#     entropy=-np.dot(histx,np.transpose(x_mul));
#     entropy=entropy+(np.sum(np_histx)/(2*k));
#
#     return entropy;


def find_pos_indices(arr):
    out=[];
    for i in range(len(arr)):
        if arr[i]>0:
            out.append(i);
    return out;

def entropy_estimate(f):
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

    fmax = find_max_pos(fLP);

    if fmax== -1:
        print("Returning x and h from here");
        x = x[1:];
        histx = histx[1:];
        #print(x,histx)

        np_x = np.array(x);
        np_histx = np.array(histx)
        x_mul = np.multiply(np_x, np.log(np_x))
        entropy = -np.matmul(histx, np.transpose(x_mul));
        entropy = entropy + (np.sum(np_histx) / (2 * k));
        return entropy;

    fmax=fmax+1
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

    beq = np.array([LPMass]).reshape(1, 1)

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

    sol2=None;

    if min_i<1:
        #print("Prining Min_i less loop:", min_i)

        objf2 = np.zeros((1, szLPx + 2 * szLPf));

        for i in range(szLPx):
            objf2[0][i] = 1

        for i in range(szLPx):
            objf2[0][i] = objf2[0][i] / xLP[i];

    # print("Objective value 2:",objf2)

        A2 = [[0 for i in range(len(A[0]))] for j in range(len(A) + 1)];
        A2 = np.zeros((A.shape[0] + 1, A.shape[1]))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A2[i][j] = A[i][j]

        for i in range(A.shape[1]):
            A2[-1][i] = objf[i];

        b2 = b;
        b2 = np.append(b, [fval1 + alpha])

        A2 = np.round(A2, 4)

        LP2_output = linprog(np.squeeze(objf2), A2, b2, Aeq, beq, options=opts)
        fval2 = LP2_output.fun
        sol2 = LP2_output.x
        #print("1st loop",LP2_output)
    else:
        sol2=sol1;


    for i in range(szLPx):
        sol2[i] = sol2[i] / xLP[i];


    if find_min_pos(x)==-1:
        xlp_arr=np.array(xLP)
        xlp_mul=np.multiply(xlp_arr,np.log(xlp_arr));
        ent=-np.matmul(np.transpose(sol2[:szLPx]),np.transpose(xlp_mul))
        return ent;
    else:
        print("Reached 2nd entropy method")
        indices=find_pos_indices(x);
        x=np.array(x);
        histx=np.array(histx);
        xlp_arr = np.array(xLP)
        xlp_mul = np.multiply(xlp_arr, np.log(xlp_arr));

        ent1=-np.matmul(histx[indices],np.transpose(np.multiply(x[indices],np.log(x[indices]))))
        ent2=(np.sum(histx[indices]))/2*k
        ent3=np.matmul(np.transpose(np.array(sol2[1:szLPx])),np.transpose(xlp_mul));
        ent=ent1+ent2+ent3;

        return ent;
