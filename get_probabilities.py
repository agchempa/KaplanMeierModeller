import csv
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt

def get_probabilities(arm):
    death_fname = "interpolated/{}_death.csv".format(arm)
    pf_fname = "interpolated/{}_pf.csv".format(arm)

    deaths = []
    with open(death_fname, 'r') as ropen:
        reader = csv.reader(ropen)
        for row in reader:
            month, percent = row
            percent = float(percent)
            deaths.append(percent)

    pfs = []
    with open(pf_fname, 'r') as ropen:
        reader = csv.reader(ropen)
        for row in reader:
            month, percent = row
            percent = float(percent)
            pfs.append(percent)

    num_months = max(len(deaths), len(pfs))
    data = np.zeros((num_months, 3))
    data[:len(pfs), 0] = pfs
    data[:len(deaths), 2] = deaths
    data[:, 1] = data[:, 2] - data[:, 0]
    data[:, 2] = 1 - data[:, 0] - data[:, 1]

    parameters = np.zeros((data.shape[0], 5))
    for idx, row in enumerate(data):
        if idx == 0: continue

        prev_F, prev_P, prev_D = data[idx - 1, :]
        F, P, D = data[idx, :]

        A = np.zeros((5, 5))
        A[0, :] = [prev_F, 0, 0, 0, 0]
        A[1, :] = [0, prev_F, 0, prev_P, 0]
        A[2, :] = [0, 0, prev_F, 0, prev_P]
        A[3, :] = [1]*3 + [0]*2
        A[4, :] = [0]*3 + [1]*2

        b = np.array([F, P, D - prev_D, 1, 1])

        def evaluate(x):
            return np.linalg.norm(np.dot(A, x) - b)**2

        bounds = [[0, 1]] * 5
        res = minimize(evaluate, [0.3]*3 + [0.5]*2, method='SLSQP', bounds=bounds)

        parameters[idx, :] = res.x

    modeled_data = np.zeros(data.shape)
    modeled_data[0, :] = [1, 0 , 0]
    for idx, row in enumerate(modeled_data):
        if idx == 0: continue

        prev_F, prev_P, prev_D = modeled_data[idx - 1, :]

        A = np.zeros((5, 5))
        A[0, :] = [prev_F, 0, 0, 0, 0]
        A[1, :] = [0, prev_F, 0, prev_P, 0]
        A[2, :] = [0, 0, prev_F, 0, prev_P]
        A[3, :] = [1]*3 + [0]*2
        A[4, :] = [0]*3 + [1]*2

        x = parameters[idx, :]
        b = np.dot(A, x)
        modeled_data[idx, :] = b[:3]
        modeled_data[idx, 2] += modeled_data[idx - 1, 2]

    # print(data)
    print(modeled_data)

    plt.figure()
    x_range = list(range(data.shape[0]))
    plt.plot(x_range, data[:, 0], 'g--', label="Unprogressed; real")
    plt.plot(x_range, data[:, 1], 'y--', label="Progressed; real")
    plt.plot(x_range, data[:, 2], 'r--', label="Dead; real")

    plt.plot(x_range, modeled_data[:, 0], 'g-', label="Unprogressed; modeled")
    plt.plot(x_range, modeled_data[:, 1], 'y-', label="Progressed; modeled")
    plt.plot(x_range, modeled_data[:, 2], 'r-', label="Dead; modeled")

    plt.title("{} treatment".format(arm))
    plt.xlabel("Months")
    plt.ylabel("Proportion of patients")

    plt.legend()
    plt.savefig("{}.png".format(arm))

    np.savetxt("parameters/{}.csv".format(arm), parameters, delimiter=",")
        

def main():
    for arm in ["control",
    # "doublet"
    # "triplet"
    ]:
        get_probabilities(arm)

if __name__ == '__main__':
    main()