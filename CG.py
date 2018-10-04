# conjugate gradient method using back tracking line search.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CGoptimizer(object):

    def __init__(self, p, threshold, alpha, c, rho):

        self.p = p
        self.threshold = threshold

        self.alpha = alpha
        self.c = c
        self.rho = rho

    def initial_values(self):

        # initial x
        self.x = np.zeros((self.p+1, self.p+1))
        # apply boundary conditions to x
        for j in range(0, self.p):
            self.x[0][j] = j / float(self.p)
            self.x[j][self.p] = 1 - (j/float(self.p))
        for j in range(1, self.p + 1):
            self.x[j][0] = j / float(self.p)
            self.x[self.p][j] = 1 - (j/float(self.p))
        # reshape to a vector
        self.x = np.reshape(self.x, ((self.p+1)**2, 1))

        # initial pk
        self.gk = df(self.p, self.x)
        self.pk = -self.gk

        # initial f
        f_initial = f(self.p, self.x)
        print 'Initial f is ' + str(f_initial)

    def compute(self):

        count = 0

        while np.linalg.norm(self.gk) >= self.threshold:

            # update ak
            self.ak = np.dot(self.gk.T, self.pk)/np.dot(self.pk.T, self.gk-df(self.p, self.x - self.gk))
            # update x new
            self.x = self.x + self.ak * self.pk
            # x new gradient
            self.gk_new = df(self.p, self.x)
            # compute beta
            self.beta = np.dot(self.gk_new.T, self.gk_new)/np.dot(self.gk.T, self.gk)
            # compute pk_new
            self.pk = -self.gk_new + self.beta * self.pk
            # update gk
            self.gk = self.gk_new

            count += 1

            print 'The minimized value after iteration ' + str(count+1) + ' is ' + str(f(self.p, self.x))


        fmin = f(self.p, self.x)
        print 'Minimized f is ', fmin

        file = open("fmin", "a")
        file.write("After iteration "+ str(count) + ", optimization finished with " + "f minimum for p = " + str(self.p) + " is " + str(fmin) + "\n")
        file.close()

        np.save("X" + str(self.p), self.x)
        print 'Optimization finished \n'


def f(p, x):

    x = np.reshape(x, (p+1, p+1))
    # calculate minimized objective value f
    f = 0
    for i in range(p):
        for j in range(p):
            f = f + 0.5 * ((x[i][j] - x[i+1][j+1]) ** 2 + (x[i][j+1] - x[i+1][j]) ** 2)
    return f


def df(p, x):

    x = np.reshape(x, (p+1, p+1))
    # find the gradient of x
    df = np.zeros((p+1, p+1))
    for i in range(1, p):
        for j in range(1, p):
            df[i][j] = 4 * x[i][j] - x[i+1][j+1] - x[i-1][j-1] - x[i+1][j-1] - x[i-1][j+1]

    # add boundary conditions to df
    df[0, :] = 0
    df[:, 0] = 0
    df[-1, :] = 0
    df[:, -1] = 0

    # reshape back into a vector
    df = np.reshape(df, ((p+1)**2, 1))
    return df


def plot_x(p, data):
    data = np.reshape(data, (p+1, p+1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx = np.arange(0, p+1, 1)
    yy = xx
    X, Y = np.meshgrid(xx, yy)
    ax.plot_surface(X, Y, data)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #plt.show()
    plt.savefig(str(p) + '.png')


if __name__ == '__main__':

    # parameters for linesearch
    alpha = 1
    c = 0.25
    rho = 0.5

    #p = [101]
    p = np.asarray([101, 201, 301, 401, 501])

    threshold = 1e-10

    for i in range(len(p)):

        print "Start optimizting for p=" + str(p[i])

        fun = CGoptimizer(p[i], threshold, alpha, c, rho)
        fun.initial_values()
        fun.compute()
        plot_x(p[i], np.load("X" + str(p[i]) + ".npy"))
        del fun
