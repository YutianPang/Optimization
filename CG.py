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
        #print 'Initial x is \n', self.x

        # initial pk
        self.pk = 0
        self.pk = self.pk - df(self.p, self.x)
        #print 'Initial pk is \n', self.pk

        # initial f
        f0 = f(self.p, self.x)
        print 'Initial f is ' + str(f0)

    def compute(self):

        count = 0

        while(np.round(np.linalg.norm(df(self.p, self.x))/.5)*.5) >= self.threshold:

            grad = df(self.p, self.x)

            # backtracking line search
            pk = df(self.p, self.x)
            # procedure 3.1, page 41
            while f(self.p, self.x - self.alpha * pk) > (f(self.p, self.x) - self.c * self.alpha * (np.linalg.norm(pk)) ** 2):  # ???
                self.alpha = self.rho * self.alpha

            # gradient descent
            self.x = self.x + self.alpha * self.pk

            # calculate new gradient
            grad_new = df(self.p, self.x)

            # check beta
            beta = np.linalg.norm(grad_new.T*(grad_new - grad))/(np.linalg.norm(grad))**2

            if beta < 0:
                beta = 0

            self.pk = -grad_new + beta * self.pk

            print 'The minimized value after iteration ' + str(count+1) + ' is ' + str(f(self.p, self.x))
            count += 1

        fmin = f(self.p, self.x)
        print 'Minimized f is ', fmin

        file = open("fmin", "a")
        file.write("f minimum for p = " + str(self.p) + " is " + str(fmin) + "\n")
        file.close()

        #print 'Corresponding x is \n', self.x
        np.save("X" + str(self.p), self.x)
        print 'Optimization finished \n'


def f(p, x):

    # calculate minimized objective value f
    f = 0
    for i in range(p):
        for j in range(p):
            f = f + 0.5 * ((x[i][j] - x[i+1][j+1]) ** 2 + (x[i][j+1] - x[i+1][j]) ** 2)
    return f


def df(p, x):
    # find the gradient of x
    df = np.zeros((p+1, p+1))
    for i in range(1, p):
        for j in range(1, p):
            df[i][j] = 4 * x[i][j] - x[i+1][j+1] - x[i-1][j-1] - x[i+1][j-1] - x[i-1][j+1]
    return df


def plot_x(p, data):
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

    # paramaters for line search
    alpha = 1
    c = 0.25
    rho = 0.5

    p = np.asarray([101, 201, 301, 401, 501])

    threshold = 1e-10

    for i in range(len(p)):

        print "Start optimizting for p=" + str(p[i])

        fun = CGoptimizer(p[i], threshold, alpha, c, rho)
        fun.initial_values()
        fun.compute()
        plot_x(p[i], np.load("X" + str(p[i]) + ".npy"))
        del fun
