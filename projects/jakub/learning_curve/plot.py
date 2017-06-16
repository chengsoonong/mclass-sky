import json
import sys

import matplotlib.pyplot as plt


def main(path):
    with open(path) as f:
        vals = json.load(f)

    sgd_x = vals['sgd_x']
    sgd_y = vals['sgd_y']
    gp_x = vals['gp_x']
    gp_y = vals['gp_y']

    plt.plot(sgd_x, sgd_y, label='SGD')
    plt.plot(gp_x, gp_y, label='GP')
    plt.title('Learning curve')
    plt.xlabel('Training points')
    plt.ylabel('$R^2$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _, path = sys.argv
    main(path)
