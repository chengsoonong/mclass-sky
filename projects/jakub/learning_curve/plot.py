import json
import sys

import matplotlib.pyplot as plt


def main(paths):
    vals = {}
    for path in paths:
        with open(path) as f:
            vals_ = json.load(f)
            vals.update(vals_)

    plt.plot(vals['sgd_x'], vals['sgd_y'], label='SGD')
    plt.plot(vals['gp_x'], vals['gp_y'], label='GP')
    plt.plot(vals['agp_x'], vals['agp_y'], label='AGP')
    plt.title('Learning curve')
    plt.xlabel('Training points')
    plt.ylabel('$R^2$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
