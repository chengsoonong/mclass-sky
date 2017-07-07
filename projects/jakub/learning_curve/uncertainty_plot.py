import json
import sys

import matplotlib.pyplot as plt


def main(path):
    vals = {}
    with open(path) as f:
        vals = json.load(f)

    plt.plot(vals['gp_x'], vals['gp_y'])
    plt.title('Uncertainty curve')
    plt.xlabel('Training points')
    plt.ylabel('Average uncertainty')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
