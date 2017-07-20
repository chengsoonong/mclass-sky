import json
import sys

import matplotlib.pyplot as plt


def main(path):
    with open(path) as f:
        plots = json.load(f)

    for plot in plots:
        plt.plot(plot['x'], plot['y'], label=plot['label'])
    plt.title('Learning curve')
    plt.xlabel('Training points')
    plt.ylabel('$R^2$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
