"""Shared constants and default perturbation settings."""

DEFAULT_NUM_NODES = 3
DEFAULT_NUM_VERSIONS = 5
DEFAULT_NODE_FLOPS_CAPACITY = 200 * 10**9
DEFAULT_MAX_NODE_PARAMS = 150_000_000
DEFAULT_COMM_DELAY_CROSS_NODE = 0.02

DEFAULT_RATE_VALUES = [100, 200, 300, 400, 500, 600, 700, 800]
DEFAULT_LENGTH_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_TASK_TYPE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]

DEFAULT_ALGORITHMS = [
    "our",
    "ffd-m",
    "cds-m",
    "random-m",
    "greedy-m",
    "lego",
    "drs",
]
