import sys
from argparse import ArgumentParser
from math import prod
from numbers import Real


def check_arguments(arguments: list[tuple[str, list]]):
    assert isinstance(arguments, list), "arguments must be a list of key-value pairs."
    assert all(len(x) == 2 and isinstance(x[1], list) for x in arguments), (
        "All elements in arguments must be key-value pairs,with value being a list."
    )


def check_index(arguments, index: int):
    assert isinstance(index, int), "index must be an integer."
    assert index >= 0, "index must be a positive integer."
    assert index < prod([len(values) for key, values in arguments]), (
        "index out of bounds."
    )


def get_job_args(arguments: list[tuple[str, list]], index: int) -> list[tuple[str, []]]:
    args = []
    for key, values in arguments:
        args.append((key, values[index % len(values)]))
        index = index // len(values)
    return args


def main(arguments: str, index: int):
    arguments = eval(arguments)
    check_arguments(arguments)
    check_index(arguments, index)
    args = get_job_args(arguments, index)
    args = [(k, v if isinstance(v, Real) else f'"{v}"') for k, v in args]
    out = " ".join([f"{k}={v}" for k, v in args])
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "arguments",
        type=str,
        help="String of a list of key-value list, with the key being the name "
        "of an argument and the value a list of potential values for this argument.",
    )
    parser.add_argument("index", type=int, help="Index in the job array.")
    args = parser.parse_args()

    out = main(args.arguments, args.index)
    print(out)
    sys.exit(0)

    # Example usage:
    # python scripts/product_args.py "[('--a', [1, 2, 3]), ('--b', [4, 5, 6])]" 1
    # Output: --a=2 --b=4
