"""Run throwaway calendar-axis cure experiments; production code is unchanged.

    uv run --extra forecast python examples/prototype_numpyro_native.py
    uv run --extra forecast python examples/prototype_numpyro_native.py --idea convolution

All alternatives retain unit identities. The experiments check likelihood and
parameter-gradient parity with TTENet, native forecast driver execution, hard
closures, delayed entry, and event conservation. The convolution experiment
also compares exact chained event-time draws against an independent Poisson
approximation using the SAME means.

Short SVI runs prove integration, not convergence or calibration. Dense fixture
arrays and predeclared future births are not an implementation of an unbounded
stochastic sales stream. No production cutover is made by these experiments.
"""

import argparse
import importlib
import json
import time
from importlib.metadata import version

IDEAS = {
    "adapter": "prototype_native_adapter",
    "distribution": "prototype_native_distribution",
    "markov": "prototype_native_markov",
    "convolution": "prototype_native_convolution",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", choices=["all", *IDEAS], default="all")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            {"versions": {name: version(name) for name in ("numpyro-forecast", "numpyro", "jax")}},
            indent=2,
        ),
        flush=True,
    )
    for name, module_name in IDEAS.items():
        if args.idea not in ("all", name):
            continue
        print(f"\n--- {name} ---", flush=True)
        module = importlib.import_module(module_name)
        start = time.perf_counter()
        result = module.run(steps=args.steps, draws=args.draws)
        result["elapsed_seconds_including_compilation"] = round(time.perf_counter() - start, 3)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
