# Genetic Algorithm - OOP vs Functional Programming

## Results

![OneMax](reports/onemax_curve.png)
![Knapsack](reports/knapsack_curve.png)

**Name:** Luong Hoang Vinh Tien  
**Student ID:** 2413477

---

## Repository Purpose

This repository is published publicly to receive feedback from the
programming community and to improve the implementation.

The code is shared for **learning, discussion, and code review purposes**.
Please do not copy or submit this code as part of academic coursework
or assignments.

---

## Genetic Algorithm Configuration

The assignment specifies the following GA parameters:

| Parameter | Value |
|----------|------|
| Representation | Bitstring |
| Population size | 100 |
| Selection | Tournament (k = 3) |
| Crossover | One-point crossover (p = 0.9) |
| Mutation | Bit-flip mutation (p = 1/L per bit) |
| Replacement | Generational with elitism (e = 2) |
| Termination | 300 generations |
| Random seed | 42 |

Both the OOP and FP implementations use the same configuration to ensure fair comparison. These parameters are defined in the course assignment and are kept identical for both implementations.

## 1. Instructions to Run

Make sure you have Python 3.10+ and the required dependencies installed:

```bash
pip install matplotlib
```

Open the `ga-assignment/` folder, then run from the **root directory**:

### a. OOP Version

```bash
# Run the full GA
python oop/run.py

# Run all tests
python oop/tests/tests.py

# Run a specific test
python oop/tests/tests.py <test_name>
```

### b. FP Version

```bash
# Run the full GA
python fp/run.py

# Run all tests
python fp/tests/tests.py

# Run a specific test
python fp/tests/tests.py <test_name>
```

Both versions will generate output files under `reports/`:
- `results_oop.json` — results from OOP version
- `results_fp.json` — results from FP version
- `onemax_curve.png` — fitness evolution plot for OneMax (FP vs OOP)
- `knapsack_curve.png` — fitness evolution plot for Knapsack (FP vs OOP)

To generate the combined comparison plots, run both versions first, then run
```bash 
python reports/generate_reports.py
```
---

## 2. Design Explanation

### a. OOP Version

The OOP implementation models each GA component as a class with clearly defined responsibilities:

- `Chromosome` - wraps a bitstring genome and exposes its fitness value
- `Population` - holds a collection of chromosomes and handles generation turnover
- `SelectionStrategy` - abstract base class; `TournamentSelection` implements tournament selection with configurable k
- `CrossoverStrategy` - abstract base class; `OnePointCrossover` implements single-point crossover
- `MutationStrategy` - abstract base class; `BitFlipMutation` implements per-bit flip mutation
- `GeneticAlgorithm` - orchestrates the full evolution loop using the above strategies via dependency injection

State is fully encapsulated inside each class. Swapping a strategy (e.g. replacing tournament selection with roulette wheel) requires no changes to `GeneticAlgorithm` - only a different strategy object is passed in. Both `OneMax` and `Knapsack` fitness functions are implemented as subclasses of a `FitnessFunction` abstract class.

### b. FP Version

The FP implementation follows pure functional programming principles throughout:

- **No classes** - all logic is expressed as standalone functions
- **Immutable data** - chromosomes and populations are represented as Python `tuple`s; no in-place mutation anywhere
- **Pure functions** - every function takes all required state as parameters and returns a new value without side effects. The global `rng` was moved into `main()` and passed explicitly down the call chain
- **Higher-order functions** - `map`, `filter`, and `reduce` are used in place of imperative loops for chromosome generation, fitness evaluation, elitism selection, and mutation
- **Factory pattern for fitness** - `fitness_function_knapsack(inventory, capacity)` returns a closure `Chromosome -> int`, keeping the same interface as `fitness_function_onemax` so `genetic_algorithm` does not need to know which problem it is solving

The `genetic_algorithm` function accepts any `FitnessFunction` and an optional `target_fitness` for early termination, making it fully reusable across problems.

---

## Academic Integrity Notice

This repository is provided for reference and discussion only.

Copying or submitting this code, in whole or in part, for coursework,
assignments, or academic evaluation is prohibited.

Please implement your own solution instead.

---

## 3. Reflection

Both implementations solve the same problems using the same parameters and produce comparable results, but the experience of writing them felt quite different.

The OOP approach made the code easy to navigate. Each class has a single job, and the use of abstract base classes enforced a consistent interface across strategies. Adding a new selection method or a new problem is straightforward - just write a new subclass. The downside is boilerplate: even simple logic requires a class definition, `__init__`, and method signatures, which can feel heavyweight for a problem of this scale.

The FP approach was more concise. Representing chromosomes as tuples and relying on `map`/`reduce` kept individual functions short and easy to reason about in isolation. Because nothing is mutated in place, bugs related to shared state simply cannot happen. The tradeoff is that threading `rng` through every function signature adds noise, and deeply nested `reduce` lambdas can become hard to read quickly.

In terms of correctness, both paradigms performed equally well - the GA converged reliably on OneMax and produced valid, high-value solutions for Knapsack. For a project like this, where the algorithm is fixed and the components are well-defined, OOP offers better long-term maintainability while FP offers better testability and fewer hidden dependencies. A hybrid approach - using dataclasses for structured data and pure functions for logic - would likely combine the best of both worlds.

---

## Project Structure

ga-assignment/
│
├── README.md
├── LICENSE
│
├── fp/
│   ├── src/
│   │   ├── fp_ga.py
│   │   
│   │   
│   │   
│   │
│   ├── tests/
│   │   ├── tests.py
│   │   
│   │   
│   │
│   └── run.py
│
├── oop/
│   ├── src/
│   │   ├── oop_ga.py
│   │   
│   │   
│   │   
│   │
│   ├── tests/
│   │   ├── tests.py
│   │   
│   │   
│   │
│   └── run.py
│
└── reports/
    ├── onemax_curve.png
    ├── knapsack_curve.png
    ├── results_oop.json
    └── results_fp.json

## Acknowledgments

Some implementation ideas were inspired by publicly available
educational resources and tutorials about Genetic Algorithms.

AI-assisted tools were used during development for debugging,
refactoring suggestions, and code improvement.

The final implementation, structure, and modifications were written
and adapted by the author.

© 2026 Luong Hoang Vinh Tien