# Copyright (c) 2026 Luong Hoang Vinh Tien
# All rights reserved.
import random
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
from collections import namedtuple
from functools import reduce
from pathlib import Path
import matplotlib.pyplot as plt
import json
import tracemalloc

# ==============================
# Configuration
# ==============================
GENOME_LENGTH   = 100
POPULATION_SIZE = 100
TOURNAMENT_K    = 3
CROSSOVER_RATE  = 0.9
MUTATION_RATE   = 1 / GENOME_LENGTH
ELITISM_SIZE    = 2
GENERATIONS     = 300
SEED            = 42

# ==============================
# Data types
# ==============================
Item      = namedtuple("Item", ["name", "value", "weight"])
Inventory = Tuple [Item, ...]


# ==============================================================
# SOLID Note:
#   S - Each class has one responsibility
#   O - Open for extension (new strategies), closed for modification
#   L - Subclasses are substitutable for their base class
#   I - Small, focused interfaces (each strategy has 1 method)
#   D - GeneticAlgorithm depends on abstractions, not concretions
# ==============================================================


# ==============================
# Chromosome
# ==============================
class Chromosome:
    """Represents a single candidate solution (bitstring genome)."""

    def __init__(self, genome: List[int]):
        self._genome: List[int] = genome
        self._fitness: Optional[int] = None

    @property
    def genome(self) -> List[int]:
        return self._genome

    @property
    def fitness(self) -> Optional[int]:
        return self._fitness

    def evaluate(self, fitness_function: "FitnessFunction") -> None:
        """Compute and cache fitness using the provided fitness function."""
        self._fitness = fitness_function.evaluate(self)

    def copy(self) -> "Chromosome":
        return Chromosome(self._genome[:])

    def __repr__(self) -> str:
        return f"Chromosome (fitness={self._fitness})"


# ==============================
# Abstract FitnessFunction  (Interface Segregation + Dependency Inversion)
# ==============================
class FitnessFunction(ABC):
    """Abstract base — all fitness functions must implement evaluate()."""

    @abstractmethod
    def evaluate(self, chromosome: Chromosome) -> int: ...


# ==============================
# Concrete fitness functions
# ==============================
class OneMaxFitness(FitnessFunction):
    """Fitness = number of 1s in the genome."""

    def evaluate(self, chromosome: Chromosome) -> int:
        return sum (chromosome.genome)


class KnapsackFitness(FitnessFunction):
    """
    Fitness = total value of selected items.
    Returns 0 if total weight exceeds capacity (hard constraint).
    """

    def __init__(self, inventory: Inventory, capacity: int):
        self._inventory = inventory
        self._capacity  = capacity

    def evaluate(self, chromosome: Chromosome) -> int:
        total_weight = 0
        total_value  = 0
        for bit, item in zip(chromosome.genome, self._inventory):
            if bit == 1:
                total_weight += item.weight
                total_value  += item.value
        return total_value if total_weight <= self._capacity else 0


# ==============================
# Adapter Pattern
# Wraps a plain callable (e.g. from FP version) into FitnessFunction interface
# ==============================
class FitnessFunctionAdapter(FitnessFunction):
    """
    Adapter: bridges a raw callable (Genome -> int) into the
    FitnessFunction interface so it can be used interchangeably
    with OOP fitness classes.
    """

    def __init__(self, function: Callable[[List[int]], int]):
        self._function = function

    def evaluate(self, chromosome: Chromosome) -> int:
        return self._function(chromosome.genome)


# ==============================
# Population
# ==============================
class Population:
    """Holds a collection of chromosomes for one generation."""

    def __init__(self, chromosomes: List[Chromosome]):
        self._chromosomes = chromosomes

    @classmethod
    def generate(
        cls,
        size: int,
        genome_length: int,
        fitness_function: FitnessFunction,
        rng: random.Random,
    ) -> "Population":
        chromosomes = []
        for _ in range(size):
            genome = [rng.randint(0, 1) for _ in range(genome_length)]
            c = Chromosome(genome)
            c.evaluate(fitness_function)
            chromosomes.append(c)
        return cls(chromosomes)

    @property
    def chromosomes(self) -> List[Chromosome]:
        return self._chromosomes

    def best(self) -> Chromosome:
        return max(self._chromosomes, key=lambda c: c.fitness)

    def elites(self, n: int) -> List[Chromosome]:
        return sorted(self._chromosomes, key=lambda c: c.fitness, reverse=True)[:n]

    def __len__(self) -> int:
        return len(self._chromosomes)


# ==============================
# Strategy Pattern — Selection
# ==============================
class SelectionStrategy(ABC):
    """Abstract selection strategy."""

    @abstractmethod
    def select(self, population: Population) -> Chromosome: ...


class TournamentSelection(SelectionStrategy):
    """Select the best individual from a random tournament of size k."""

    def __init__(self, k: int, rng: random.Random):
        self._k   = k
        self._rng = rng

    def select(self, population: Population) -> Chromosome:
        contestants = self._rng.sample(population.chromosomes, self._k)
        return max(contestants, key=lambda c: c.fitness)


# ==============================
# Strategy Pattern — Crossover
# ==============================
class CrossoverStrategy(ABC):
    """Abstract crossover strategy."""

    @abstractmethod
    def crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]: ...


class OnePointCrossover(CrossoverStrategy):
    """Single-point crossover with configurable probability."""

    def __init__(self, crossover_rate: float, rng: random.Random):
        self._crossover_rate = crossover_rate
        self._rng  = rng

    def crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        if self._rng.random() < self._crossover_rate:
            point = self._rng.randint(1, len(parent1.genome) - 1)
            genome1 = parent1.genome[:point] + parent2.genome[point:]
            genome2 = parent2.genome[:point] + parent1.genome[point:]
            return Chromosome(genome1), Chromosome(genome2)
        return parent1.copy(), parent2.copy()


# ==============================
# Strategy Pattern — Mutation
# ==============================
class MutationStrategy(ABC):
    """Abstract mutation strategy."""

    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome: ...


class BitFlipMutation(MutationStrategy):
    """Flip each bit independently with probability rate."""

    def __init__(self, mutation_rate: float, rng: random.Random):
        self._mutation_rate = mutation_rate
        self._rng  = rng

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        genome = [
            1 - bit if self._rng.random() < self._mutation_rate else bit
            for bit in chromosome.genome
        ]
        return Chromosome(genome)


# ==============================
# GAResult — simple data container
# ==============================
class GAResult:
    def __init__(
        self,
        best_genome: List[int],
        best_fitness: int,
        history: List[int],
        runtime: float,
    ):
        self.best_genome   = best_genome
        self.best_fitness  = best_fitness
        self.history       = history
        self.runtime       = runtime
        self.best_solution = "".join(str(b) for b in best_genome)


# ==============================
# Template Method Pattern — GeneticAlgorithm
# ==============================
class GeneticAlgorithm:
    """
    Coordinates the GA evolution loop.

    Uses Template Method: run() defines the fixed skeleton;
    each step delegates to injected Strategy objects.
    Depends on abstractions (FitnessFunction, SelectionStrategy, etc.)
    not on concrete implementations — satisfies Dependency Inversion.
    """

    def __init__(
        self,
        fitness_function:  FitnessFunction,
        selection:   SelectionStrategy,
        crossover:   CrossoverStrategy,
        mutation:    MutationStrategy,
        population_size: int  = POPULATION_SIZE,
        genome_length:   int  = GENOME_LENGTH,
        elitism_size:    int  = ELITISM_SIZE,
        generations:     int  = GENERATIONS,
        rng: random.Random    = None,
        target_fitness:  int  = None,
        verbose: bool = True
    ):
        self._fitness_function      = fitness_function
        self._selection       = selection
        self._crossover       = crossover
        self._mutation        = mutation
        self._population_size = population_size
        self._genome_length   = genome_length
        self._elitism_size    = elitism_size
        self._generations     = generations
        self._rng             = rng or random.Random(SEED)
        self._target_fitness  = target_fitness
        self._verbose = verbose

    # ----------------------------------------------------------
    # Template Method — fixed skeleton, each step is a hook
    # ----------------------------------------------------------
    def run(self) -> GAResult:
        start = time.time()

        population = self._init_population()
        history: List[int] = []

        for generation in range(self._generations):
            population = self._next_generation(population)
            best = population.best()
            history.append(best.fitness)
            if self._verbose:
                print(f"Generation {generation:>3}: Best fitness = {best.fitness:>15,}")

            if self._target_fitness is not None and best.fitness >= self._target_fitness:
                if self._verbose:
                    print(f"Converged at generation {generation}!")
                break

        runtime = time.time() - start
        best = population.best()
        return GAResult(best.genome, best.fitness, history, runtime)

    # ----------------------------------------------------------
    # Hooks (can be overridden in subclasses)
    # ----------------------------------------------------------
    def _init_population(self) -> Population:
        return Population.generate(
            self._population_size,
            self._genome_length,
            self._fitness_function,
            self._rng,
        )

    def _next_generation(self, population: Population) -> Population:
        elites    = [c.copy() for c in population.elites(self._elitism_size)]
        offspring = self._reproduce(population)
        new_chromosomes = elites + offspring

        for c in new_chromosomes:
            c.evaluate(self._fitness_function)

        return Population(new_chromosomes)

    def _reproduce(self, population: Population) -> List[Chromosome]:
        offspring = []
        remaining = self._population_size - self._elitism_size
        for _ in range(remaining // 2):
            parent1 = self._selection.select(population)
            parent2 = self._selection.select(population)
            offspring1, offspring2 = self._crossover.crossover(parent1, parent2)
            offspring.append(self._mutation.mutate(offspring1))
            offspring.append(self._mutation.mutate(offspring2))
        return offspring


# ==============================
# Helpers
# ==============================
def generate_inventory(length: int, rng: random.Random) -> Tuple[Inventory, int]:
    inventory = tuple(
        Item("Item " + str(i + 1), rng.randint(10, 100), rng.randint(5, 50))
        for i in range(length)
    )
    capacity = int(0.4 * sum(item.weight for item in inventory))
    return inventory, capacity


# ==============================
# Main
# ==============================
def main():
    rng = random.Random(SEED)

    # ----------------------------------------------------------
    # OneMax
    # ----------------------------------------------------------
    print(f"\n{'=' * 45}")
    print(f"{'ONEMAX':^45}")
    print(f"{'=' * 45}")

    ga_onemax = GeneticAlgorithm(
        fitness_function = OneMaxFitness(),
        selection        = TournamentSelection(k=TOURNAMENT_K, rng=rng),
        crossover        = OnePointCrossover(crossover_rate=CROSSOVER_RATE, rng=rng),
        mutation         = BitFlipMutation(mutation_rate=MUTATION_RATE, rng=rng),
        target_fitness   = GENOME_LENGTH,
        rng              = rng,
    )

    tracemalloc.start()
    result_onemax = ga_onemax.run()
    _, peak_onemax = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nBest solution : {result_onemax.best_solution}")
    print(f"Best fitness  : {result_onemax.best_fitness}")
    print(f"Runtime       : {result_onemax.runtime:.4f}s")
    print(f"Peak memory   : {peak_onemax / 1024:.1f} KB")

    plt.clf()
    plt.plot(result_onemax.history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Genetic Algorithm - OneMax (OOP)\nRuntime: {result_onemax.runtime:.4f}s | Peak memory: {peak_onemax / 1024:.1f} KB")
    report_path = Path(__file__).resolve().parents[2] / "reports" / "onemax_curve.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path)
    plt.show()

    # ----------------------------------------------------------
    # Knapsack
    # ----------------------------------------------------------
    print(f"\n{'=' * 45}")
    print(f"{'KNAPSACK':^45}")
    print(f"{'=' * 45}")

    inventory, capacity = generate_inventory(GENOME_LENGTH, rng)

    ga_knapsack = GeneticAlgorithm(
        fitness_function = KnapsackFitness(inventory, capacity),
        selection        = TournamentSelection(k=TOURNAMENT_K, rng=rng),
        crossover        = OnePointCrossover(crossover_rate=CROSSOVER_RATE, rng=rng),
        mutation         = BitFlipMutation(mutation_rate=MUTATION_RATE, rng=rng),
        rng              = rng,
    )

    tracemalloc.start()
    result_knapsack = ga_knapsack.run()
    _, peak_knapsack = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    selected_items = [
        item for bit, item in zip(result_knapsack.best_genome, inventory) if bit == 1
    ]
    total_value  = sum(item.value  for item in selected_items)
    total_weight = sum(item.weight for item in selected_items)

    print(f"\n{'The best combination':=^60}")
    print(f"{'Name':<12} {'Value':>15} {'Weight':>15}")
    print("-" * 60)
    for item in selected_items:
        print(f"{item.name:<12} {item.value:>15,} {item.weight:>15,}")
    print("=" * 60)
    print(f"{'Total value: ':<12} {total_value:>15,}")
    print(f"{'Total weight:':<12} {total_weight:>15,}")
    print(f"{'Capacity:    ':<12} {capacity:>15,}")
    print(f"{'Items chosen:':<12} {len(selected_items):>15}")
    print(f"\nBest solution : {result_knapsack.best_solution}")
    print(f"Best fitness  : {result_knapsack.best_fitness:,}")
    print(f"Runtime       : {result_knapsack.runtime:.4f}s")
    print(f"Peak memory   : {peak_knapsack / 1024:.1f} KB")

    plt.clf()
    plt.plot(result_knapsack.history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Genetic Algorithm - Knapsack (OOP)\nRuntime: {result_knapsack.runtime:.4f}s | Peak memory: {peak_knapsack / 1024:.1f} KB")
    report_path = Path(__file__).resolve().parents[2] / "reports" / "knapsack_curve.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path)
    plt.show()

    # ----------------------------------------------------------
    # Export JSON
    # ----------------------------------------------------------
    report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "results_oop.json", "w") as f:
        json.dump({
            "OneMax": {
                "best_fitness": result_onemax.best_fitness,
                "best_solution": result_onemax.best_solution,
                "runtime": result_onemax.runtime,
                "peak_memory_kb": peak_onemax / 1024,
                "history": result_onemax.history,
            },
            "Knapsack": {
                "best_fitness": result_knapsack.best_fitness,
                "best_solution": result_knapsack.best_solution,
                "runtime": result_knapsack.runtime,
                "history": result_knapsack.history,
                "peak_memory_kb": peak_knapsack / 1024,
                "selected_items": [
                    {"name": i.name, "value": i.value, "weight": i.weight}
                    for i in selected_items
                ],
            },
        }, f, indent=4)


if __name__ == "__main__":
    main()