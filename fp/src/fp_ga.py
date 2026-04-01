# Copyright (c) 2026 Luong Hoang Vinh Tien
# All rights reserved.
import random
from typing import Tuple, Callable
from functools import reduce
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import namedtuple
import json
import tracemalloc

# ==============================
# Configuration
# ==============================

GENOME_LENGTH = 100 # Length of binary string
POPULATION_SIZE = 100 # Starts with 100 solutions
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1 / GENOME_LENGTH 
ELITISM_SIZE = 2
GENERATIONS = 300
SEED = 42

# ==============================
# Variable types
# ==============================
Chromosome = Tuple [int, ...]
Population = Tuple [Chromosome, ...]
Item = namedtuple("Item", ["name", "value", "weight"])
Inventory = Tuple[Item, ...]
FitnessFunction = Callable[[Chromosome], int]

# Generate chromosome
def generate_chromosome(length: int, rng: random.Random) -> Chromosome:
    return tuple(rng.randint(0, 1) for _ in range (length))

# Generate items
def generate_inventory(length: int, rng: random.Random) -> Tuple [Inventory, int]:
    inventory = tuple(Item("Item " + str(i + 1), rng.randint(10, 100), rng.randint (5, 50)) for i in range (0, length))
    capacity = int(0.4 * reduce(lambda acc, cur: acc + cur.weight, inventory, 0))
    return inventory, capacity

# Generate population
def generate_population(population_size: int, chromosome_length: int, rng: random.Random) -> Population:
    return tuple(generate_chromosome(chromosome_length, rng) for _ in range (population_size))

# Fitness function 1
def fitness_function_onemax(chromosome: Chromosome) -> int:
    # Sum of all elements = Number of 1s
    return reduce(lambda acc, cur: acc + cur, chromosome, 0)

# Fitness function 2
def fitness_function_knapsack(inventory: Inventory, capacity: int) -> FitnessFunction:
    def _fitness(chromosome: Chromosome) -> int:
        if len(chromosome) != len(inventory):
            raise ValueError("Chromosome length and number of items must be the same")
        total_weight, total_value = reduce(
            lambda acc, pair: (
                acc [0] + pair [1].weight * pair[0],
                acc [1] + pair [1].value  * pair[0],
            ),
            zip(chromosome, inventory),
            (0, 0)
        )
        return total_value if total_weight <= capacity else 0
    return _fitness

# Select parent
def select_parent(population: Population, fitness_function: FitnessFunction, rng: random.Random) -> Chromosome:
    selected = tuple(rng.sample(population, TOURNAMENT_K))
    scored = tuple(map(lambda chromosome: (chromosome, fitness_function(chromosome)), selected))
    best = max(scored, key = lambda pair: pair [1])
    return best [0]

# Crossover function
def crossover_function(chromosome1: Chromosome, chromosome2: Chromosome, rng: random.Random) -> Tuple [Chromosome, Chromosome]:
    if rng.random() < CROSSOVER_RATE:
        crossover_point = rng.randint(1, len(chromosome1) - 1)
        return chromosome1 [:crossover_point] + chromosome2 [crossover_point:],  chromosome2 [:crossover_point] + chromosome1 [crossover_point:]
    else:
        return chromosome1, chromosome2
    
def mutation(chromosome: Chromosome, rng: random.Random) -> Chromosome:
    return tuple(map(lambda x: 1 - x if (rng.random() < MUTATION_RATE) else x, chromosome))

def genetic_algorithm(fitness_function: FitnessFunction, rng: random.Random, target_fitness: int = None, verbose: bool = True):
    population = generate_population(POPULATION_SIZE, GENOME_LENGTH, rng)
    history = ()

    for generation in range(GENERATIONS):
        fitness_values = tuple(map(fitness_function, population))
        scored = tuple(zip(population, fitness_values))

        elites = tuple(
            map(lambda x: x [0],
                sorted(scored, key=lambda x: x[1], reverse=True)[:ELITISM_SIZE])
        )

        new_population = elites
        remaining = POPULATION_SIZE - ELITISM_SIZE

        for _ in range(remaining // 2):
            parent1 = select_parent(population, fitness_function, rng)
            parent2 = select_parent(population, fitness_function, rng)

            offspring1, offspring2 = crossover_function(parent1, parent2, rng)

            new_population = new_population + (
                mutation(offspring1, rng),
                mutation(offspring2, rng)
            )

        population = new_population

        fitness_values = tuple(map(fitness_function, population))
        best_fitness = max(fitness_values)

        history = history + (best_fitness,)

        if verbose:
            print(f"Generation {generation:>3}: Best fitness = {best_fitness:>15,}")
        if target_fitness is not None and best_fitness >= target_fitness:
            if verbose: 
                print(f"Converged at generation {generation}!")
            break

    best_fitness = max(fitness_values)
    best_index = fitness_values.index(best_fitness)
    best_chromosome = population[best_index]                                         
    best_solution_bit_string = reduce(lambda acc, cur: acc + str(cur), best_chromosome, "")
    return best_solution_bit_string, best_fitness, history, best_chromosome

def main():
    rng = random.Random(SEED)

    # ----------------------------------------------------------
    # OneMax
    # ----------------------------------------------------------
    print(f"\n{'=' * 45}")
    print(f"{'ONEMAX':^45}")
    print(f"{'=' * 45}")

    tracemalloc.start()
    start = time.time()

    best_solution_bit_string_onemax, best_fitness_onemax, history_onemax, _ = genetic_algorithm(
        fitness_function_onemax, rng, GENOME_LENGTH
    )

    end = time.time()
    _, peak_onemax = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_onemax = end - start

    print(f"Best solution : {best_solution_bit_string_onemax}")
    print(f"Best fitness  : {best_fitness_onemax}")
    print(f"Runtime       : {runtime_onemax:.4f}s")
    print(f"Peak memory   : {peak_onemax / 1024:.1f} KB")

    plt.clf()
    plt.plot(history_onemax)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(
        f"Genetic Algorithm - OneMax (FP)\n"
        f"Runtime: {runtime_onemax:.4f}s | Peak memory: {peak_onemax / 1024:.1f} KB"
    )
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

    tracemalloc.start()
    start = time.time()

    best_solution_bit_string_knapsack, best_fitness_knapsack, history_knapsack, best_chromosome_knapsack = genetic_algorithm(
        fitness_function_knapsack(inventory, capacity), rng
    )

    end = time.time()
    _, peak_knapsack = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_knapsack = end - start

    selected_items = tuple(
        item for bit, item in zip(best_chromosome_knapsack, inventory) if bit == 1
    )

    total_value = reduce(lambda acc, item: acc + item.value, selected_items, 0)
    total_weight = reduce(lambda acc, item: acc + item.weight, selected_items, 0)

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

    print(f"\nBest solution : {best_solution_bit_string_knapsack}")
    print(f"Best fitness  : {best_fitness_knapsack:,}")
    print(f"Runtime       : {runtime_knapsack:.4f}s")
    print(f"Peak memory   : {peak_knapsack / 1024:.1f} KB")

    plt.clf()
    plt.plot(history_knapsack)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(
        f"Genetic Algorithm - Knapsack (FP)\n"
        f"Runtime: {runtime_knapsack:.4f}s | Peak memory: {peak_knapsack / 1024:.1f} KB"
    )
    report_path = Path(__file__).resolve().parents[2] / "reports" / "knapsack_curve.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path)
    plt.show()

    # ----------------------------------------------------------
    # Export JSON
    # ----------------------------------------------------------
    report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "results_fp.json", "w") as f:
        json.dump({
            "OneMax": {
                "best_fitness": best_fitness_onemax,
                "best_solution": best_solution_bit_string_onemax,
                "runtime": runtime_onemax,
                "peak_memory_kb": peak_onemax / 1024,
                "history": list(history_onemax),
            },
            "Knapsack": {
                "best_fitness": best_fitness_knapsack,
                "best_solution": best_solution_bit_string_knapsack,
                "runtime": runtime_knapsack,
                "peak_memory_kb": peak_knapsack / 1024,
                "history": list(history_knapsack),
                "selected_items": [
                    {"name": i.name, "value": i.value, "weight": i.weight}
                    for i in selected_items
                ]
            }
        }, f, indent=4)


if __name__ == "__main__":
    main()