import random
from typing import Tuple, Callable
from functools import *
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import namedtuple
import json

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
Genome = Tuple [int, ...]
Population = Tuple [Genome, ...]
Item = namedtuple("Item", ["name", "value", "weight"])
Inventory = Tuple[Item, ...]
FitnessFunction = Callable[[Genome], int]

# Generate genome
def generate_genome(length: int, rng: random.Random) -> Genome:
    return tuple(rng.randint(0, 1) for _ in range (length))

# Generate items
def generate_inventory(length: int, rng: random.Random) -> Tuple [Inventory, int]:
    inventory = tuple(Item("Item " + str(i + 1), rng.randint(1, 10 ** 9), rng.randint (1, 10 ** 9)) for i in range (0, length))
    capacity = int(0.4 * reduce(lambda acc, cur: acc + cur.weight, inventory, 0))
    return inventory, capacity

# Generate population
def generate_population(population_size: int, genome_length: int, rng: random.Random) -> Population:
    return tuple(generate_genome(genome_length, rng) for _ in range (population_size))

# Fitness function 1
def fitness_function_OneMax(genome: Genome) -> int:
    # Sum of all elements = Number of 1s
    return reduce(lambda acc, cur: acc + cur, genome, 0)

# Fitness function 2
def fitness_function_Knapsack(inventory: Inventory, capacity: int) -> FitnessFunction:
    def _fitness(genome: Genome) -> int:
        if len(genome) != len(inventory):
            raise ValueError("Genome length and number of items must be the same")
        total_weight, total_value = reduce(
            lambda acc, pair: (
                acc[0] + pair[1].weight * pair[0],
                acc[1] + pair[1].value  * pair[0],
            ),
            zip(genome, inventory),
            (0, 0)
        )
        return total_value if total_weight <= capacity else 0
    return _fitness

# Select parent
def select_parent(population: Population, fitness_function: FitnessFunction, rng: random.Random) -> Genome:
    selected = tuple(rng.sample(population, TOURNAMENT_K))
    scored = tuple(map(lambda genome: (genome, fitness_function(genome)), selected))
    best = max(scored, key = lambda pair: pair [1])
    return best [0]

# Crossover function
def crossover_function(genome1: Genome, genome2: Genome, rng: random.Random) -> Tuple [Genome, Genome]:
    if rng.random() < CROSSOVER_RATE:
        crossover_point = rng.randint(1, len(genome1) - 1)
        return genome1 [:crossover_point] + genome2 [crossover_point:],  genome2 [:crossover_point] + genome1 [crossover_point:]
    else:
        return genome1, genome2
    
def mutation(genome: Genome, rng: random.Random) -> Genome:
    return tuple(map(lambda x: 1 - x if (rng.random() < MUTATION_RATE) else x, genome))

def genetic_algorithm(fitness_function: FitnessFunction, rng: random.Random, target_fitness: int = None):
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

        print(f"Generation {generation:>3}: Best fitness: {best_fitness:>15,}")
        if target_fitness is not None and best_fitness >= target_fitness:
            print(f"Converged at generation {generation}!")
            break

    best_fitness = max(fitness_values)
    best_index = fitness_values.index(best_fitness)
    best_genome = population[best_index]                                         
    best_solution_bit_string = reduce(lambda acc, cur: acc + str(cur), best_genome, "")
    return best_solution_bit_string, best_fitness, history, best_genome

def main():
    rng = random.Random(SEED)
    # OneMax problem
    start = time.time()
    best_solution_bit_string_OneMax, best_fitness_OneMax, history_OneMax, _ = genetic_algorithm(fitness_function_OneMax, rng, GENOME_LENGTH)
    end = time.time()
    runtime_OneMax = end - start
    print(f"Best solution: {best_solution_bit_string_OneMax}")
    print(f"Final best fitness: {best_fitness_OneMax}")
    print(f"runtime: {runtime_OneMax:.20f}s")
    plt.plot(history_OneMax)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - OneMax")
    report_path = Path(__file__).resolve().parents[2] / "reports" / "onemax_curve.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path)
    plt.show()

    # Knapsack problem
    inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
    start = time.time()
    best_solution_bit_string_Knapsack, best_fitness_Knapsack, history_Knapsack, best_genome_Knapsack = genetic_algorithm(fitness_function_Knapsack(inventory, capacity), rng)
    end = time.time()
    runtime_Knapsack = end - start
    # Print out selected items
    selected_items = tuple(item for bit, item in zip(best_genome_Knapsack, inventory) if bit == 1)
    total_value = reduce(lambda acc, item: acc + item.value, selected_items, 0)
    total_weight = reduce(lambda acc, item: acc + item.weight, selected_items, 0)
    # Header
    print(f"\n{'The best combination':=^60}")
    print(f"{'Name':<12} {'Value':>15} {'Weight':>15}")
    print("-" * 60)

    # Rows
    for item in selected_items:
        print(f"{item.name:<12} {item.value:>15,} {item.weight:>15,}")
    print(f"Best solution: {best_solution_bit_string_Knapsack}")
    print(f"Final best fitness: {best_fitness_Knapsack:>15,}")
    print(f"runtime: {runtime_Knapsack:.20f}s")
    print(f"{'Total value: ':<12} {total_value:>15,}")
    print(f"{'Total weight:':<12} {total_weight:>15,}")
    print(f"{'Capacity:    ':<12} {capacity:>15,}")
    print(f"{'Items chosen:':<12} {len(selected_items):>15}")
    plt.clf()
    plt.plot(history_Knapsack)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Knapsack")
    report_path = Path(__file__).resolve().parents[2] / "reports" / "knapsack_curve.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path)
    plt.show()
    report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "results_fp.json", "w") as f:
        json.dump({
            "OneMax": {
                "best_fitness": best_fitness_OneMax,
                "best_solution": best_solution_bit_string_OneMax,
                "runtime": runtime_OneMax,
                "history": list(history_OneMax)
            },
            "Knapsack": {
                "best_fitness": best_fitness_Knapsack,
                "best_solution": best_solution_bit_string_Knapsack,
                "runtime": runtime_Knapsack,
                "history": list(history_Knapsack),
                "selected_items": [
                    {"name": i.name, "value": i.value, "weight": i.weight}
                    for i in selected_items
                ]
            }
        }, f, indent=4)

if __name__ == "__main__":
    main()