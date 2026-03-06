import random
import sys
from functools import reduce
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fp_ga import (
    generate_genome,
    generate_inventory,
    generate_population,
    fitness_function_OneMax,
    fitness_function_Knapsack,
    select_parent,
    crossover_function,
    mutation,
    genetic_algorithm,
    GENOME_LENGTH,
    SEED,
)

# ==============================
# Shared fixtures
# ==============================
Item = namedtuple("Item", ["name", "value", "weight"])

# 100 reproducible seeds derived from SEED=42
_base_rng = random.Random(SEED)
SEEDS_100 = [_base_rng.randint(0, 10 ** 6) for _ in range(100)]
SEEDS_10  = SEEDS_100[:10]

# Small hand-crafted instance for deterministic unit tests
SMALL_ITEMS = (
    Item("A", value=10, weight=5),
    Item("B", value=6,  weight=4),
    Item("C", value=3,  weight=3),
)
SMALL_CAPACITY = 8
small_fit = fitness_function_Knapsack(SMALL_ITEMS, SMALL_CAPACITY)


# ==============================
# 1. Fitness - OneMax (4 tests)
# ==============================
def test_onemax_all_ones():
    assert fitness_function_OneMax((1,) * 5) == 5

def test_onemax_all_zeros():
    assert fitness_function_OneMax((0,) * 5) == 0

def test_onemax_mixed():
    assert fitness_function_OneMax((1, 0, 1, 0, 1)) == 3

def test_onemax_single_one():
    assert fitness_function_OneMax((0, 0, 1, 0, 0)) == 1


# ==============================
# 2. Fitness - Knapsack unit (8 tests)
# ==============================
def test_knapsack_single_item_within_capacity():
    # Only A, weight=5 <= 8
    assert small_fit((1, 0, 0)) == 10

def test_knapsack_two_items_within_capacity():
    # B + C, weight=7 <= 8
    assert small_fit((0, 1, 1)) == 9

def test_knapsack_exceeds_capacity():
    # A + B, weight=9 > 8, penalty applied
    assert small_fit((1, 1, 0)) == 0

def test_knapsack_all_items_exceeds_capacity():
    # All items, weight=12 > 8, penalty applied
    assert small_fit((1, 1, 1)) == 0

def test_knapsack_empty_selection():
    # No items selected, value=0
    assert small_fit((0, 0, 0)) == 0

def test_knapsack_optimal_selection():
    # A + C = value 13, weight 8 = capacity (valid)
    assert small_fit((1, 0, 1)) == 13

def test_knapsack_exactly_at_capacity():
    # weight == capacity is still valid
    assert small_fit((1, 0, 1)) == 13

def test_knapsack_genome_length_mismatch():
    try:
        small_fit((1, 0))
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# ==============================
# 3. Fitness - Knapsack over 100 instances (4 tests)
# ==============================
def test_knapsack_empty_genome_always_zero():
    # Selecting no items must always yield fitness=0 regardless of instance
    for seed in SEEDS_100:
        rng = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
        fit = fitness_function_Knapsack(inventory, capacity)
        assert fit((0,) * GENOME_LENGTH) == 0

def test_knapsack_fitness_non_negative():
    # Fitness must always be >= 0
    for seed in SEEDS_100:
        rng = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
        fit = fitness_function_Knapsack(inventory, capacity)
        assert fit((1,) * GENOME_LENGTH) >= 0

def test_knapsack_single_item_always_valid():
    # Selecting only the lightest item must always be within capacity
    for seed in SEEDS_100:
        rng = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
        fit = fitness_function_Knapsack(inventory, capacity)
        lightest_idx = min(range(GENOME_LENGTH), key=lambda i: inventory[i].weight)
        genome = tuple(1 if i == lightest_idx else 0 for i in range(GENOME_LENGTH))
        assert fit(genome) == inventory[lightest_idx].value

def test_knapsack_capacity_scales_with_total_weight():
    # Capacity must always equal 40% of total weight
    for seed in SEEDS_100:
        rng = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
        total_weight = reduce(lambda acc, item: acc + item.weight, inventory, 0)
        assert capacity == int(0.4 * total_weight)


# ==============================
# 4. Selection (3 tests)
# ==============================
def test_select_parent_returns_valid_genome():
    rng_sel = random.Random(1)
    population = generate_population(10, 5, rng_sel)
    parent = select_parent(population, fitness_function_OneMax, rng_sel)
    assert parent in population

def test_select_parent_prefers_higher_fitness():
    # 50 best vs 50 worst — best should win more than 50% of tournaments
    rng_sel = random.Random(0)
    best  = (1,) * 10
    worst = (0,) * 10
    population = (best,) * 50 + (worst,) * 50
    wins = sum(
        1 for _ in range(200)
        if select_parent(population, fitness_function_OneMax, rng_sel) == best
    )
    assert wins > 100

def test_select_parent_never_returns_outside_population():
    # Selected parent must always be a member of the population
    for seed in SEEDS_100:
        rng_sel = random.Random(seed)
        population = generate_population(20, GENOME_LENGTH, rng_sel)
        parent = select_parent(population, fitness_function_OneMax, rng_sel)
        assert parent in population


# ==============================
# 5. Crossover (4 tests)
# ==============================
def test_crossover_offspring_length():
    rng_cx = random.Random(0)
    g1 = (1,) * 10
    g2 = (0,) * 10
    o1, o2 = crossover_function(g1, g2, rng_cx)
    assert len(o1) == 10 and len(o2) == 10

def test_crossover_bits_are_valid():
    rng_cx = random.Random(0)
    g1 = (1,) * 10
    g2 = (0,) * 10
    o1, o2 = crossover_function(g1, g2, rng_cx)
    assert all(b in (0, 1) for b in o1 + o2)

def test_crossover_skipped_returns_originals():
    # Force no crossover — random() always returns 1.0 > CROSSOVER_RATE
    class FakeRng:
        def random(self): return 1.0
    g1 = (1, 0, 1, 0, 1)
    g2 = (0, 1, 0, 1, 0)
    o1, o2 = crossover_function(g1, g2, FakeRng())
    assert o1 == g1 and o2 == g2

def test_crossover_produces_recombination():
    # Over 20 trials at least one crossover must produce a mixed offspring
    rng_cx = random.Random(5)
    g1 = (1,) * 10
    g2 = (0,) * 10
    results = [crossover_function(g1, g2, rng_cx) for _ in range(20)]
    mixed = [(o1, o2) for o1, o2 in results if o1 != g1 and o1 != g2]
    assert len(mixed) > 0


# ==============================
# 6. Mutation (4 tests)
# ==============================
def test_mutation_preserves_length():
    rng_mut = random.Random(0)
    genome = (1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
    assert len(mutation(genome, rng_mut)) == len(genome)

def test_mutation_bits_are_valid():
    rng_mut = random.Random(0)
    genome = (1,) * 20
    assert all(b in (0, 1) for b in mutation(genome, rng_mut))

def test_mutation_returns_tuple():
    rng_mut = random.Random(0)
    assert isinstance(mutation((1,) * 10, rng_mut), tuple)

def test_mutation_flips_all_bits_at_max_rate():
    # At rate 1.0 every bit must flip
    genome = (1, 0, 1, 0, 1)
    mutated = tuple(map(lambda x: 1 - x, genome))
    assert mutated == (0, 1, 0, 1, 0)


# ==============================
# 7. GA improvement - OneMax (3 tests)
# ==============================
def test_onemax_fitness_improves():
    rng_ga = random.Random(SEED)
    _, _, history, _ = genetic_algorithm(
        fitness_function_OneMax, rng_ga, target_fitness=GENOME_LENGTH
    )
    assert history[-1] > history[0]

def test_onemax_converges_to_optimal():
    rng_ga = random.Random(SEED)
    _, best_fitness, _, _ = genetic_algorithm(
        fitness_function_OneMax, rng_ga, target_fitness=GENOME_LENGTH
    )
    assert best_fitness == GENOME_LENGTH

def test_onemax_history_is_non_decreasing():
    rng_ga = random.Random(SEED)
    _, _, history, _ = genetic_algorithm(
        fitness_function_OneMax, rng_ga, target_fitness=GENOME_LENGTH
    )
    # Best fitness must never drop between generations due to elitism
    assert all(history[i] <= history[i + 1] for i in range(len(history) - 1))


# ==============================
# 8. GA improvement - Knapsack over 10 instances (2 tests)
# ==============================
def test_knapsack_fitness_improves_10_instances():
    for seed in SEEDS_10:
        rng_inv = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng_inv)
        knapsack_fitness = fitness_function_Knapsack(inventory, capacity)
        rng_ga = random.Random(seed)
        _, _, history, _ = genetic_algorithm(knapsack_fitness, rng_ga)
        assert history[-1] >= history[0], f"Fitness did not improve for seed={seed}"

def test_knapsack_solution_respects_capacity_10_instances():
    for seed in SEEDS_10:
        rng_inv = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng_inv)
        knapsack_fitness = fitness_function_Knapsack(inventory, capacity)
        rng_ga = random.Random(seed)
        _, _, _, best_genome = genetic_algorithm(knapsack_fitness, rng_ga)
        total_weight = reduce(
            lambda acc, pair: acc + pair[1].weight * pair[0],
            zip(best_genome, inventory),
            0
        )
        assert total_weight <= capacity, f"Capacity exceeded for seed={seed}"


# ==============================
# Runner
# ==============================
if __name__ == "__main__":
    test_groups = {
        "Fitness - OneMax": [
            test_onemax_all_ones,
            test_onemax_all_zeros,
            test_onemax_mixed,
            test_onemax_single_one,
        ],
        "Fitness - Knapsack Unit": [
            test_knapsack_single_item_within_capacity,
            test_knapsack_two_items_within_capacity,
            test_knapsack_exceeds_capacity,
            test_knapsack_all_items_exceeds_capacity,
            test_knapsack_empty_selection,
            test_knapsack_optimal_selection,
            test_knapsack_exactly_at_capacity,
            test_knapsack_genome_length_mismatch,
        ],
        "Fitness - Knapsack 100 Instances": [
            test_knapsack_empty_genome_always_zero,
            test_knapsack_fitness_non_negative,
            test_knapsack_single_item_always_valid,
            test_knapsack_capacity_scales_with_total_weight,
        ],
        "Selection": [
            test_select_parent_returns_valid_genome,
            test_select_parent_prefers_higher_fitness,
            test_select_parent_never_returns_outside_population,
        ],
        "Crossover": [
            test_crossover_offspring_length,
            test_crossover_bits_are_valid,
            test_crossover_skipped_returns_originals,
            test_crossover_produces_recombination,
        ],
        "Mutation": [
            test_mutation_preserves_length,
            test_mutation_bits_are_valid,
            test_mutation_returns_tuple,
            test_mutation_flips_all_bits_at_max_rate,
        ],
        "GA Improvement - OneMax": [
            test_onemax_fitness_improves,
            test_onemax_converges_to_optimal,
            test_onemax_history_is_non_decreasing,
        ],
        "GA Improvement - Knapsack 10 Instances": [
            test_knapsack_fitness_improves_10_instances,
            test_knapsack_solution_respects_capacity_10_instances,
        ],
    }

    total_passed = 0
    total_failed = 0

    for group, tests in test_groups.items():
        print(f"\n{f' {group} ':=^50}")
        g_passed = 0
        g_failed = 0
        for test in tests:
            try:
                test()
                print(f"  PASSED: {test.__name__}")
                g_passed += 1
            except Exception as e:
                print(f"  FAILED: {test.__name__}: {e}")
                g_failed += 1
        print(f"  {g_passed}/{g_passed + g_failed} passed")
        total_passed += g_passed
        total_failed += g_failed

    print(f"\n{'=' * 50}")
    print(f"  Total: {total_passed}/{total_passed + total_failed} passed")
    print(f"{'=' * 50}")