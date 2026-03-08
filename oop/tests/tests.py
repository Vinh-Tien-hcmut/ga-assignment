import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oop_ga import (
    Chromosome,
    Population,
    OneMaxFitness,
    KnapsackFitness,
    FitnessFunctionAdapter,
    TournamentSelection,
    OnePointCrossover,
    BitFlipMutation,
    GeneticAlgorithm,
    GAResult,
    generate_inventory,
    Item,
    GENOME_LENGTH,
    SEED,
    TOURNAMENT_K,
    CROSSOVER_RATE,
    MUTATION_RATE,
)

# ==============================
# Shared fixtures
# ==============================
SMALL_ITEMS = (
    Item("A", value=10, weight=5),
    Item("B", value=6,  weight=4),
    Item("C", value=3,  weight=3),
)
SMALL_CAPACITY = 8

_base_rng  = random.Random(SEED)
SEEDS_100  = [_base_rng.randint(0, 10 ** 6) for _ in range(100)]
SEEDS_10   = SEEDS_100[:10]

onemax_ff   = OneMaxFitness()
knapsack_ff = KnapsackFitness(SMALL_ITEMS, SMALL_CAPACITY)


def _make_ga(fitness_fn, rng, target=None):
    return GeneticAlgorithm(
        fitness_fn     = fitness_fn,
        selection      = TournamentSelection(k=TOURNAMENT_K, rng=rng),
        crossover      = OnePointCrossover(rate=CROSSOVER_RATE, rng=rng),
        mutation       = BitFlipMutation(rate=MUTATION_RATE, rng=rng),
        rng            = rng,
        tarfitness = target,
        verbose = False
    )


# ==============================
# 1. Chromosome (4 tests)
# ==============================
def test_chromosome_stores_genome():
    c = Chromosome([1, 0, 1])
    assert c.genome == [1, 0, 1]

def test_chromosome_fitness_none_before_evaluate():
    c = Chromosome([1, 0, 1])
    assert c.fitness is None

def test_chromosome_evaluate_sets_fitness():
    c = Chromosome([1, 1, 1, 0, 0])
    c.evaluate(onemax_ff)
    assert c.fitness == 3

def test_chromosome_copy_is_independent():
    c = Chromosome([1, 0, 1])
    c2 = c.copy()
    c2.genome[0] = 0
    assert c.genome[0] == 1   # original must not be affected


# ==============================
# 2. Fitness - OneMax (4 tests)
# ==============================
def test_onemax_all_ones():
    c = Chromosome([1] * 5)
    assert onemax_ff.evaluate(c) == 5

def test_onemax_all_zeros():
    c = Chromosome([0] * 5)
    assert onemax_ff.evaluate(c) == 0

def test_onemax_mixed():
    c = Chromosome([1, 0, 1, 0, 1])
    assert onemax_ff.evaluate(c) == 3

def test_onemax_single_one():
    c = Chromosome([0, 0, 1, 0, 0])
    assert onemax_ff.evaluate(c) == 1


# ==============================
# 3. Fitness - Knapsack unit (8 tests)
# ==============================
def test_knapsack_single_item_within_capacity():
    # Only A, weight=5 <= 8
    assert knapsack_ff.evaluate(Chromosome([1, 0, 0])) == 10

def test_knapsack_two_items_within_capacity():
    # B + C, weight=7 <= 8
    assert knapsack_ff.evaluate(Chromosome([0, 1, 1])) == 9

def test_knapsack_exceeds_capacity():
    # A + B, weight=9 > 8, penalty applied
    assert knapsack_ff.evaluate(Chromosome([1, 1, 0])) == 0

def test_knapsack_all_items_exceeds_capacity():
    # All items, weight=12 > 8, penalty applied
    assert knapsack_ff.evaluate(Chromosome([1, 1, 1])) == 0

def test_knapsack_empty_selection():
    assert knapsack_ff.evaluate(Chromosome([0, 0, 0])) == 0

def test_knapsack_optimal_selection():
    # A + C = value 13, weight 8 = capacity (valid)
    assert knapsack_ff.evaluate(Chromosome([1, 0, 1])) == 13

def test_knapsack_exactly_at_capacity():
    # weight == capacity is still valid
    assert knapsack_ff.evaluate(Chromosome([1, 0, 1])) == 13

def test_knapsack_fitness_non_negative():
    # Fitness must always be >= 0 across 100 instances
    for seed in SEEDS_100:
        rng = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng)
        ff = KnapsackFitness(inventory, capacity)
        assert ff.evaluate(Chromosome([1] * GENOME_LENGTH)) >= 0


# ==============================
# 4. Fitness - Adapter (2 tests)
# ==============================
def test_adapter_wraps_callable():
    adapted = FitnessFunctionAdapter(lambda genome: sum(genome))
    c = Chromosome([1, 1, 0, 1])
    assert adapted.evaluate(c) == 3

def test_adapter_interchangeable_with_onemax():
    adapted = FitnessFunctionAdapter(lambda genome: sum(genome))
    c = Chromosome([1] * 10)
    assert adapted.evaluate(c) == onemax_ff.evaluate(c)


# ==============================
# 5. Population (3 tests)
# ==============================
def test_population_generate_correct_size():
    rng = random.Random(0)
    pop = Population.generate(20, 10, onemax_ff, rng)
    assert len(pop) == 20

def test_population_best_has_highest_fitness():
    rng = random.Random(0)
    pop = Population.generate(50, GENOME_LENGTH, onemax_ff, rng)
    best = pop.best()
    assert all(best.fitness >= c.fitness for c in pop.chromosomes)

def test_population_elites_are_sorted():
    rng = random.Random(0)
    pop = Population.generate(20, GENOME_LENGTH, onemax_ff, rng)
    elites = pop.elites(3)
    assert elites[0].fitness >= elites[1].fitness >= elites[2].fitness


# ==============================
# 6. Selection (3 tests)
# ==============================
def test_tournament_returns_chromosome_from_population():
    rng = random.Random(0)
    pop = Population.generate(20, 5, onemax_ff, rng)
    sel = TournamentSelection(k=3, rng=rng)
    winner = sel.select(pop)
    assert winner in pop.chromosomes

def test_tournament_prefers_higher_fitness():
    # 50 all-ones vs 50 all-zeros — best should win majority
    rng = random.Random(0)
    best  = Chromosome([1] * 10)
    worst = Chromosome([0] * 10)
    best.evaluate(onemax_ff)
    worst.evaluate(onemax_ff)
    pop = Population([best] * 50 + [worst] * 50)
    sel = TournamentSelection(k=3, rng=rng)
    wins = sum(1 for _ in range(200) if sel.select(pop) is best or sel.select(pop).fitness == 10)
    assert wins > 100

def test_tournament_never_returns_outside_population():
    for seed in SEEDS_100:
        rng = random.Random(seed)
        pop = Population.generate(20, GENOME_LENGTH, onemax_ff, rng)
        sel = TournamentSelection(k=3, rng=rng)
        winner = sel.select(pop)
        assert winner in pop.chromosomes


# ==============================
# 7. Crossover (4 tests)
# ==============================
def test_crossover_offspring_correct_length():
    rng = random.Random(0)
    cx  = OnePointCrossover(rate=1.0, rng=rng)
    p1  = Chromosome([1] * 10)
    p2  = Chromosome([0] * 10)
    o1, o2 = cx.crossover(p1, p2)
    assert len(o1.genome) == 10 and len(o2.genome) == 10

def test_crossover_bits_are_valid():
    rng = random.Random(0)
    cx  = OnePointCrossover(rate=1.0, rng=rng)
    p1  = Chromosome([1] * 10)
    p2  = Chromosome([0] * 10)
    o1, o2 = cx.crossover(p1, p2)
    assert all(b in (0, 1) for b in o1.genome + o2.genome)

def test_crossover_skipped_returns_copies():
    # rate=0.0 — crossover never happens, offspring are copies of parents
    rng = random.Random(0)
    cx  = OnePointCrossover(rate=0.0, rng=rng)
    p1  = Chromosome([1, 0, 1, 0, 1])
    p2  = Chromosome([0, 1, 0, 1, 0])
    o1, o2 = cx.crossover(p1, p2)
    assert o1.genome == p1.genome and o2.genome == p2.genome

def test_crossover_produces_recombination():
    rng     = random.Random(5)
    cx      = OnePointCrossover(rate=1.0, rng=rng)
    p1      = Chromosome([1] * 10)
    p2      = Chromosome([0] * 10)
    results = [cx.crossover(p1, p2) for _ in range(20)]
    mixed   = [(o1, o2) for o1, o2 in results if o1.genome != p1.genome]
    assert len(mixed) > 0


# ==============================
# 8. Mutation (4 tests)
# ==============================
def test_mutation_preserves_length():
    rng = random.Random(0)
    mut = BitFlipMutation(rate=MUTATION_RATE, rng=rng)
    c   = Chromosome([1, 0] * 5)
    assert len(mut.mutate(c).genome) == 10

def test_mutation_bits_are_valid():
    rng = random.Random(0)
    mut = BitFlipMutation(rate=MUTATION_RATE, rng=rng)
    c   = Chromosome([1] * 20)
    assert all(b in (0, 1) for b in mut.mutate(c).genome)

def test_mutation_returns_new_chromosome():
    rng = random.Random(0)
    mut = BitFlipMutation(rate=MUTATION_RATE, rng=rng)
    c   = Chromosome([1] * 10)
    assert mut.mutate(c) is not c   # must be a new object

def test_mutation_rate_one_flips_all():
    # At rate=1.0 every bit must flip
    rng = random.Random(0)
    mut = BitFlipMutation(rate=1.0, rng=rng)
    c   = Chromosome([1, 0, 1, 0, 1])
    assert mut.mutate(c).genome == [0, 1, 0, 1, 0]


# ==============================
# 9. GA improvement - OneMax (3 tests)
# ==============================
def test_onemax_fitness_improves():
    rng    = random.Random(SEED)
    result = _make_ga(OneMaxFitness(), rng, target=GENOME_LENGTH).run()
    assert result.history[-1] > result.history[0]

def test_onemax_converges_to_optimal():
    rng    = random.Random(SEED)
    result = _make_ga(OneMaxFitness(), rng, target=GENOME_LENGTH).run()
    assert result.best_fitness == GENOME_LENGTH

def test_onemax_history_non_decreasing():
    rng    = random.Random(SEED)
    result = _make_ga(OneMaxFitness(), rng, target=GENOME_LENGTH).run()
    assert all(result.history[i] <= result.history[i + 1] for i in range(len(result.history) - 1))


# ==============================
# 10. GA improvement - Knapsack 10 instances (2 tests)
# ==============================
def test_knapsack_fitness_improves_10_instances():
    for seed in SEEDS_10:
        rng_inv = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng_inv)
        rng_ga  = random.Random(seed)
        result  = _make_ga(KnapsackFitness(inventory, capacity), rng_ga).run()
        assert result.history[-1] >= result.history[0], f"Fitness did not improve for seed={seed}"

def test_knapsack_solution_respects_capacity_10_instances():
    for seed in SEEDS_10:
        rng_inv = random.Random(seed)
        inventory, capacity = generate_inventory(GENOME_LENGTH, rng_inv)
        rng_ga  = random.Random(seed)
        result  = _make_ga(KnapsackFitness(inventory, capacity), rng_ga).run()
        total_weight = sum(
            item.weight for bit, item in zip(result.best_genome, inventory) if bit == 1
        )
        assert total_weight <= capacity, f"Capacity exceeded for seed={seed}"


# ==============================
# Runner
# ==============================
if __name__ == "__main__":
    import sys
    import os
    os.system("")  # enable ANSI on Windows

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    
    test_groups = {
        "Chromosome": [
            test_chromosome_stores_genome,
            test_chromosome_fitness_none_before_evaluate,
            test_chromosome_evaluate_sets_fitness,
            test_chromosome_copy_is_independent,
        ],
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
            test_knapsack_fitness_non_negative,
        ],
        "Fitness - Adapter": [
            test_adapter_wraps_callable,
            test_adapter_interchangeable_with_onemax,
        ],
        "Population": [
            test_population_generate_correct_size,
            test_population_best_has_highest_fitness,
            test_population_elites_are_sorted,
        ],
        "Selection": [
            test_tournament_returns_chromosome_from_population,
            test_tournament_prefers_higher_fitness,
            test_tournament_never_returns_outside_population,
        ],
        "Crossover": [
            test_crossover_offspring_correct_length,
            test_crossover_bits_are_valid,
            test_crossover_skipped_returns_copies,
            test_crossover_produces_recombination,
        ],
        "Mutation": [
            test_mutation_preserves_length,
            test_mutation_bits_are_valid,
            test_mutation_returns_new_chromosome,
            test_mutation_rate_one_flips_all,
        ],
        "GA Improvement - OneMax": [
            test_onemax_fitness_improves,
            test_onemax_converges_to_optimal,
            test_onemax_history_non_decreasing,
        ],
        "GA Improvement - Knapsack 10 Instances": [
            test_knapsack_fitness_improves_10_instances,
            test_knapsack_solution_respects_capacity_10_instances,
        ],
    }

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        all_tests = {
            test.__name__: test
            for tests in test_groups.values()
            for test in tests
        }
        if test_name in all_tests:
            try:
                all_tests[test_name]()
                print(f"  {GREEN}PASSED{RESET}: {test_name}")
            except Exception as e:
                print(f"  {RED}FAILED{RESET}: {test_name}: {e}")
        else:
            print(f"{RED}Test '{test_name}' not found.{RESET}")
            print("Available tests:")
            for name in all_tests:
                print(f"  {name}")

    else:
        total_passed = 0
        total_failed = 0

        for group, tests in test_groups.items():
            print(f"\n{CYAN}{f' {group} ':=^50}{RESET}")  # header
            g_passed = 0
            g_failed = 0

            for test in tests:
                try:
                    test()
                    print(f"  {GREEN}PASSED{RESET}: {test.__name__}")
                    g_passed += 1
                except Exception as e:
                    print(f"  {RED}FAILED{RESET}: {test.__name__}: {e}")
                    g_failed += 1

            print(f"  {g_passed}/{g_passed + g_failed} passed")
            total_passed += g_passed
            total_failed += g_failed

        # Final summary
        print(f"\n{BOLD}{'=' * 50}{RESET}")
        if total_failed == 0:
            print(f"  {GREEN}{BOLD}Total: {total_passed}/{total_passed + total_failed} passed{RESET}")
        else:
            print(f"  {YELLOW}{BOLD}Total: {total_passed}/{total_passed + total_failed} passed{RESET}")
        print(f"{BOLD}{'=' * 50}{RESET}")