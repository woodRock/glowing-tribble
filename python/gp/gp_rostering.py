import operator
import random
import copy
import functools
from deap import algorithms, base, creator, tools, gp
from roster_problem import (
    RosterProblem, Staff, Shift, shifts_overlap, is_staff_available_for_shift,
    ContractDetails, ShiftOffRequest, ShiftOnRequest, CoverRequirement
)
from gp_decisions import (
    D1_Decision, D2_Decision, D3_Decision, D4_Decision, D5_Decision,
    evaluate_hyper_heuristic_roster
)

MAX_HEIGHT = 6 # Maximum height for GP trees
GENERATIONS = 1000
POPULATION_SIZE = 100

# --- 1. DEAP Setup ---

# Define a new fitness type with a single objective (minimization)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define a "MetaIndividual" type that holds PrimitiveTree objects for all decisions
if not hasattr(creator, "MetaIndividual"):
    creator.create("MetaIndividual", list, fitness=creator.FitnessMin)

def initMetaIndividual(icls, d1_expr, d2_expr, d3_expr, d4_expr, d5_expr):
    """Initializes a MetaIndividual with PrimitiveTrees for each decision."""
    return icls([
        gp.PrimitiveTree(d1_expr()),
        gp.PrimitiveTree(d2_expr()),
        gp.PrimitiveTree(d3_expr()),
        gp.PrimitiveTree(d4_expr()),
        gp.PrimitiveTree(d5_expr())
    ])

def evaluate_meta_individual(roster_problem, meta_individual):
    # The orchestrator expects the MetaIndividual and the roster_problem
    fitness, _ = evaluate_hyper_heuristic_roster(meta_individual, roster_problem)
    return fitness,

def cxMetaIndividual(ind1, ind2):
    # Apply one-point crossover to each corresponding PrimitiveTree
    for i in range(len(ind1)):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

def mutMetaIndividual(individual):
    # Use a lambda to provide min_ and max_ to genFull required by gp.mutUniform
    # using min_=0, max_=2 for subtree generation during mutation
    expr = lambda pset, type_: gp.genFull(pset, min_=0, max_=6, type_=type_)
    
    individual[0], = gp.mutUniform(individual[0], expr, D1_Decision.pset)
    individual[1], = gp.mutUniform(individual[1], expr, D2_Decision.pset)
    individual[2], = gp.mutUniform(individual[2], expr, D3_Decision.pset)
    individual[3], = gp.mutUniform(individual[3], expr, D4_Decision.pset)
    individual[4], = gp.mutUniform(individual[4], expr, D5_Decision.pset)
    return individual,

def setup_gp_toolbox(roster_problem):
    """Sets up the DEAP toolbox for the multi-tree GP."""
    toolbox = base.Toolbox()

    # Register expression generators for each decision
    # Using genFull with min_=1, max_=1 for initial simplicity
    toolbox.register("d1_expr", gp.genFull, pset=D1_Decision.pset, min_=2, max_=6)
    toolbox.register("d2_expr", gp.genFull, pset=D2_Decision.pset, min_=2, max_=6)
    toolbox.register("d3_expr", gp.genFull, pset=D3_Decision.pset, min_=2, max_=6)
    toolbox.register("d4_expr", gp.genFull, pset=D4_Decision.pset, min_=2, max_=6)
    toolbox.register("d5_expr", gp.genFull, pset=D5_Decision.pset, min_=2, max_=6)

    # Register the MetaIndividual initializer
    toolbox.register(
        "meta_individual",
        initMetaIndividual,
        creator.MetaIndividual,
        toolbox.d1_expr, toolbox.d2_expr, toolbox.d3_expr, toolbox.d4_expr, toolbox.d5_expr
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.meta_individual)

    # Register the evaluation function for the MetaIndividual using partial
    toolbox.register("evaluate", functools.partial(evaluate_meta_individual, roster_problem))
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Use base.clone
    toolbox.register("clone", copy.deepcopy) 
    
    # Custom crossover for MetaIndividual
    toolbox.register("mate", cxMetaIndividual)
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: max(tree.height for tree in ind), max_value=MAX_HEIGHT))

    # Custom mutation for MetaIndividual
    toolbox.register("mutate", mutMetaIndividual)
    toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: max(tree.height for tree in ind), max_value=MAX_HEIGHT))

    return toolbox

# --- 2. Main Execution Logic ---

def run_gp_rostering(staff_data, shift_data, requests=None, generations=10, population_size=50, cxpb=0.5, mutpb=0.2):
    random.seed(42)
    roster_problem = RosterProblem(staff_data, shift_data, requests=requests)
    toolbox = setup_gp_toolbox(roster_problem)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda data: sum(data)/len(data) if data else 0.0)
    stats.register("min", min)

    print(f"Starting genetic algorithm (gens={generations}, pop={population_size}, cxpb={cxpb}, mutpb={mutpb})...")

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)

    for gen in range(1, generations + 1):
        # Elitism: Select the best individuals to preserve
        elite_count = max(1, int(population_size * 0.05)) # Top 5% elitism
        elites = tools.selBest(pop, elite_count)
        elites = [toolbox.clone(ind) for ind in elites]

        # Select the rest of the offspring
        offspring_count = population_size - elite_count
        offspring = toolbox.select(pop, offspring_count)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation to the offspring
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the elites and offspring
        pop[:] = elites + offspring

        # Update the hall of fame
        hof.update(pop)
        
        # Log progress
        record = stats.compile(pop)
        if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen}: Min {record['min']:.4f}, Avg {record['avg']:.4f}")

    best_meta_individual = hof[0]
    
    # Extract the best individual for each decision
    best_d1_individual = best_meta_individual[0]
    best_d2_individual = best_meta_individual[1]
    best_d3_individual = best_meta_individual[2]
    best_d4_individual = best_meta_individual[3]
    best_d5_individual = best_meta_individual[4]

    print(f"Best overall fitness: {best_meta_individual.fitness.values[0]}")

    # Generate the final roster using the best MetaIndividual
    final_fitness, final_assignments = evaluate_hyper_heuristic_roster(best_meta_individual, roster_problem)
    final_roster_details = roster_problem.get_roster_details(final_assignments)
    
    return final_roster_details, {
        "D1": str(best_d1_individual),
        "D2": str(best_d2_individual),
        "D3": str(best_d3_individual),
        "D4": str(best_d4_individual),
        "D5": str(best_d5_individual),
    }

# --- 3. Example Usage (Dummy Data) ---

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # Dummy classes necessary for the example usage block to run standalone
    # (Classes are now imported from roster_problem.py)
    
    print("Setting up data structures...")

    staff_data = [
        {
            'id': 'S1', 
            'skills': ['Cashier', 'Barista'], 
            'availability_slots': [
                {'start_time': datetime(2025, 12, 8, 8, 0), 'end_time': datetime(2025, 12, 8, 18, 0)},
                {'start_time': datetime(2025, 12, 9, 8, 0), 'end_time': datetime(2025, 12, 9, 20, 0)},
                {'start_time': datetime(2025, 12, 10, 8, 0), 'end_time': datetime(2025, 12, 10, 18, 0)},
                {'start_time': datetime(2025, 12, 11, 8, 0), 'end_time': datetime(2025, 12, 11, 18, 0)},
            ],
            'contract_details': [
                ContractDetails(min_rest_time=60, max_workload_minutes=8*60, max_seq_shifts={'value': 3}, min_seq_days_off={'value': 1})
            ],
            'preferences': {'desired_hours': 7}
        },
        {
            'id': 'S2', 
            'skills': ['Barista', 'Manager'], 
            'availability_slots': [
                {'start_time': datetime(2025, 12, 8, 10, 0), 'end_time': datetime(2025, 12, 8, 20, 0)},
                {'start_time': datetime(2025, 12, 9, 10, 0), 'end_time': datetime(2025, 12, 9, 20, 0)},
                {'start_time': datetime(2025, 12, 10, 10, 0), 'end_time': datetime(2025, 12, 10, 20, 0)},
            ],
            'contract_details': [
                ContractDetails(min_rest_time=60, max_workload_minutes=10*60, max_seq_shifts={'value': 4}, max_weekend_patterns=1)
            ],
            'preferences': {'desired_hours': 8}
        },
        {
            'id': 'S3', 
            'skills': ['Cashier'], 
            'availability_slots': [
                {'start_time': datetime(2025, 12, 8, 9, 0), 'end_time': datetime(2025, 12, 8, 17, 0)},
                {'start_time': datetime(2025, 12, 9, 9, 0), 'end_time': datetime(2025, 12, 9, 17, 0)},
                {'start_time': datetime(2025, 12, 10, 9, 0), 'end_time': datetime(2025, 12, 10, 17, 0)},
                {'start_time': datetime(2025, 12, 11, 9, 0), 'end_time': datetime(2025, 12, 11, 17, 0)},
            ],
            'contract_details': [
                ContractDetails(min_rest_time=30, max_workload_minutes=7*60)
            ],
            'preferences': {'desired_hours': 6}
        },
    ]

    shift_data = [
        {'id': 'SH1', 'required_skills': ['Cashier'], 'start_time': datetime(2025, 12, 8, 9, 0), 'end_time': datetime(2025, 12, 8, 13, 0), 'role': 'Cashier'},
        {'id': 'SH2', 'required_skills': ['Barista'], 'start_time': datetime(2025, 12, 8, 12, 0), 'end_time': datetime(2025, 12, 8, 16, 0), 'role': 'Barista'}, 
        {'id': 'SH3', 'required_skills': ['Cashier', 'Barista'], 'start_time': datetime(2025, 12, 9, 15, 0), 'end_time': datetime(2025, 12, 9, 19, 0), 'role': 'AllRounder', 'min_staff_for_role': 1},
        {'id': 'SH4', 'required_skills': ['Manager'], 'start_time': datetime(2025, 12, 10, 10, 0), 'end_time': datetime(2025, 12, 10, 14, 0), 'role': 'Manager'},
        {'id': 'SH5', 'required_skills': ['Cashier'], 'start_time': datetime(2025, 12, 11, 9, 0), 'end_time': datetime(2025, 12, 11, 13, 0), 'role': 'Cashier'},
        {'id': 'SH6', 'required_skills': ['Barista'], 'start_time': datetime(2025, 12, 11, 14, 0), 'end_time': datetime(2025, 12, 11, 18, 0), 'role': 'Barista'},
    ]

    requests_data = {
        'shift_off': [
            {'employee_id': 'S1', 'day_offset': 0, 'shift_id': 'SH1', 'weight': 500} 
        ],
        'shift_on': [
            {'employee_id': 'S2', 'day_offset': 1, 'shift_id': 'Barista', 'weight': 300} 
        ],
        'cover': [
            {'day_offset': 0, 'shift_id': 'Cashier', 'min_cover': 1, 'max_cover': 1, 'min_weight': 200, 'max_weight': 100}
        ]
    }

    print("Running GP for roster generation...")
    try:
        best_roster, best_heuristic_trees = run_gp_rostering(staff_data, shift_data, requests=requests_data, generations=1000, population_size=100)
        print("\n--- Best Roster Details ---")
        print(best_roster)
        print("\n--- Best Heuristic Trees ---")
        for decision, tree in best_heuristic_trees.items():
            print(f"{decision}: {tree}")
    except NameError as e:
        print(f"\nError: A NameError occurred. Ensure RosterProblem, Staff, Shift, and all GP-related classes are correctly imported and defined. Details: {e}")