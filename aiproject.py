import streamlit as st
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# Set page config
st.set_page_config(layout="wide", page_title="AI Resource Allocator", page_icon="🧠")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #1E88E5; text-align: center; margin-bottom: 2rem;}
    .algorithm-section {background-color: #f5f5f5; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border: 1px solid #ddd;}
    .algorithm-title {font-size: 1.5rem; margin-bottom: 1rem; color: #333; font-weight: bold; border-bottom: 2px solid #1E88E5; padding-bottom: 0.5rem;}
    .process-allocated {background-color: #c8e6c9; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;}
    .process-skipped {background-color: #ffcdd2; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;}
    .stButton button {width: 100%; margin-top: 1rem;}
    .process-details {margin-top: 2rem; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🧠 AI-Based Resource Allocator</div>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("## 🎛️ Resource Configuration")
cpu_total = st.sidebar.number_input("Total CPU Cores", min_value=1, value=8)
mem_total = st.sidebar.number_input("Total Memory (GB)", min_value=1, value=10)
num_processes = st.sidebar.number_input("Number of Processes", min_value=1, max_value=20, value=5)

# Algorithm parameters
st.sidebar.markdown("## ⚙️ Algorithm Parameters")
POP_SIZE = st.sidebar.slider("GA: Population Size", min_value=10, max_value=100, value=20)
GENS = st.sidebar.slider("GA: Generations", min_value=10, max_value=100, value=50)
MUTATION_RATE = st.sidebar.slider("GA: Mutation Rate", min_value=0.01, max_value=0.5, value=0.1)
MCTS_ITERATIONS = st.sidebar.slider("MCTS: Iterations", min_value=10, max_value=200, value=50)

available_resources = {"cpu": cpu_total, "memory": mem_total}

st.markdown("<div class='process-details'>", unsafe_allow_html=True)
st.subheader("📋 Enter Process Details")

# Generate random processes
if st.button("🎲 Generate Random Processes"):
    for i in range(num_processes):
        st.session_state[f"cpu{i}"] = random.randint(1, max(1, cpu_total // 2))
        st.session_state[f"mem{i}"] = random.randint(1, max(1, mem_total // 2))
        st.session_state[f"cpu_freed{i}"] = random.randint(0, st.session_state[f"cpu{i}"])
        st.session_state[f"mem_freed{i}"] = random.randint(0, st.session_state[f"mem{i}"])

# Process inputs
processes = []
cols = st.columns(5)
process_colors = {}

for i in range(num_processes):
    col_idx = i % 5
    with cols[col_idx]:
        st.markdown(f"#### Process {i+1}")
        cpu = st.number_input(f"CPU Required", min_value=1, value=st.session_state.get(f"cpu{i}", 1), key=f"cpu{i}")
        mem = st.number_input(f"Memory Required", min_value=1, value=st.session_state.get(f"mem{i}", 1), key=f"mem{i}")
        cpu_freed = st.number_input(f"CPU Freed", min_value=0, max_value=cpu, value=st.session_state.get(f"cpu_freed{i}", 0), key=f"cpu_freed{i}")
        mem_freed = st.number_input(f"Memory Freed", min_value=0, max_value=mem, value=st.session_state.get(f"mem_freed{i}", 0), key=f"mem_freed{i}")
        
        # Calculate resource intensity for color
        intensity = (cpu/cpu_total + mem/mem_total) / 2
        color = f"rgb({int(255 * intensity)}, {int(100 * (1-intensity))}, {int(150 * (1-intensity))})"
        process_colors[i+1] = color
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 5px; border-radius: 5px; color: white; text-align: center;">
            Process {i+1}<br>CPU: {cpu} | Mem: {mem}<br>Freed: CPU {cpu_freed} | Mem {mem_freed}
        </div>
        """, unsafe_allow_html=True)
        
    processes.append({
        "id": i+1, 
        "cpu": cpu, 
        "memory": mem, 
        "cpu_freed": cpu_freed, 
        "memory_freed": mem_freed
    })

st.markdown("</div>", unsafe_allow_html=True)

# Calculate scores for processes
def calculate_score(process, processes):
    max_cpu_req = max(p["cpu"] for p in processes)
    max_mem_req = max(p["memory"] for p in processes)
    max_cpu_freed = max(p["cpu_freed"] for p in processes) if any(p["cpu_freed"] > 0 for p in processes) else 1
    max_mem_freed = max(p["memory_freed"] for p in processes) if any(p["memory_freed"] > 0 for p in processes) else 1
    
    # Normalize requirements and freed resources
    norm_cpu_req = process["cpu"] / max_cpu_req if max_cpu_req > 0 else 0
    norm_mem_req = process["memory"] / max_mem_req if max_mem_req > 0 else 0
    norm_cpu_freed = process["cpu_freed"] / max_cpu_freed if max_cpu_freed > 0 else 0
    norm_mem_freed = process["memory_freed"] / max_mem_freed if max_mem_freed > 0 else 0
    
    # Calculate score (benefit vs cost)
    req_score = (norm_cpu_req + norm_mem_req) / 2
    freed_score = (norm_cpu_freed + norm_mem_freed) / 2
    
    return 0.6 * freed_score - 0.4 * req_score

# Add scores to processes
for p in processes:
    p["score"] = calculate_score(p, processes)

# Visualization function
def visualize_allocation(allocated, skipped, remaining, title):
    st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resource usage chart
        used_cpu = cpu_total - remaining['cpu']
        used_mem = mem_total - remaining['memory']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [used_cpu, used_mem], width, label='Used', color=['#2196F3', '#FF9800'])
        ax.bar(x - width/2, [remaining['cpu'], remaining['memory']], width, bottom=[used_cpu, used_mem], label='Free', color=['#90CAF9', '#FFCC80'])
        
        ax.set_ylabel('Amount')
        ax.set_title('Resource Utilization')
        ax.set_xticks(x)
        ax.set_xticklabels(['CPU', 'Memory'])
        ax.legend()
        
        st.pyplot(fig)
    
    with col2:
        # Process allocation details
        st.markdown("<h4>🟢 Allocated Processes</h4>", unsafe_allow_html=True)
        for p in allocated:
            st.markdown(f"""
            <div class='process-allocated'>
                <b>Process {p['id']}</b> - CPU: {p['cpu']} | Memory: {p['memory']} | 
                Freed: CPU {p['cpu_freed']} | Memory {p['memory_freed']} | Score: {p['score']:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<h4>🔴 Skipped Processes</h4>", unsafe_allow_html=True)
        if skipped:
            for p in skipped:
                st.markdown(f"""
                <div class='process-skipped'>
                    <b>Process {p['id']}</b> - CPU: {p['cpu']} | Memory: {p['memory']} | 
                    Would Free: CPU {p['cpu_freed']} | Memory {p['memory_freed']} | Score: {p['score']:.2f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("All processes allocated!")
            
        st.write(f"**Remaining:** CPU: {remaining['cpu']} | Memory: {remaining['memory']}")

# --- Score-based Allocation ---
st.markdown("<div class='algorithm-section'>", unsafe_allow_html=True)
st.markdown("<div class='algorithm-title'>1️⃣ Score-Based Allocation</div>", unsafe_allow_html=True)

def score_based_allocation(processes, available):
    # Sort processes by score (highest first)
    sorted_processes = sorted(processes, key=lambda p: p["score"], reverse=True)
    allocated, skipped = [], []
    
    # Create a deep copy of available resources
    resources = available.copy()
    
    for p in sorted_processes:
        if p["cpu"] <= resources["cpu"] and p["memory"] <= resources["memory"]:
            # Allocate resources
            resources["cpu"] -= p["cpu"]
            resources["memory"] -= p["memory"]
            
            # Free resources
            resources["cpu"] += p["cpu_freed"]
            resources["memory"] += p["memory_freed"]
            
            allocated.append(p)
        else:
            skipped.append(p)
    
    return allocated, skipped, resources

st.write("This algorithm prioritizes processes based on their score (freed resources vs. required resources).")
if st.button("▶️ Run Score-Based Allocation"):
    with st.spinner("Running score-based allocation..."):
        allocated, skipped, remaining = score_based_allocation(processes, available_resources.copy())
    
    st.success("✅ Score-Based Allocation Complete")
    visualize_allocation(allocated, skipped, remaining, "Score-Based Allocation Results")

st.markdown("</div>", unsafe_allow_html=True)

# --- Heuristic Allocation ---
st.markdown("<div class='algorithm-section'>", unsafe_allow_html=True)
st.markdown("<div class='algorithm-title'>2️⃣ Heuristic Allocation</div>", unsafe_allow_html=True)

def heuristic(process):
    return process["cpu"] + process["memory"] - (process["cpu_freed"] + process["memory_freed"])

def heuristic_allocation(processes, available):
    # Use min heap for processes with lowest resource requirements minus freed resources
    pq = [(heuristic(p), p["id"], p) for p in processes]
    heapq.heapify(pq)
    allocated, skipped = [], []
    resources = available.copy()
    
    while pq:
        _, _, p = heapq.heappop(pq)
        if p["cpu"] <= resources["cpu"] and p["memory"] <= resources["memory"]:
            resources["cpu"] -= p["cpu"]
            resources["memory"] -= p["memory"]
            
            # Add freed resources back
            resources["cpu"] += p["cpu_freed"]
            resources["memory"] += p["memory_freed"]
            
            allocated.append(p)
        else:
            skipped.append(p)
    
    return allocated, skipped, resources

st.write("This algorithm prioritizes processes based on their net resource requirements.")
if st.button("▶️ Run Heuristic Allocation"):
    with st.spinner("Running heuristic allocation..."):
        allocated, skipped, remaining = heuristic_allocation(processes, available_resources.copy())
    
    st.success("✅ Heuristic Allocation Complete")
    visualize_allocation(allocated, skipped, remaining, "Heuristic Allocation Results")

st.markdown("</div>", unsafe_allow_html=True)

# --- Genetic Algorithm ---
st.markdown("<div class='algorithm-section'>", unsafe_allow_html=True)
st.markdown("<div class='algorithm-title'>3️⃣ Genetic Algorithm</div>", unsafe_allow_html=True)

def evaluate_chromosome(chromosome, processes, total_resources):
    """Calculate fitness for a chromosome (sequence of process indices)"""
    remaining = total_resources.copy()
    allocated_count = 0
    total_score = 0
    executed = set()
    
    for idx in chromosome:
        if idx in executed:
            continue
            
        p = processes[idx]
        if p["cpu"] <= remaining["cpu"] and p["memory"] <= remaining["memory"]:
            # Allocate resources
            remaining["cpu"] -= p["cpu"]
            remaining["memory"] -= p["memory"]
            
            # Free resources
            remaining["cpu"] += p["cpu_freed"]
            remaining["memory"] += p["memory_freed"]
            
            allocated_count += 1
            total_score += p["score"]
            executed.add(idx)
    
    # Return allocated count and score (we want to maximize both)
    return allocated_count, total_score

def crossover(parent1, parent2):
    """Create a child by combining parts of two parents"""
    # Single point crossover
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point]
    
    # Add genes from parent2 that aren't already in the child
    for gene in parent2:
        if gene not in child:
            child.append(gene)
    
    return child

def mutate(chromosome):
    """Mutate a chromosome by swapping two positions"""
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def genetic_algorithm(processes, total_resources):
    n = len(processes)
    # Initialize population with random permutations
    population = [random.sample(range(n), n) for _ in range(POP_SIZE)]
    best_fitness_history = []
    
    for gen in range(GENS):
        # Evaluate fitness for each chromosome
        fitness_values = [evaluate_chromosome(chromosome, processes, total_resources) for chromosome in population]
        
        # Sort population by fitness (allocated count first, then score)
        population_fitness = list(zip(population, fitness_values))
        population_fitness.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        
        # Keep track of best solution
        best_chromosome, best_fitness = population_fitness[0]
        best_fitness_history.append(best_fitness)
        
        # Create the next generation
        next_gen = [best_chromosome]  # Elitism - keep best solution
        
        # Fill rest of population through selection and crossover
        while len(next_gen) < POP_SIZE:
            # Tournament selection
            tournament_size = 3
            candidates = random.sample(population_fitness, tournament_size)
            parent1 = max(candidates, key=lambda x: (x[1][0], x[1][1]))[0]
            
            candidates = random.sample(population_fitness, tournament_size)
            parent2 = max(candidates, key=lambda x: (x[1][0], x[1][1]))[0]
            
            # Create child and possibly mutate
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)
        
        population = next_gen
    
    # Return best solution from final population
    fitness_values = [evaluate_chromosome(chromosome, processes, total_resources) for chromosome in population]
    best_idx = fitness_values.index(max(fitness_values, key=lambda x: (x[0], x[1])))
    best_chromosome = population[best_idx]
    
    return best_chromosome, best_fitness_history

st.write("This algorithm evolves different allocation orders to find an optimal solution.")
if st.button("▶️ Run Genetic Algorithm"):
    with st.spinner("Running genetic algorithm..."):
        best_chromosome, fitness_history = genetic_algorithm(processes, available_resources.copy())
    
    # Process results using the best chromosome
    allocated = []
    skipped = []
    remaining = available_resources.copy()
    executed = set()
    
    for idx in best_chromosome:
        if idx in executed:
            continue
            
        p = processes[idx]
        if p["cpu"] <= remaining["cpu"] and p["memory"] <= remaining["memory"]:
            # Allocate resources
            remaining["cpu"] -= p["cpu"]
            remaining["memory"] -= p["memory"]
            
            # Free resources
            remaining["cpu"] += p["cpu_freed"]
            remaining["memory"] += p["memory_freed"]
            
            allocated.append(p)
            executed.add(idx)
        else:
            skipped.append(p)
    
    # Add any processes not in the chromosome yet
    for i, p in enumerate(processes):
        if i not in executed and p not in allocated and p not in skipped:
            skipped.append(p)
    
    # Visualize evolution
    fig, ax = plt.subplots(figsize=(6, 3))
    generations = range(1, len(fitness_history) + 1)
    allocated_counts = [f[0] for f in fitness_history]
    scores = [f[1] for f in fitness_history]
    
    ax.plot(generations, allocated_counts, marker='o', label='Allocated Processes')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of Allocated Processes')
    ax.set_title('Genetic Algorithm Evolution')
    ax.grid(True)
    st.pyplot(fig)
    
    st.success(f"✅ GA Complete - Allocated {len(allocated)}/{len(processes)} processes")
    visualize_allocation(allocated, skipped, remaining, "Genetic Algorithm Results")

st.markdown("</div>", unsafe_allow_html=True)

# --- A* Search ---
st.markdown("<div class='algorithm-section'>", unsafe_allow_html=True)
st.markdown("<div class='algorithm-title'>4️⃣ A* Search</div>", unsafe_allow_html=True)

def a_star_allocation(processes, total_resources):
    """A* search for resource allocation with dynamic cost and heuristic functions"""
    
    # Define state as (cpu_left, mem_left, executed_processes)
    start = (total_resources["cpu"], total_resources["memory"], tuple())
    goal = len(processes)  # We want to execute as many processes as possible
    
    # Priority queue for open set (f_score, state)
    open_set = [(0, start)]
    # Closed set to avoid revisiting states
    closed = set()
    
    # g_score: cost to reach a state
    g_score = {start: 0}
    # f_score: estimated total cost
    f_score = {start: len(processes)}
    
    # Heuristic function: estimate of processes left to allocate
    def h(state):
        cpu_left, mem_left, executed = state
        return goal - len(executed)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if len(current[2]) == goal:
            # Found solution with all processes
            return current[2]
            
        # If no more improvement possible or state already explored
        if current in closed:
            continue
            
        closed.add(current)
        cpu_left, mem_left, executed = current
        
        # Try each process that hasn't been executed
        for i, p in enumerate(processes):
            if i in executed:
                continue
                
            if p["cpu"] <= cpu_left and p["memory"] <= mem_left:
                # Calculate new state after executing process
                new_cpu = cpu_left - p["cpu"] + p["cpu_freed"]
                new_mem = mem_left - p["memory"] + p["memory_freed"]
                new_executed = executed + (i,)
                new_state = (new_cpu, new_mem, new_executed)
                
                # Calculate cost (negative of process score to minimize)
                cost = -p["score"]
                tentative_g = g_score[current] + cost
                
                if new_state not in g_score or tentative_g < g_score[new_state]:
                    # Found a better path to this state
                    g_score[new_state] = tentative_g
                    f_score[new_state] = tentative_g + h(new_state)
                    heapq.heappush(open_set, (f_score[new_state], new_state))
    
    # Return the best partial solution if can't allocate all processes
    best_state = max(closed, key=lambda s: len(s[2]))
    return best_state[2]

st.write("This algorithm searches through the solution space using a heuristic function to guide the search.")
if st.button("▶️ Run A* Search"):
    with st.spinner("Running A* search..."):
        allocated_indices = a_star_allocation(processes, available_resources.copy())
    
    # Process results
    allocated = []
    skipped = []
    remaining = available_resources.copy()
    executed = set()
    
    for i in allocated_indices:
        p = processes[i]
        remaining["cpu"] -= p["cpu"]
        remaining["memory"] -= p["memory"]
        
        # Add freed resources back
        remaining["cpu"] += p["cpu_freed"]
        remaining["memory"] += p["memory_freed"]
        
        allocated.append(p)
        executed.add(i)
    
    # Add any processes not in the solution
    for i, p in enumerate(processes):
        if i not in executed:
            skipped.append(p)
    
    st.success(f"✅ A* Search Complete - Found allocation for {len(allocated_indices)}/{len(processes)} processes")
    visualize_allocation(allocated, skipped, remaining, "A* Search Results")

st.markdown("</div>", unsafe_allow_html=True)

# Compare all algorithms
st.markdown("<div class='algorithm-section'>", unsafe_allow_html=True)
st.markdown("<div class='algorithm-title'>🏆 Algorithm Comparison</div>", unsafe_allow_html=True)
st.write("Compare the performance of all algorithms on the current process set.")

if st.button("Run All Algorithms"):
    with st.spinner("Running all algorithms..."):
        results = {}
        
        # Score-based
        allocated, skipped, remaining = score_based_allocation(processes, available_resources.copy())
        results["Score-based"] = {
            "allocated": len(allocated), 
            "skipped": len(skipped),
            "remaining_cpu": remaining["cpu"],
            "remaining_mem": remaining["memory"]
        }
        
        # Heuristic
        allocated, skipped, remaining = heuristic_allocation(processes, available_resources.copy())
        results["Heuristic"] = {
            "allocated": len(allocated), 
            "skipped": len(skipped),
            "remaining_cpu": remaining["cpu"],
            "remaining_mem": remaining["memory"]
        }
        
        # Genetic Algorithm
        best_chromosome, _ = genetic_algorithm(processes, available_resources.copy())
        allocated = []
        remaining = available_resources.copy()
        executed = set()
        
        for idx in best_chromosome:
            if idx in executed:
                continue
                
            p = processes[idx]
            if p["cpu"] <= remaining["cpu"] and p["memory"] <= remaining["memory"]:
                remaining["cpu"] -= p["cpu"]
                remaining["memory"] -= p["memory"]
                
                # Free resources
                remaining["cpu"] += p["cpu_freed"]
                remaining["memory"] += p["memory_freed"]
                
                allocated.append(p)
                executed.add(idx)
        
        results["GA"] = {
            "allocated": len(allocated), 
            "skipped": len(processes) - len(allocated),
            "remaining_cpu": remaining["cpu"],
            "remaining_mem": remaining["memory"]
        }
        
        # A*
        allocated_indices = a_star_allocation(processes, available_resources.copy())
        allocated = []
        remaining = available_resources.copy()
        
        for i in allocated_indices:
            p = processes[i]
            remaining["cpu"] -= p["cpu"]
            remaining["memory"] -= p["memory"]
            
            # Add freed resources back
            remaining["cpu"] += p["cpu_freed"]
            remaining["memory"] += p["memory_freed"]
            
            allocated.append(p)
        
        results["A*"] = {
            "allocated": len(allocated), 
            "skipped": len(processes) - len(allocated),
            "remaining_cpu": remaining["cpu"],
            "remaining_mem": remaining["memory"]
        }
    
    # Display comparison chart
    algorithms = list(results.keys())
    allocated_counts = [results[alg]["allocated"] for alg in algorithms]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax.bar(x, allocated_counts, width, label='Allocated', color='#4CAF50')
    ax.bar(x, [results[alg]["skipped"] for alg in algorithms], width, bottom=allocated_counts, label='Skipped', color='#F44336')
    
    ax.set_ylabel('Number of Processes')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    st.pyplot(fig)
    
    # Display resource utilization
    st.subheader("Resource Utilization by Algorithm")
    data = []
    for alg in algorithms:
        data.append({
            "Algorithm": alg,
            "Allocated": results[alg]["allocated"],
            "Skipped": results[alg]["skipped"],
            "CPU Left": results[alg]["remaining_cpu"],
            "Memory Left": results[alg]["remaining_mem"],
            "CPU Utilization": (1 - results[alg]["remaining_cpu"]/cpu_total) * 100,
            "Memory Utilization": (1 - results[alg]["remaining_mem"]/mem_total) * 100,
        })
    
    st.table(data)

st.markdown("</div>", unsafe_allow_html=True)