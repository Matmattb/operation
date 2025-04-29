import numpy as np
from collections import deque
import time
import random
import matplotlib.pyplot as plt

# Function to read flow network from .txt file
def read_flow_network(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    capacity = np.zeros((n, n), dtype=int)
    for i in range(n):
        row = list(map(int, lines[i + 1].strip().split()))
        for j in range(n):
            capacity[i][j] = row[j]
    return n, capacity

# Function to display a matrix (capacity, cost, or Bellman’s table)
def display_matrix(matrix, title, labels=None):
    print(f"\n{title}:")
    n = matrix.shape[0]
    if labels:
        header = "    " + " ".join(f"{lbl:>4}" for lbl in labels)
        print(header)
        for i, row in enumerate(matrix):
            row_str = f"{labels[i]:>2} |" + " ".join(f"{x:>4}" if x != float('inf') else "  inf" for x in row)
            print(row_str)
    else:
        for row in matrix:
            print(" ".join(f"{x:>2}" for x in row))

# Ford-Fulkerson with Edmonds-Karp (BFS for shortest augmenting paths)
def ford_fulkerson(n, capacity, trace_file):
    flow = np.zeros((n, n), dtype=int)
    residual = capacity.copy()
    max_flow = 0
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]

    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Ford-Fulkerson Execution Trace\n")
        display_matrix(residual, "Capacity table display", vertices)
        f.write("\nCapacity table display:\n")
        for row in residual:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write("\nThe initial residual graph is the starting graph.\n")

        iteration = 1
        while True:
            parent = [-1] * n
            parent[0] = -2
            queue = deque([0])
            path_flow = float('inf')
            path = []
            
            while queue:
                u = queue.popleft()
                for v in range(n):
                    if parent[v] == -1 and residual[u][v] > 0:
                        parent[v] = u
                        queue.append(v)
                        if v == n - 1:
                            break
            
            if parent[n - 1] == -1:
                break
            
            f.write(f"\n* Iteration {iteration}:\n\nBreadth-first search:\n")
            queue = deque([0])
            visited = {0}
            bfs_trace = []
            while queue:
                u = queue.popleft()
                neighbors = []
                for v in range(n):
                    if residual[u][v] > 0 and v not in visited:
                        neighbors.append(vertices[v])
                        visited.add(v)
                        queue.append(v)
                if neighbors:
                    bfs_trace.append(f"{vertices[u]}{''.join(neighbors)};")
                    for v in range(n):
                        if residual[u][v] > 0 and parent[v] == u:
                            f.write(f"    Π({vertices[v]})={vertices[u]}\n")
            
            f.write("\n".join(bfs_trace) + "\n")
            
            v = n - 1
            while v != 0:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                path.append((u, v))
                v = u
            path.reverse()
            
            for u, v in path:
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                flow[u][v] += path_flow
            
            max_flow += path_flow
            path_str = "".join(vertices[u] for u, _ in path) + vertices[path[-1][1]]
            f.write(f"Detection of an improving chain: {path_str} with a flow {path_flow}.\n")
            f.write("\nModifications to the residual graph:\n")
            display_matrix(residual, f"Residual graph after iteration {iteration}", vertices)
            for row in residual:
                f.write(" ".join(f"{x:>2}" for x in row) + "\n")
            
            iteration += 1
        
        f.write("\n* Max flow display:\n")
        flow_display = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                flow_display[i][j] = f"{flow[i][j]}/{capacity[i][j]}" if capacity[i][j] > 0 else "0"
        display_matrix(flow_display, "Max flow display", vertices)
        for row in flow_display:
            f.write(" ".join(f"{x:>5}" for x in row) + "\n")
        f.write(f"\nValue of the max flow = {max_flow}\n")
    
    return max_flow, flow

# Push-Relabel algorithm
def push_relabel(n, capacity, trace_file):
    flow = np.zeros((n, n), dtype=int)
    excess = [0] * n
    height = [0] * n
    height[0] = n
    residual = capacity.copy()
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Push-Relabel Execution Trace\n")
        display_matrix(capacity, "Initial capacity table", vertices)
        f.write("\nInitial capacity table:\n")
        for row in capacity:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        
        for v in range(n):
            if capacity[0][v] > 0:
                flow[0][v] = capacity[0][v]
                residual[0][v] = 0
                residual[v][0] = capacity[0][v]
                excess[v] = capacity[0][v]
                excess[0] -= capacity[0][v]
        
        f.write("\nAfter initialization:\n")
        f.write(f"Excess: {excess}\n")
        f.write(f"Heights: {height}\n")
        
        iteration = 1
        while any(e > 0 for e in excess[1:n-1]):
            max_height = -1
            u = -1
            for i in range(1, n-1):
                if excess[i] > 0 and height[i] > max_height:
                    max_height = height[i]
                    u = i
                elif excess[i] > 0 and height[i] == max_height and u != -1:
                    if vertices[i] < vertices[u]:
                        u = i
            if u == -1:
                break
            
            pushed = False
            for v in [n-1] + sorted(range(n), key=lambda x: vertices[x]):
                if v != u and residual[u][v] > 0 and height[u] == height[v] + 1:
                    push_flow = min(excess[u], residual[u][v])
                    flow[u][v] += push_flow
                    residual[u][v] -= push_flow
                    residual[v][u] += push_flow
                    excess[u] -= push_flow
                    excess[v] += push_flow
                    pushed = True
                    f.write(f"\nIteration {iteration}: Push {push_flow} from {vertices[u]} to {vertices[v]}\n")
                    f.write(f"Excess: {excess}\n")
                    f.write(f"Heights: {height}\n")
                    break
            
            if not pushed:
                min_height = float('inf')
                for v in range(n):
                    if residual[u][v] > 0:
                        min_height = min(min_height, height[v])
                if min_height != float('inf'):
                    height[u] = 1 + min_height
                    f.write(f"\nIteration {iteration}: Relabel {vertices[u]} to height {height[u]}\n")
                    f.write(f"Excess: {excess}\n")
                    f.write(f"Heights: {height}\n")
            
            iteration += 1
        
        max_flow = sum(flow[i][n-1] for i in range(n))
        f.write("\nFinal flow:\n")
        display_matrix(flow, "Final flow", vertices)
        for row in flow:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write(f"\nMaximum flow value: {max_flow}\n")
    
    return max_flow, flow

# Minimum-Cost Flow using Bellman-Ford
def min_cost_flow(n, capacity, cost, target_flow, trace_file):
    flow = np.zeros((n, n), dtype=int)
    residual = capacity.copy()
    total_flow = 0
    total_cost = 0
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Minimum-Cost Flow Execution Trace\n")
        display_matrix(capacity, "Capacity table", vertices)
        display_matrix(cost, "Cost table", vertices)
        f.write("\nCapacity table:\n")
        for row in capacity:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write("\nCost table:\n")
        for row in cost:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        
        iteration = 1
        while total_flow < target_flow:
            dist = [float('inf')] * n
            dist[0] = 0
            parent = [-1] * n
            bellman_table = [[float('inf')] * n for _ in range(n + 1)]
            bellman_table[0] = dist.copy()
            
            for k in range(1, n):
                for u in range(n):
                    bellman_table[k][u] = bellman_table[k-1][u]
                    for v in range(n):
                        if residual[u][v] > 0:
                            if bellman_table[k-1][v] != float('inf'):
                                new_dist = bellman_table[k-1][v] + cost[u][v]
                                if new_dist < bellman_table[k][u]:
                                    bellman_table[k][u] = new_dist
                                    parent[u] = v
                if bellman_table[k] == bellman_table[k-1]:
                    break
            
            f.write(f"\nBellman-Ford table for iteration {iteration}:\n")
            display_matrix(np.array(bellman_table[:k+1]), f"Bellman-Ford table iteration {iteration}", vertices)
            f.write("\n")
            for row in bellman_table[:k+1]:
                f.write(" ".join(f"{x:>4}" if x != float('inf') else "  inf" for x in row) + "\n")
            
            if parent[n-1] == -1:
                f.write("No augmenting path found.\n")
                break
            
            path = []
            v = n - 1
            while v != 0:
                u = parent[v]
                path.append((u, v))
                v = u
            path.reverse()
            
            path_flow = min(residual[u][v] for u, v in path)
            path_flow = min(path_flow, target_flow - total_flow)
            
            for u, v in path:
                flow[u][v] += path_flow
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                total_cost += path_flow * cost[u][v]
            
            total_flow += path_flow
            path_str = "".join(vertices[u] for u, _ in path) + vertices[path[-1][1]]
            f.write(f"\nAugmenting path: {path_str} with flow {path_flow}\n")
            f.write("Updated residual graph:\n")
            display_matrix(residual, f"Residual graph iteration {iteration}", vertices)
            for row in residual:
                f.write(" ".join(f"{x:>2}" for x in row) + "\n")
            
            iteration += 1
        
        f.write(f"\nFinal flow:\n")
        display_matrix(flow, "Final flow", vertices)
        for row in flow:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write(f"\nTotal flow: {total_flow}, Total cost: {total_cost}\n")
    
    return total_flow, total_cost, flow

# Function to generate random flow problem
def generate_random_flow_problem(n):
    capacity = np.zeros((n, n), dtype=int)
    cost = np.zeros((n, n), dtype=int)
    num_edges = int(n * n / 2)
    edges = random.sample([(i, j) for i in range(n) for j in range(n) if i != j], num_edges)
    
    for i, j in edges:
        capacity[i][j] = random.randint(1, 100)
        cost[i][j] = random.randint(1, 100)
    
    return capacity, cost

# Complexity analysis
def complexity_analysis():
    n_values = [10, 20, 40, 100, 400, 1000, 4000]
    ff_times = {n: [] for n in n_values}
    pr_times = {n: [] for n in n_values}
    min_times = {n: [] for n in n_values}
    
    for n in n_values:
        for _ in range(100):
            capacity, cost = generate_random_flow_problem(n)
            
            start = time.time()
            ford_fulkerson(n, capacity, "temp_ff.txt")
            ff_times[n].append(time.time() - start)
            
            start = time.time()
            push_relabel(n, capacity, "temp_pr.txt")
            pr_times[n].append(time.time() - start)
            
            max_flow, _ = ford_fulkerson(n, capacity, "temp_ff.txt")
            target_flow = max_flow // 2
            start = time.time()
            min_cost_flow(n, capacity, cost, target_flow, "temp_min.txt")
            min_times[n].append(time.time() - start)
    
    for algo, times in [("Ford-Fulkerson", ff_times), ("Push-Relabel", pr_times), ("Min-Cost", min_times)]:
        plt.figure()
        for n in n_values:
            plt.scatter([n] * 100, times[n], alpha=0.5)
        plt.xlabel('n (Number of vertices)')
        plt.ylabel('Execution time (seconds)')
        plt.title(f'{algo} Execution Times')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{algo.lower().replace(" ", "_")}_point_cloud.png')
        plt.close()
    
    for algo, times in [("Ford-Fulkerson", ff_times), ("Push-Relabel", pr_times), ("Min-Cost", min_times)]:
        max_times = [max(times[n]) for n in n_values]
        plt.plot(n_values, max_times, label=algo)
        
        if algo == "Ford-Fulkerson":
            expected = [n * (n ** 2) / 1e6 for n in n_values]
        elif algo == "Push-Relabel":
            expected = [(n ** 3) / 1e6 for n in n_values]
        else:
            expected = [(n ** 3 * np.log(n)) / 1e6 for n in n_values]
        plt.plot(n_values, expected, '--', label=f'{algo} expected')
    
    plt.xlabel('n')
    plt.ylabel('Max execution time (seconds)')
    plt.title('Worst-Case Complexity')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('worst_case_complexity.png')
    plt.close()
    
    ratios = [max(ff_times[n]) / max(pr_times[n]) for n in n_values]
    plt.plot(n_values, ratios, label='θ_FF / θ_PR')
    plt.xlabel('n')
    plt.ylabel('Ratio')
    plt.title('Ford-Fulkerson vs Push-Relabel')
    plt.xscale('log')
    plt.savefig('ff_vs_pr_ratio.png')
    plt.close()

# Main program
def main():
    while True:
        problem_num = input("Enter problem number (1-10, or 'exit' to quit): ")
        if problem_num.lower() == 'exit':
            break
        problem_num = int(problem_num)
        filename = f"proposal_{problem_num}.txt"
        
        n, capacity = read_flow_network(filename)
        vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
        
        print("Running Ford-Fulkerson...")
        max_flow_ff, flow_ff = ford_fulkerson(n, capacity, f"B4-trace{problem_num}-FF.txt")
        print(f"Ford-Fulkerson Max Flow: {max_flow_ff}")
        
        print("Running Push-Relabel...")
        max_flow_pr, flow_pr = push_relabel(n, capacity, f"B4-trace{problem_num}-PR.txt")
        print(f"Push-Relabel Max Flow: {max_flow_pr}")
        
        cost = np.random.randint(1, 10, size=(n, n)) * (capacity > 0)
        target_flow = max_flow_ff // 2
        print(f"Running Min-Cost Flow with target flow {target_flow}...")
        total_flow, total_cost, flow_min = min_cost_flow(n, capacity, cost, target_flow, f"B4-trace{problem_num}-MIN.txt")
        print(f"Min-Cost Flow: Total Flow = {total_flow}, Total Cost = {total_cost}")
        
        display_matrix(capacity, "Capacity Matrix", vertices)
        display_matrix(cost, "Cost Matrix", vertices)
        display_matrix(flow_ff, "Ford-Fulkerson Flow", vertices)
        display_matrix(flow_pr, "Push-Relabel Flow", vertices)
        display_matrix(flow_min, "Min-Cost Flow", vertices)
    
    print("Running complexity analysis...")
    complexity_analysis()

if __name__ == "__main__":
    main()