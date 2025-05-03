import numpy as np
from collections import deque

# Ford-Fulkerson with Edmonds-Karp (BFS for shortest augmenting paths)
def ford_fulkerson(n, capacity, trace_file):
    flow = np.zeros((n, n), dtype=int)
    residual = capacity.copy()
    max_flow = 0
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
    residual_matrices = []  # Liste pour stocker les matrices résiduelles

    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Ford-Fulkerson Execution Trace\n")
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
            residual_matrices.append(residual.copy())  # Ajout de la matrice résiduelle à la liste
            path_str = "".join(vertices[u] for u, _ in path) + vertices[path[-1][1]]
            f.write(f"Detection of an improving chain: {path_str} with a flow {path_flow}.\n")
            f.write("\nModifications to the residual graph:\n")
            # display_matrix(residual, f"Residual graph after iteration {iteration}", vertices)
            for row in residual:
                f.write(" ".join(f"{x:>2}" for x in row) + "\n")
            
            iteration += 1
        
        f.write("\n* Max flow display:\n")
        flow_display = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                flow_display[i][j] = f"{flow[i][j]}/{capacity[i][j]}" if capacity[i][j] > 0 else "0"
        for row in flow_display:
            f.write(" ".join(f"{x:>5}" for x in row) + "\n")
        f.write(f"\nValue of the max flow = {max_flow}\n")
    
    return residual_matrices, flow_display, max_flow, flow