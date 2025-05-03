import numpy as np

# Push-Relabel algorithm
def push_relabel(n, capacity, trace_file):
    # INITIALIZE
    flow = np.zeros((n, n), dtype=int)
    excess = [0] * n  # e[u]
    height = [0] * n  # h[u]
    height[0] = n     # h[s] = |V|
    residual = capacity.copy()  # c_f
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Push-Relabel Execution Trace\n")
        f.write("\nInitial capacity table:\n")
        for row in capacity:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        
        # INITIALIZE (lignes 6-9)
        for v in range(n):
            if capacity[0][v] > 0:  # (s,u) ∈ E
                flow[0][v] = capacity[0][v]  # f[s,u] = c[s,u]
                excess[v] = capacity[0][v]   # e[u] = c[s,u]
                excess[0] -= capacity[0][v]  # e[s] = e[s] - c[s,u]
        
        f.write("\nAfter initialization:\n")
        f.write(f"Excess: {excess}\n")
        f.write(f"Heights: {height}\n")
        
        iteration = 1
        while True:
            # Chercher un nœud avec excès positif
            u = -1
            for i in range(1, n-1):
                if excess[i] > 0:
                    u = i
                    break
            
            if u == -1:
                break
            
            # Essayer PUSH
            pushed = False
            for v in range(n):
                # PUSH conditions: e[u] > 0, c_f[u,v] > 0, h[u] - h[v] = 1
                if (excess[u] > 0 and residual[u][v] > 0 and height[u] - height[v] == 1):
                    # PUSH operation
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
            
            # Si PUSH impossible, faire RELABEL
            if not pushed:
                # RELABEL conditions: e[u] > 0 et h[u] - h[v] < 1 pour tout (u,v) ∈ E_f
                min_height = float('inf')
                for v in range(n):
                    if residual[u][v] > 0:
                        min_height = min(min_height, height[v])
                if min_height != float('inf'):
                    height[u] = 1 + min_height
                    f.write(f"\nIteration {iteration}: Relabel {vertices[u]} to height {height[u]}\n")
                    f.write(f"Excess: {excess}\n")
                    f.write(f"Heights: {height}\n")
                else:
                    # Si on ne peut ni PUSH ni RELABEL, on passe au nœud suivant
                    break
            
            iteration += 1
            
            # Condition de sécurité pour éviter une boucle infinie
            if iteration > n * n:
                break
        
        max_flow = sum(flow[i][n-1] for i in range(n))
        f.write("\nFinal flow:\n")
        for row in flow:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write(f"\nMaximum flow value: {max_flow}\n")
    
    # Création du flow_display
    flow_display = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if capacity[i][j] > 0:
                flow_display[i][j] = f"{flow[i][j]}/{capacity[i][j]}"
            else:
                flow_display[i][j] = "0"
    
    return max_flow, flow, flow_display