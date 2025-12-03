import heapq

# --- 1. HELPER CLASSES (Edge and MinHeap/Priority Queue) ---

# Edge class to represent a weighted edge
class Edge:
    """Represents a weighted edge between two nodes."""
    def __init__(self, from_node, to, weight):
        self.from_node = from_node
        self.to = to
        self.weight = weight

    # Used for sorting edges in Kruskal's
    def __lt__(self, other):
        return self.weight < other.weight

# MinHeap (Priority Queue) implementation using heapq
class MinHeap:
    """A minimal Priority Queue wrapper for heapq."""
    def __init__(self):
        self.heap = []

    def push(self, distance, vertex):
        # Stores (distance, vertex) tuples. heapq maintains the min-heap property
        heapq.heappush(self.heap, (distance, vertex))

    def pop(self):
        if self.heap:
            # FIX: Correctly pass the internal list 'self.heap' to heappop()
            return heapq.heappop(self.heap)
        return None

    def empty(self):
        return len(self.heap) == 0

    # Required for heapq to work correctly on the class instance
    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

# --- 2. DISJOINT SET (UNION-FIND) FOR KRUSKAL'S ---

class DisjointSet:
    """Implements the Union-Find data structure for Kruskal's Algorithm."""
    def __init__(self, vertices):
        # Parent array: parent[i] stores the parent of element i
        self.parent = list(range(vertices))
        # Rank/Size array: helps keep the tree flat (union by rank/size)
        self.rank = [0] * vertices

    def find(self, i):
        """Finds the representative (root) of the set containing element i."""
        if self.parent[i] == i:
            return i
        # Path compression optimization
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Unites the set containing i and the set containing j."""
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank optimization
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True # Successfully united
        return False # Already in the same set

# --- 3. GRAPH CLASS (Dijkstra, Kruskal, Prim-Jarnik) ---

class Graph:
    def __init__(self, vertices, vertex_map):
        self.V = vertices
        # Adjacency list for Dijkstra's and Prim's
        self.adj = [[] for _ in range(vertices)]
        # List of all edges for Kruskal's
        self.all_edges = []
        # Map for clearer output (e.g., 0 -> 'A')
        self.v_map = vertex_map

    def add_edge(self, u_idx, v_idx, weight):
        """Adds an edge to the graph (undirected)."""
        # For adjacency list (Dijkstra/Prim)
        self.adj[u_idx].append(Edge(u_idx, v_idx, weight))
        self.adj[v_idx].append(Edge(v_idx, u_idx, weight))
        
        # For Kruskal's (only need to store once)
        self.all_edges.append(Edge(u_idx, v_idx, weight))

    def _get_vertex_name(self, index):
        """Helper to get the human-readable name from index."""
        return self.v_map.get(index, str(index))

    def print_adjacency_list(self):
        """Prints the initial graph representation."""
        print("--- Initial Graph Adjacency List ---")
        for u in range(self.V):
            u_name = self._get_vertex_name(u)
            neighbors = []
            for edge in self.adj[u]:
                v_name = self._get_vertex_name(edge.to)
                neighbors.append(f"({v_name}, {edge.weight})")
            print(f"Vertex {u_name}: {', '.join(neighbors)}")
        print("-" * 40)

    # --- DIJKSTRA'S ALGORITHM (Provided in Assignment Context) ---
    def dijkstra(self, src):
        """Finds the shortest paths from a source vertex."""
        src_name = self._get_vertex_name(src)
        print(f"\n--- Executing Dijkstra's Algorithm from Source {src_name} ---")

        dist = [float('inf')] * self.V
        pq = MinHeap()

        dist[src] = 0
        pq.push(0, src)
        
        step = 0
        while not pq.empty():
            step += 1
            u_dist, u = pq.pop()
            u_name = self._get_vertex_name(u)

            # Important: Check if the popped distance is still the shortest known one
            # This handles outdated entries in the priority queue
            if u_dist > dist[u]:
                continue
            
            print(f"Step {step}: Extracting vertex {u_name} with distance {u_dist}")

            for edge in self.adj[u]:
                v, weight = edge.to, edge.weight
                v_name = self._get_vertex_name(v)

                if dist[u] + weight < dist[v]:
                    # Relaxation step
                    dist[v] = dist[u] + weight
                    pq.push(dist[v], v)
                    print(f"    Relax: {u_name} -> {v_name}. New dist[{v_name}] = {dist[v]}")
            
            # Intermediate step output
            current_dists = {}
            for i in range(self.V):
                 d = dist[i]
                 current_dists[self._get_vertex_name(i)] = "INF" if d == float('inf') else d

            # The original code only printed this step inside the loop, so we'll keep it simple
            # print("Current shortest distances:", current_dists)


        print("\nFinal shortest distances from source", src_name, ":")
        final_dists = {}
        for i in range(self.V):
            d = dist[i]
            final_dists[self._get_vertex_name(i)] = 'INF' if d == float('inf') else d
        print(final_dists)
        print("-" * 40)


    # --- KRUSKAL'S ALGORITHM ---
    def kruskal_mst(self):
        """Finds the Minimum Spanning Tree using Kruskal's Algorithm (Union-Find)."""
        print("\n--- Executing Kruskal's MST Algorithm ---")

        mst = []
        # 1. Sort all edges by weight
        sorted_edges = sorted(self.all_edges, key=lambda edge: edge.weight)
        
        # 2. Initialize Disjoint Set
        ds = DisjointSet(self.V)
        
        mst_cost = 0
        edges_processed = 0

        print(f"Total Edges: {len(sorted_edges)}. Target MST Edges: {self.V - 1}")
        print("Steps: Edge (u-v, weight) -> Action")
        
        # 3. Iterate through sorted edges
        for edge in sorted_edges:
            u, v, weight = edge.from_node, edge.to, edge.weight
            u_name = self._get_vertex_name(u)
            v_name = self._get_vertex_name(v)
            
            edges_processed += 1
            
            # 4. Check if adding the edge creates a cycle
            if ds.union(u, v):
                # No cycle, add edge to MST
                mst.append(edge)
                mst_cost += weight
                print(f"  {edges_processed}. Edge ({u_name}-{v_name}, {weight}) -> ACCEPTED (MST Edge Count: {len(mst)})")
                
                if len(mst) == self.V - 1:
                    break
            else:
                # Cycle detected, reject edge
                print(f"  {edges_processed}. Edge ({u_name}-{v_name}, {weight}) -> REJECTED (Creates Cycle)")
        
        print("\n--- Final Kruskal's MST ---")
        print(f"Total Cost: {mst_cost}")
        print("Edges in MST:")
        for edge in mst:
            u_name = self._get_vertex_name(edge.from_node)
            v_name = self._get_vertex_name(edge.to)
            print(f"  {u_name} --- {edge.weight} --- {v_name}")
        print("-" * 40)


    # --- PRIM-JARNIK'S ALGORITHM ---
    def prim_jarnik_mst(self, start_node=0):
        """Finds the Minimum Spanning Tree using Prim-Jarnik's Algorithm (Min-Heap)."""
        start_name = self._get_vertex_name(start_node)
        print(f"\n--- Executing Prim-Jarnik's MST Algorithm (Start: {start_name}) ---")

        # Key stores the minimum weight connecting a vertex to the MST
        key = [float('inf')] * self.V
        # Parent stores the MST parent of the vertex
        parent = [-1] * self.V
        # Boolean to track if a vertex is already included in the MST
        in_mst = [False] * self.V
        
        # Priority Queue stores (weight, vertex)
        pq = MinHeap()

        # Start with the source vertex
        key[start_node] = 0
        pq.push(0, start_node)
        
        mst_cost = 0
        edges_in_mst = 0
        
        step = 0
        while not pq.empty() and edges_in_mst < self.V:
            step += 1
            weight_u, u = pq.pop()
            u_name = self._get_vertex_name(u)
            
            # Check if already added (handles stale entries in PQ)
            if in_mst[u]:
                continue
            
            # Add u to MST
            in_mst[u] = True
            
            # Update cost and edge count (if not the starting node)
            if parent[u] != -1:
                mst_cost += weight_u
                edges_in_mst += 1
                parent_name = self._get_vertex_name(parent[u])
                print(f"Step {step}: Adding Edge ({parent_name}-{u_name}, {weight_u}). MST Cost: {mst_cost}")
            else:
                print(f"Step {step}: Initializing with Start Node {u_name}.")

            # Check all neighbors v of u
            for edge in self.adj[u]:
                v, weight = edge.to, edge.weight
                v_name = self._get_vertex_name(v)
                
                # If v is not in MST and edge weight is less than current key[v]
                if not in_mst[v] and weight < key[v]:
                    # Update key and parent
                    key[v] = weight
                    parent[v] = u
                    # Add to Priority Queue
                    pq.push(key[v], v)
                    print(f"    Relax: {u_name} -> {v_name}. New Key[{v_name}] = {weight}")

        print("\n--- Final Prim-Jarnik's MST ---")
        print(f"Total Cost: {mst_cost}")
        print("Edges in MST:")
        # Skip the starting node (u=0) as it has no parent
        for u in range(1, self.V):
            u_name = self._get_vertex_name(u)
            p_name = self._get_vertex_name(parent[u])
            w = key[u]
            print(f"  {p_name} --- {w} --- {u_name}")
        print("-" * 40)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Graph data setup for 9 vertices (A-I)
    V = 9
    vertex_map = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I'
    }

    # Initialize graph
    g = Graph(V, vertex_map)

    # Add edges from the assignment image (A=0, B=1, ..., I=8)
    edges_list = [
        # (U, V, WEIGHT)
        (0, 1, 144), (0, 2, 184), (0, 3, 187), (0, 6, 849), (0, 8, 1258),
        (1, 3, 867),
        (2, 6, 621), (2, 7, 1391), (2, 8, 1090),
        (3, 6, 740),
        (4, 5, 337), (4, 7, 1235), (4, 8, 2342),
        (5, 6, 1846), (5, 7, 1464), (5, 8, 2704),
        (6, 7, 802),
        (7, 8, 1121)
    ]

    for u, v, weight in edges_list:
        g.add_edge(u, v, weight)

    # 1. Print Initial Graph Representation (Requirement 4)
    g.print_adjacency_list()

    # 2. Execute and show steps for Kruskal's MST (Requirement 2 & 4)
    g.kruskal_mst()

    # 3. Execute and show steps for Prim-Jarnik's MST (Requirement 3 & 4)
    # Start node A (index 0) for consistency
    g.prim_jarnik_mst(start_node=0)
    
    # Optional: Run Dijkstra's for completeness (Source A)
    # g.dijkstra(src=0)