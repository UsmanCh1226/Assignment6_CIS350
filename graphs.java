package graphs.testing;

import java.util.*;

// Edge class to represent a weighted edge
class Edge {
    int from, to, weight;

    public Edge(int from, int to, int weight) {
        this.from = from;
        this.to = to;
        this.weight = weight;
    }
}

// MinHeap (Priority Queue) using Java's PriorityQueue
class MinHeap {
    private PriorityQueue<int[]> heap;

    public MinHeap() {
        heap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
    }

    public void push(int distance, int vertex) {
        heap.add(new int[]{distance, vertex});
    }

    public int[] pop() {
        return heap.poll();
    }

    public boolean isEmpty() {
        return heap.isEmpty();
    }
}

// Graph class
class Graph {
    private int V;
    private List<List<Edge>> adj;

    public Graph(int vertices) {
        this.V = vertices;
        adj = new ArrayList<>();
        for (int i = 0; i < vertices; i++)
            adj.add(new ArrayList<>());
    }

    public void addEdge(int from, int to, int weight) {
        adj.get(from).add(new Edge(from, to, weight));
        adj.get(to).add(new Edge(to, from, weight)); // Undirected graph
    }

    public void dijkstra(int src) {
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        MinHeap pq = new MinHeap();

        dist[src] = 0;
        pq.push(0, src);

        while (!pq.isEmpty()) {
            int[] current = pq.pop();
            int u = current[1];
            int uDist = current[0];

            for (Edge edge : adj.get(u)) {
                int v = edge.to, weight = edge.weight;

                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push(dist[v], v);
                }
            }

            System.out.println("Current shortest distances: " + Arrays.toString(dist));
        }

        System.out.println("\nFinal shortest distances from source " + src + ":");
        for (int i = 0; i < V; i++) {
            System.out.println("Vertex " + i + " : " + (dist[i] == Integer.MAX_VALUE ? "INF" : dist[i]));
        }
    }

    public static void main(String[] args) {
        Graph g = new Graph(6);
        g.addEdge(0, 1, 4);
        g.addEdge(0, 2, 4);
        g.addEdge(1, 2, 2);
        g.addEdge(1, 3, 5);
        g.addEdge(2, 3, 1);
        g.addEdge(2, 4, 3);
        g.addEdge(3, 5, 2);
        g.addEdge(4, 5, 6);

        System.out.println("Executing Dijkstra's Algorithm using Min Heap...");
        g.dijkstra(0);
    }
}
