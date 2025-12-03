#include <iostream>
#include <vector>
#include <limits>

using namespace std;

// Edge class to represent a weighted edge
class Edge {
public:
    int from, to, weight;

    Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
};

// Min Heap implementation for Dijkstra’s priority queue
class MinHeap {
private:
    vector<pair<int, int>> heap; // Stores (distance, vertex) pairs
    int size;

    // Function to swap two heap elements
    void swap(int i, int j) {
        pair<int, int> temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }

    // Heapify up
    void heapifyUp(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (heap[index].first < heap[parent].first) {
                swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    // Heapify down
    void heapifyDown(int index) {
        int left, right, smallest;
        while (true) {
            left = 2 * index + 1;
            right = 2 * index + 2;
            smallest = index;

            if (left < size && heap[left].first < heap[smallest].first)
                smallest = left;
            if (right < size && heap[right].first < heap[smallest].first)
                smallest = right;

            if (smallest != index) {
                swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

public:
    MinHeap() : size(0) {}

    // Insert a new element into the heap
    void push(int distance, int vertex) {
        heap.push_back({distance, vertex});
        size++;
        heapifyUp(size - 1);
    }

    // Extract the minimum element
    pair<int, int> pop() {
        if (size == 0) return {-1, -1}; // Error case
        pair<int, int> minElement = heap[0];
        heap[0] = heap[size - 1];
        heap.pop_back();
        size--;
        heapifyDown(0);
        return minElement;
    }

    // Check if heap is empty
    bool empty() {
        return size == 0;
    }
};

// Graph class
class Graph {
private:
    int V; // Number of vertices
    vector<vector<Edge>> adj; // Adjacency list

public:
    // Constructor
    Graph(int vertices) : V(vertices), adj(vertices) {}

    // Function to add an edge
    void addEdge(int from, int to, int weight) {
        adj[from].push_back(Edge(from, to, weight));
        adj[to].push_back(Edge(to, from, weight)); // Since the graph is undirected
    }

    // Dijkstra’s algorithm implementation using a Min Heap
    void dijkstra(int src) {
        vector<int> dist(V, numeric_limits<int>::max()); // Distance array
        MinHeap pq; // Min Heap for priority queue

        // Set distance of source to 0 and push into heap
        dist[src] = 0;
        pq.push(0, src);

        while (!pq.empty()) {
            pair<int, int> current = pq.pop();
            int u = current.second;
            int uDist = current.first;

            // Process all adjacent vertices
            for (const auto& edge : adj[u]) {
                int v = edge.to;
                int weight = edge.weight;

                // If a shorter path is found
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push(dist[v], v);
                }
            }

            // Print the step-by-step execution
            cout << "Current shortest distances: ";
            for (int d : dist) {
                if (d == numeric_limits<int>::max()) cout << "INF ";
                else cout << d << " ";
            }
            cout << endl;
        }

        // Print final shortest distances
        cout << "\nFinal shortest distances from source " << src << ":\n";
        for (int i = 0; i < V; i++) {
            cout << "Vertex " << i << " : " << (dist[i] == numeric_limits<int>::max() ? "INF" : to_string(dist[i])) << endl;
        }
    }
};

// Main function to test the implementation
int main() {
    // Create a graph with 6 vertices
    Graph g(6);

    // Adding edges to the graph
    g.addEdge(0, 1, 4);
    g.addEdge(0, 2, 4);
    g.addEdge(1, 2, 2);
    g.addEdge(1, 3, 5);
    g.addEdge(2, 3, 1);
    g.addEdge(2, 4, 3);
    g.addEdge(3, 5, 2);
    g.addEdge(4, 5, 6);

    // Run Dijkstra's algorithm from source vertex 0
    cout << "Executing Dijkstra's Algorithm using Min Heap...\n";
    g.dijkstra(0);

    return 0;
}
