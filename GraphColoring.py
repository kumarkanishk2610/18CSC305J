class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
    def greedy_coloring(self):
        result = [-1] * self.V  
        result[0] = 0
        available = [False] * self.V
        for u in range(1, self.V):
            for v in self.graph[u]:
                if result[v] != -1:
                    available[result[v]] = True
            for color in range(self.V):
                if not available[color]:
                    break
            result[u] = color
            for v in self.graph[u]:
                if result[v] != -1:
                    available[result[v]] = False
        for u in range(self.V):
            print(f"Vertex {u} --> Color {result[u]}")
if __name__ == "__main__":
    graph = Graph(5)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    print("Coloring of vertices:")
    graph.greedy_coloring()