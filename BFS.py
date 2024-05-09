from collections import defaultdict

def addEdge(graph, u, v):
    graph[u].append(v)

def BFS(graph, s):
    visited = [False] * (max(graph) + 1)
    queue = []
    queue.append(s)
    visited[s] = True
    while queue:
        s = queue.pop(0)
        print(s, end=" ")
        for i in graph[s]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

if __name__ == '__main__':
    graph = defaultdict(list)
    addEdge(graph, 0, 1)
    addEdge(graph, 0, 2)
    addEdge(graph, 1, 2)
    addEdge(graph, 2, 0)
    addEdge(graph, 2, 3)
    addEdge(graph, 3, 3)
    print("Following is Breadth First Traversal (starting from vertex 2)")
    BFS(graph, 2)
