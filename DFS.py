from collections import defaultdict
def addEdge(graph, u, v):
    graph[u].append(v)
def DFSUtil(graph, v, visited):
    visited.add(v)
    print(v, end=' ')
    for neighbour in graph[v]:
        if neighbour not in visited:
            DFSUtil(graph, neighbour, visited)
def DFS(graph, v):
    visited = set()
    DFSUtil(graph, v, visited)
graph = defaultdict(list)
addEdge(graph, 0, 1)
addEdge(graph, 0, 2)
addEdge(graph, 1, 2)
addEdge(graph, 2, 0)
addEdge(graph, 2, 3)
addEdge(graph, 3, 3)
print("Following is Depth First Traversal (starting from vertex 2)")
DFS(graph, 2)