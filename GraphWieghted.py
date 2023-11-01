class Vertex:
    def __init__(self, vertex_id):
        self.id = vertex_id
        self.neighbors = {}  # Use a dictionary to store neighbors and their corresponding weights

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight
class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.id not in self.vertices:
            self.vertices[vertex.id] = vertex
            return True
        else:
            return False

    def add_edge(self, v1, v2, weight):
        if v1 in self.vertices and v2 in self.vertices:
            self.vertices[v1].add_neighbor(v2, weight)
            self.vertices[v2].add_neighbor(v1, weight)
            return True
        else:
            return False

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())
    def iterr(self):
        self.__iter__()

# Create a graph with 4 vertices and 5 weighted edges
graph = Graph()
for i in range(4):
    graph.add_vertex(Vertex(i))

# Add weighted edges
graph.add_edge(0, 1, 2)  # Edge between 0 and 1 with weight 2
graph.add_edge(0, 2, 3)  # Edge between 0 and 2 with weight 3
graph.add_edge(1, 2, 1)  # Edge between 1 and 2 with weight 1
graph.add_edge(1, 3, 4)  # Edge between 1 and 3 with weight 4
graph.add_edge(2, 3, 5)  # Edge between 2 and 3 with weight 5

print (graph.get_vertices())
print (graph.iterr())
for v in graph:
    print (v.id, " -> ",v.neighbors)
