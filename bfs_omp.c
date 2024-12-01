#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef struct Node {
    int vertex;
    struct Node *next;
} Node;

typedef struct Graph {
    int num_vertices;
    Node **adj_lists;
} Graph;

Node *create_node(int vertex) {
    Node *new_node = malloc(sizeof(Node));
    if (!new_node) {
        fprintf(stderr, "Failed to allocate memory for a Node\n");
        exit(EXIT_FAILURE);
    }
    new_node->vertex = vertex;
    new_node->next = NULL;
    return new_node;
}

Graph *create_graph(int num_vertices) {
    Graph *graph = malloc(sizeof(Graph));
    if (!graph) {
        fprintf(stderr, "Failed to allocate memory for Graph\n");
        exit(EXIT_FAILURE);
    }
    graph->num_vertices = num_vertices;
    graph->adj_lists = malloc(num_vertices * sizeof(Node *));
    if (!graph->adj_lists) {
        free(graph);
        fprintf(stderr, "Failed to allocate memory for adjacency lists\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_vertices; i++) {
        graph->adj_lists[i] = NULL;
    }
    return graph;
}

void add_edge(Graph *graph, int src, int dest) {
    Node *new_node = create_node(dest);
    new_node->next = graph->adj_lists[src];
    graph->adj_lists[src] = new_node;

    new_node = create_node(src);
    new_node->next = graph->adj_lists[dest];
    graph->adj_lists[dest] = new_node;
}

void generate_random_graph(Graph *graph, int num_edges) {
    int num_vertices = graph->num_vertices;
    srand(time(NULL));
    for (int i = 0; i < num_edges; i++) {
        int src = rand() % num_vertices;
        int dest;
        do {
            dest = rand() % num_vertices;
        } while (src == dest || dest == src);  // Avoid self-loops and duplicate edges
        add_edge(graph, src, dest);
    }
}

void bfs_parallel(Graph *graph, int start_vertex, int *visited) {
    int *queue = malloc(graph->num_vertices * sizeof(int));
    int front = 0, rear = 0;

    queue[rear++] = start_vertex;
    visited[start_vertex] = 1;

    #pragma omp parallel
    {
        while (front < rear) {
            #pragma omp for schedule(dynamic, 100) nowait
            for (int q = front; q < rear; q++) {
                int current = queue[q];
                Node *temp = graph->adj_lists[current];

                while (temp) {
                    int adj_vertex = temp->vertex;
                    if (!visited[adj_vertex]) {
                        #pragma omp critical
                        {
                            if (!visited[adj_vertex]) {
                                visited[adj_vertex] = 1;
                                queue[rear++] = adj_vertex;
                            }
                        }
                    }
                    temp = temp->next;
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                front = rear;
            }
        }
    }

    free(queue);
}

int main() {
    int edge_counts[] = {100000, 200000, 1000000};  // Edge counts to test
    int num_vertices = 100000;  // Adjust number of vertices as needed

    for (int i = 0; i < 3; i++) {
        int edges = edge_counts[i];
        Graph *graph = create_graph(num_vertices);
        int *visited = calloc(num_vertices, sizeof(int));

        printf("Running for %d edges...\n", edges);
        generate_random_graph(graph, edges);

        double start_time = omp_get_wtime();
        bfs_parallel(graph, 0, visited);
        double end_time = omp_get_wtime();
        printf("Parallel BFS (OpenMP): %d edges, %f seconds\n", edges, end_time - start_time);

        free(visited);
        for (int i = 0; i < num_vertices; i++) {
            Node *node = graph->adj_lists[i];
            while (node != NULL) {
                Node *temp = node;
                node = node->next;
                free(temp);
            }
        }
        free(graph->adj_lists);
        free(graph);
    }

    return 0;
}