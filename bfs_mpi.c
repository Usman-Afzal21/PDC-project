#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
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

void bfs_mpi(Graph *graph, int start_vertex, int *visited, int rank, int size) {
    int num_vertices = graph->num_vertices;
    int *queue = malloc(num_vertices * sizeof(int));
    int front = 0, rear = 0;

    if (rank == 0) {
        queue[rear++] = start_vertex;
        visited[start_vertex] = 1;
    }

    while (1) {
        int local_count = 0;
        int *local_queue = malloc(num_vertices * sizeof(int));

        // Broadcast the queue to all processes
        MPI_Bcast(queue, num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rear, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Exit if queue is empty
        if (rear == front) {
            free(local_queue);
            break;
        }

        // Process assigned portion of the queue
        for (int i = front + rank; i < rear; i += size) {
            int current = queue[i];
            Node *temp = graph->adj_lists[current];

            while (temp) {
                int adj_vertex = temp->vertex;
                if (!visited[adj_vertex]) {
                    visited[adj_vertex] = 1;
                    local_queue[local_count++] = adj_vertex;
                }
                temp = temp->next;
            }
        }

        // Gather local queues at rank 0
        int *counts = malloc(size * sizeof(int));
        int *displs = malloc(size * sizeof(int));
        MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int total_count = 0;
            for (int i = 0; i < size; i++) {
                displs[i] = total_count;
                total_count += counts[i];
            }

            MPI_Gatherv(local_queue, local_count, MPI_INT,
                        &queue[rear], counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
            rear += total_count;
        } else {
            MPI_Gatherv(local_queue, local_count, MPI_INT,
                        NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
        }

        free(local_queue);
        free(counts);
        free(displs);

        // Broadcast updated queue and rear
        MPI_Bcast(queue, num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rear, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(queue);
}

int main(int argc, char *argv[]) {
    int edge_counts[] = {1000, 200000, 1000000};  // Edge counts to test
    int num_vertices = 100000;  // Adjust number of vertices as needed
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < 3; i++) {
        int edges = edge_counts[i];
        Graph *graph = create_graph(num_vertices);
        int *visited = calloc(num_vertices, sizeof(int));

        if (rank == 0) {
            printf("Running for %d edges...\n", edges);
            generate_random_graph(graph, edges);
        }

        double start_time = MPI_Wtime();
        bfs_mpi(graph, 0, visited, rank, size);
        double end_time = MPI_Wtime();

        if (rank == 0) {
            printf("Parallel BFS (MPI): %d edges, %f seconds\n", edges, end_time - start_time);
        }

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

    MPI_Finalize();
    return 0;
}