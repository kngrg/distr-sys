#include <stdio.h>
#include <mpi.h>

#include <stdlib.h>
#include <time.h>

#define SIGN(x) (2 * (x) - 1)
#define CLOSENESS_TO(x, center) (center - abs(center - x))

enum {
    SIDE = 5,
    CENTER = SIDE / 2,
    MAX_NEIGHBOURS = 4,
};


void init_values(int* values, int* max, int timestamp) {
    srand(timestamp);

    *max = -1;
    for (int i = 0; i < SIDE * SIDE; i++) {
        values[i] = rand()%100;
        if (values[i] > *max) {
            *max = values[i];
        }
    }
}

void print_matrix(int* values) {
    printf("Initialized matrix:\n");
    for (int i = 0; i < SIDE; i++) {
        for (int j = 0; j < SIDE; j++) {
            printf("%3d ", values[i * SIDE + j]);
        }
        printf("\n");
    }
}

void reduce(int rank, int val, int ans) {
    MPI_Status statuses[MAX_NEIGHBOURS];
    int tmp;
    int x = rank / SIDE;
    int y = rank % SIDE;

    int num_to_receive = CLOSENESS_TO(x, CENTER) + (x == CENTER) * CLOSENESS_TO(y, CENTER);

    for (int i = 0; i < num_to_receive; ++i) {
        MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &statuses[i]);
        val = tmp > val ? tmp : val;
    }

    MPI_Request request;
    int dest = x == CENTER ? rank + SIGN(y < CENTER) : rank + SIDE * SIGN(x < CENTER);

    if (!(x == CENTER && y == CENTER)) {
        MPI_Isend(&val, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &request);
        MPI_Recv(&val, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &(MPI_Status){});
    }

    for (int i = 0; i < num_to_receive; ++i) {
        MPI_Isend(&val, 1, MPI_INT, statuses[i].MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
    }

    if (val != ans) {
        printf("Rank %d - ERROR! - value = %d, ans = %d\n", rank, val, ans);
    } 

}



int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int timestamp;
    if (rank == 0) {
        timestamp = time(NULL);
    } 

    MPI_Bcast(&timestamp, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int values[SIDE * SIDE];
    int max;

    init_values(values, &max, timestamp);

    if (rank == 0) {
        print_matrix(values);
    }

    reduce(rank, values[rank], max);

    printf("Maximum value found by %d: %d\n", rank, max);
   
    MPI_Finalize();
    return 0;
}