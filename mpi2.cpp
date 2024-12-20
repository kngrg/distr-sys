#include <mpi.h>
#include <mpi-ext.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <ctime> 
#include <signal.h>
#include <iostream>

using namespace std;

#define N (2 * 2 * 2 * 2 * 2 * 2 + 2)  
#define checkpoint_freq 10         
#define N2 N*N
#define A(i, j, k) A[(i) * N2 + (j) * N + (k)]
#define Max(a,b) ((a)>(b)?(a):(b))
#define FILE_NAME "checkpoint.bin"    

int myrank, ranksize;                 
int itmax = 100;                      
int it = 1;          
int failure_iteration = 23;                  
double eps = 0.0, maxeps = 0.1e-7;    
double w = 0.5;                       
double s_global;
bool killed = false;
bool checkpoint_exsits = false;
int nrow, startrow;                   
MPI_Comm communicator;                
MPI_Errhandler errh;                 
vector<double> A;                     


void relax();
void verify();
void save_checkpoint();
void load_checkpoint();
void error_handler(MPI_Comm* pcomm, int* error_code, ...);
void calculate_bounds(int rank, int size, int& startrow, int& nrow);

void initialize() {
    MPI_Comm_rank(communicator, &myrank);
    MPI_Comm_size(communicator, &ranksize);

    calculate_bounds(myrank, ranksize, startrow, nrow);
    A.resize((nrow + 2) * N2, 0.0);

    for (int i = 1; i <= nrow; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (i + startrow == 0 || i + startrow == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    A(i, j, k) = 0.0;
                } else {
                    A(i, j, k) = 4.0 + i + j + k + startrow;
                }
            }
        }
    }
}

 void relax() {
    MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    if (myrank != 0) {
        MPI_Isend(&A(1, 0, 0), N2, MPI_DOUBLE, myrank - 1, 0, communicator, &reqs[0]);
        MPI_Irecv(&A(0, 0, 0), N2, MPI_DOUBLE, myrank - 1, 0, communicator, &reqs[1]);
    }
    if (myrank != ranksize - 1) {
        MPI_Isend(&A(nrow, 0, 0), N2, MPI_DOUBLE, myrank + 1, 0, communicator, &reqs[2]);
        MPI_Irecv(&A(nrow + 1, 0, 0), N2, MPI_DOUBLE, myrank + 1, 0, communicator, &reqs[3]);
    }

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    double local_eps = eps;
    for (int i = 1; i <= nrow; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            for (int k = 1 + (i + j + startrow) % 2; k < N - 1; k+=2) {
                double b = w * ((A(i - 1, j, k) + A(i + 1, j, k) +
                                A(i, j - 1, k) + A(i, j + 1, k) +
                                A(i, j, k - 1) + A(i, j, k + 1)) / 6.0 -
                                A(i, j, k));
                A(i, j,k ) += b;
                local_eps = Max(local_eps, fabs(b));
            }
        }
    }


    if (myrank != 0) {
        MPI_Isend(&A(1, 0, 0), N2, MPI_DOUBLE, myrank - 1, 0, communicator, &reqs[0]);
        MPI_Irecv(&A(0, 0, 0), N2, MPI_DOUBLE, myrank - 1, 0, communicator, &reqs[1]);
    }
    if (myrank != ranksize - 1) {
        MPI_Isend(&A(nrow, 0, 0), N2, MPI_DOUBLE, myrank + 1, 0, communicator, &reqs[2]);
        MPI_Irecv(&A(nrow + 1, 0, 0), N2, MPI_DOUBLE, myrank + 1, 0, communicator, &reqs[3]);
    }

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    for (int i = 1; i <= nrow; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            for (int k = 1 + (i + j + 1 + startrow) % 2 ; k < N - 1; k+=2) {
                double b = w * ((A(i - 1, j, k) + A(i + 1, j, k) +
                                A(i, j - 1, k) + A(i, j + 1, k) +
                                A(i, j, k - 1) + A(i, j, k + 1)) /
                                6.0 -
                                A(i, j, k));
                A(i, j, k) += b;
                local_eps = Max(local_eps, fabs(b));
            }
        }
    }

    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, communicator);
}

void verify() {
    double s_local = 0.0;
    for (int i = 1; i <= nrow; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                s_local += A(i, j, k) * (startrow + i + 1) * (j + 1) * (k + 1) / (N2 * N);
            }
        }
    }
    MPI_Allreduce(&s_local, &s_global, 1, MPI_DOUBLE, MPI_SUM, communicator);
    if (myrank == 0) {
        printf("S = %lf\n", s_global);
    }
}

void save_checkpoint() {
    MPI_File fh;
    checkpoint_exsits = true;

    MPI_File_open(communicator, FILE_NAME, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (myrank == 0) {
        MPI_File_write_at(fh, 0, &it, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Offset offset = sizeof(int) + startrow * N2 * sizeof(double);
    MPI_File_write_at(fh, offset, &A.data()[N2], nrow * N2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void load_checkpoint() {
    MPI_File fh;

    MPI_File_open(communicator, FILE_NAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, 0, &it, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Offset offset = sizeof(int) + startrow * N2 * sizeof(double);
    MPI_File_read_at(fh, offset, &A.data()[N2], nrow * N2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    printf("Process %d: Loaded checkpoint (it = %d, startrow = %d, nrow = %d)\n",
           myrank, it, startrow, nrow);
    fflush(stdout);
}


void error_handler(MPI_Comm* pcomm, int* error_code, ...) {
    printf("Process %d: Starting recovery\n", myrank);
    fflush(stdout);

    MPIX_Comm_revoke(*pcomm); 
    MPIX_Comm_shrink(*pcomm, &communicator);
    MPI_Comm_set_errhandler(communicator, errh);

    MPI_Comm_rank(communicator, &myrank);
    MPI_Comm_size(communicator, &ranksize);
    printf("New communicator info: myrank = %d size = %d\n", myrank, ranksize);
    fflush(stdout);

    calculate_bounds(myrank, ranksize, startrow, nrow);
    A.resize((nrow + 2) * N2, 0.0);

    if (checkpoint_exsits) { 
        load_checkpoint();
        it++;
        printf("Process %d: Checkpoint loaded\n", myrank);
        fflush(stdout);
    } else {
        printf("Process %d: Failed to load checkpoint, reinitializing\n", myrank);
        fflush(stdout);
        initialize();
        it = 1;
    }

    printf("Process %d: Recovery complete\n", myrank);
    fflush(stdout);
}

void calculate_bounds(int rank, int size, int& startrow, int& nrow) {
    startrow = rank * (N - 2) / size;
    int endrow = (rank + 1) * (N - 2) / size;
    nrow = endrow - startrow;
}

void simulate_failure() {
    if ((myrank != 0 && myrank % 2 == 0) && !killed) { 
        printf("Process %d is failing intentionally...\n", myrank);
        fflush(stdout);

        raise(SIGKILL);
    }
    killed = true; 
    MPI_Barrier(communicator);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    communicator = MPI_COMM_WORLD;

    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(communicator, errh);

    if (myrank == 0) {
        if (argc > 1) {
            failure_iteration = atoi(argv[1]);
            if (failure_iteration <= 0 || failure_iteration > 100) {
                failure_iteration = 23;
            }
        }
    }

    initialize();

    while (it <= itmax) {
        eps = 0.0;

        if (it == failure_iteration && !killed) { 
            simulate_failure();
        }

        printf("Process %d: Performing relaxation step %d\n", myrank, it);
        fflush(stdout);

        relax();

        if (eps < maxeps) {
            printf("Process %d: Converged after %d iterations\n", myrank, it);
            fflush(stdout);
            break;
        }
        if (it % checkpoint_freq == 0) {
            printf("Process %d: Storing checkpoint at iteration %d\n", myrank, it);
            fflush(stdout);
            save_checkpoint();
        }
        it++;
    }

    verify();
    fflush(stdout);

    if (myrank == 0) {
        printf("All processes finished successfully\n");
        if (remove(FILE_NAME) != 0) {
            perror("Error deleting the checkpoint file");
        } else {
            printf("Checkpoint file deleted successfully\n");
        }
    }

    MPI_Finalize();
    return 0;
}