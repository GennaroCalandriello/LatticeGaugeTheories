/****************************************************************************
 * FILE: mpi_heat2D.cpp
 * OTHER FILES: draw_heat.cpp  
 * DESCRIPTIONS:  
 *   HEAT2D Example - Parallelized C++ Version
 *   This example is based on a simplified two-dimensional heat 
 *   equation domain decomposition.  The initial temperature is computed to be 
 *   high in the middle of the domain and zero at the boundaries.  The 
 *   boundaries are held at zero throughout the simulation.  During the 
 *   time-stepping, an array containing two domains is used; these domains 
 *   alternate between old data and new data.
 *
 *   In this parallelized version, the grid is decomposed by the master
 *   process and then distributed by rows to the worker processes.  At each 
 *   time step, worker processes must exchange border data with neighbors, 
 *   because a grid point's current temperature depends upon it's previous
 *   time step value plus the values of the neighboring grid points.  Upon
 *   completion of all time steps, the worker processes return their results
 *   to the master process.
 *
 *   Two data files are produced: an initial data set and a final data set.
 *   An X graphic of these two states displays after all calculations have
 *   completed.
 * AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
 *   to MPI: George L. Gusciora (1/95)
 * LAST REVISED: 06/12/13 Blaise Barney
 ****************************************************************************/

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

// extern void draw_heat(int nx, int ny);       /* X routine to create graph */

#define NXPROB      2000                 /* x dimension of problem grid */
#define NYPROB      2000                 /* y dimension of problem grid */
#define STEPS       100               /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define NONE        0                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

struct Parms { 
  float cx;
  float cy;
} parms = {0.1f, 0.1f};

void inidat(int nx, int ny, std::vector<std::vector<std::vector<float>>> &u);
void prtdat(int nx, int ny, const std::vector<std::vector<std::vector<float>>> &u, const std::string &fnam);
void update(int start, int end, int ny, std::vector<std::vector<float>> &u1, std::vector<std::vector<float>> &u2);

int main(int argc, char *argv[])
{
    std::vector<std::vector<std::vector<float>>> u(2, std::vector<std::vector<float>>(NXPROB, std::vector<float>(NYPROB)));
    int taskid, numworkers, numtasks, averow, rows, offset, extra, dest, source, left, right, msgtype, rc, start, end;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    numworkers = 6;

    if (taskid == MASTER) {
        /************************* master code *******************************/
        if ((numworkers > MAXWORKER) || (numworkers < MINWORKER)) {
            std::cerr << "ERROR: the number of tasks must be between " << MINWORKER + 1 << " and " << MAXWORKER + 1 << ".\n";
            std::cerr << "Quitting...\n";
            MPI_Abort(MPI_COMM_WORLD, rc);
            exit(1);
        }
        std::cout << "Starting mpi_heat2D with " << numworkers << " worker tasks.\n";
        std::cout << "Grid size: X= " << NXPROB << "  Y= " << NYPROB << "  Time steps= " << STEPS << "\n";
        std::cout << "Initializing grid and writing initial.dat file...\n";

        inidat(NXPROB, NYPROB, u);
        prtdat(NXPROB, NYPROB, u, "initial.dat");

        averow = NXPROB / numworkers;
        extra = NXPROB % numworkers;
        offset = 0;

        for (int i = 1; i <= numworkers; i++) {
            rows = (i <= extra) ? averow + 1 : averow; 
            left = (i == 1) ? NONE : i - 1;
            right = (i == numworkers) ? NONE : i + 1;

            dest = i;
            MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&left, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&right, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(u[0][offset].data(), rows * NYPROB, MPI_FLOAT, dest, BEGIN, MPI_COMM_WORLD);
            std::cout << "Sent to task " << dest << ": rows= " << rows << " offset= " << offset << " left= " << left << " right= " << right << "\n";
            offset += rows;
        }

        for (int i = 1; i <= numworkers; i++) {
            source = i;
            msgtype = DONE;
            MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
            MPI_Recv(u[0][offset].data(), rows * NYPROB, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &status);
        }

        std::cout << "Writing final.dat file and generating graph...\n";
        prtdat(NXPROB, NYPROB, u, "final.dat");
        std::cout << "Click on MORE button to view initial/final states.\n";
        std::cout << "Click on EXIT button to quit program.\n";
        // draw_heat(NXPROB, NYPROB);
        MPI_Finalize();
    } else {
        /************************* workers code **********************************/
        for (int iz = 0; iz < 2; iz++) {
            for (int ix = 0; ix < NXPROB; ix++) {
                for (int iy = 0; iy < NYPROB; iy++) {
                    u[iz][ix][iy] = 0.0f;
                }
            }
        }

        source = MASTER;
        msgtype = BEGIN;
        MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&left, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&right, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(u[0][offset].data(), rows * NYPROB, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &status);

        start = offset;
        end = offset + rows - 1;
        if (offset == 0) start = 1;
        if ((offset + rows) == NXPROB) end--;

        std::cout << "task=" << taskid << "  start=" << start << "  end=" << end << "\n";
        std::cout << "Task " << taskid << " received work. Beginning time steps...\n";

        int iz = 0;
        for (int it = 1; it <= STEPS; it++) {
            if (left != NONE) {
                MPI_Send(u[iz][offset].data(), NYPROB, MPI_FLOAT, left, RTAG, MPI_COMM_WORLD);
                MPI_Recv(u[iz][offset - 1].data(), NYPROB, MPI_FLOAT, left, LTAG, MPI_COMM_WORLD, &status);
            }
            if (right != NONE) {
                MPI_Send(u[iz][offset + rows - 1].data(), NYPROB, MPI_FLOAT, right, LTAG, MPI_COMM_WORLD);
                MPI_Recv(u[iz][offset + rows].data(), NYPROB, MPI_FLOAT, right, RTAG, MPI_COMM_WORLD, &status);
            }
            update(start, end, NYPROB, u[iz], u[1 - iz]);
            iz = 1 - iz;
        }

        MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
        MPI_Send(u[iz][offset].data(), rows * NYPROB, MPI_FLOAT, MASTER, DONE, MPI_COMM_WORLD);
        MPI_Finalize();
    }

    return 0;
}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, std::vector<std::vector<float>> &u1, std::vector<std::vector<float>> &u2)
{
    for (int ix = start; ix <= end; ix++) {
        for (int iy = 1; iy <= ny - 2; iy++) {
            u2[ix][iy] = u1[ix][iy] + 
                         parms.cx * (u1[ix + 1][iy] + u1[ix - 1][iy] - 2.0f * u1[ix][iy]) +
                         parms.cy * (u1[ix][iy + 1] + u1[ix][iy - 1] - 2.0f * u1[ix][iy]);
        }
    }
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, std::vector<std::vector<std::vector<float>>> &u) 
{
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            u[0][ix][iy] = static_cast<float>(ix * (nx - ix - 1) * iy * (ny - iy - 1));
        }
    }
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, const std::vector<std::vector<std::vector<float>>> &u, const std::string &fnam) 
{
    std::ofstream fp(fnam);
    for (int iy = ny - 1; iy >= 0; iy--) {
        for (int ix = 0; ix < nx; ix++) {
            fp << u[0][ix][iy] << " ";
        }
        fp << "\n";
    }
    fp.close();
}
