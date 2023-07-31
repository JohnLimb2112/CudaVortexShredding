#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"


/**
 * @brief Computation of tentative velocity field (f, g)
 * 
 */
void compute_tentative_velocity() {
    for (int i = 1; i < imax; i++) {
        for (int j = 1; j < jmax+1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i*flag_size_y+j] & C_F) && (flag[(i+1)*flag_size_y+j] & C_F)) {
                double du2dx = ((u[i][j] + u[i+1][j]) * (u[i][j] + u[i+1][j]) +
                                y * fabs(u[i][j] + u[i+1][j]) * (u[i][j] - u[i+1][j]) -
                                (u[i-1][j] + u[i][j]) * (u[i-1][j] + u[i][j]) -
                                y * fabs(u[i-1][j] + u[i][j]) * (u[i-1][j]-u[i][j]))
                                / (4.0 * delx);
                double duvdy = ((v[i][j] + v[i+1][j]) * (u[i][j] + u[i][j+1]) +
                                y * fabs(v[i][j] + v[i+1][j]) * (u[i][j] - u[i][j+1]) -
                                (v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] + u[i][j]) -
                                y * fabs(v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] - u[i][j]))
                                / (4.0 * dely);
                double laplu = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / delx / delx +
                                (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / dely / dely;
   
                f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i*flag_size_y+j] & C_F) && (flag[i*flag_size_y+j+1] & C_F)) {
                double duvdx = ((u[i][j] + u[i][j+1]) * (v[i][j] + v[i+1][j]) +
                                y * fabs(u[i][j] + u[i][j+1]) * (v[i][j] - v[i+1][j]) -
                                (u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] + v[i][j]) -
                                y * fabs(u[i-1][j] + u[i-1][j+1]) * (v[i-1][j]-v[i][j]))
                                / (4.0 * delx);
                double dv2dy = ((v[i][j] + v[i][j+1]) * (v[i][j] + v[i][j+1]) +
                                y * fabs(v[i][j] + v[i][j+1]) * (v[i][j] - v[i][j+1]) -
                                (v[i][j-1] + v[i][j]) * (v[i][j-1] + v[i][j]) -
                                y * fabs(v[i][j-1] + v[i][j]) * (v[i][j-1] - v[i][j]))
                                / (4.0 * dely);
                double laplv = (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / delx / delx +
                                (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / dely / dely;

                g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    /* f & g at external boundaries */
    for (int j = 1; j < jmax+1; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (int i = 1; i < imax+1; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation 
 * 
 */
void compute_rhs() {
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1;j < jmax+1; j++) {
            if (flag[i*flag_size_y+j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i*rhs_size_y+j] = ((f[i][j] - f[i-1][j]) / delx + 
                             (g[i][j] - g[i][j-1]) / dely)
                             / del_t;
            }
        }
    }
}


/**
 * @brief Red/Black SOR to solve the poisson equation.
 * 
 * @return Calculated residual of the computation
 * 
 */
// 

#define N 514 //imax +2
#define M 130 //jmax +2
#define MatrixSize 66820 //(jmax+2) * (imax+2)



//DEVICE CODE
__global__ void poissonKernel(double *p, char *flag, double *rhs, double rdx2,
    double rdy2, double beta_2, double omega, int rb){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    int iSkip = 130; //Value to move to next row/i, is p_size_y
    int i = idx/iSkip;
    if((i+idx) % 2 != rb){
        //If within indexes of values we are interested in
        if((idx > iSkip) && (idx % iSkip != 0) && (idx % iSkip != iSkip-1) && (idx < MatrixSize - iSkip)){
            if (flag[idx] == (C_F | B_NSEW)) {
                /* five point star for interior fluid cells */
                p[idx] = (1.0 - omega) * p[idx] - 
                        beta_2 * ((p[idx + iSkip] + p[idx-iSkip] ) *rdx2
                            + (p[idx+1] + p[idx-1]) * rdy2
                            - rhs[idx]);
            } else if (flag[idx] & C_F) { 
                /* modified star near boundary */

                double eps_E = ((flag[idx+iSkip] & C_F) ? 1.0 : 0.0);
                double eps_W = ((flag[idx-iSkip] & C_F) ? 1.0 : 0.0);
                double eps_N = ((flag[idx+1] & C_F) ? 1.0 : 0.0);
                double eps_S = ((flag[idx-1] & C_F) ? 1.0 : 0.0);

                double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                p[idx] = (1.0 - omega) * p[idx] -
                    beta_mod * ((eps_E * p[idx+iSkip] + eps_W * p[idx-iSkip]) * rdx2
                        + (eps_N * p[idx+1] + eps_S * p[idx-1]) * rdy2
                        - rhs[idx]);
            }
        }
    }    
}

__global__ void residualKernel(double *p, char *flag, double *rhs, double rdx2,
     double rdy2, double p0, int fluid_cells, double eps, double *d_res){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int iSkip = 130; //Value to move to next row/i, is p_size_y
    if((idx > iSkip) && (idx % iSkip != 0) && (idx % iSkip != iSkip-1) && (idx < MatrixSize - iSkip)){
        //If within indexes of values we are interested in
        if (flag[idx] & C_F) {
            double eps_E = ((flag[idx+iSkip] & C_F) ? 1.0 : 0.0);
            double eps_W = ((flag[idx-iSkip] & C_F) ? 1.0 : 0.0);
            double eps_N = ((flag[idx+1] & C_F) ? 1.0 : 0.0);
            double eps_S = ((flag[idx-1] & C_F) ? 1.0 : 0.0);

            /* only fluid cells */
            double add = (eps_E * (p[idx+iSkip] - p[idx]) - 
                eps_W * (p[idx] - p[idx-iSkip])) * rdx2  +
                (eps_N * (p[idx+1] - p[idx]) -
                eps_S * (p[idx] - p[idx-1])) * rdy2  -  rhs[idx];
            *d_res += add * add;
        }
    }
    
}



double poisson(double *d_p, char *d_flag, double *d_rhs, double *d_res) {

    cudaError_t err = cudaSuccess;

    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0 = 0.0;
    /* Calculate sum of squares */
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i*flag_size_y+j] & C_F) { p0 += p[i*flag_size_y+j] * p[i*p_size_y+j]; }
        }
    }
   
    p0 = sqrt(p0 / fluid_cells); 
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */

    double *p2, *rhs2;
    char *flag2;

    //MEMORY ALLOCATIONS FOR HOST
    p2 = (double *) malloc(sizeof(double)*MatrixSize);
    rhs2 = (double *) malloc(sizeof(double)*MatrixSize);
    flag2 = (char *) malloc(sizeof(char)*MatrixSize);

    if(p2 == NULL || rhs2 == NULL || flag2 == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    for(int i =0; i < MatrixSize; i++){
        rhs2[i] = rhs[i];
        flag2[i] = flag[i];
    }

    //TRANSFER FROM HOST TO DEVICE MEMORY
    err = cudaMemcpy(d_rhs, rhs2, sizeof(double) * MatrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector RHS from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_flag, flag2, sizeof(char) * MatrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector RHS from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    int block_size = 520; //Means grid is 257
    int grid_size = MatrixSize/block_size;

    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {
        for(int rb = 0; rb <2; rb++){
            //First Kernel for p values
            poissonKernel<<<grid_size,block_size>>>(d_p, d_flag, d_rhs, rdx2, rdy2, beta_2, omega, rb);
            cudaError_t error = cudaPeekAtLastError();
            if (error == 0) error = cudaDeviceSynchronize(); // wait for GPU threads to finish
            if (error != 0) printf("CUDA Error %s (%d)", cudaGetErrorString(error), (int)error);
        }
        //Send current residual to device
        cudaMemcpy(d_res, &res, sizeof(double), cudaMemcpyHostToDevice);

        //Second Kernel for Residual
        residualKernel<<<grid_size,block_size>>>(d_p, d_flag, d_rhs, rdx2, rdy2, p0, fluid_cells, eps, d_res);
        cudaDeviceSynchronize();
        //Retrieve residual after calc
        cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

        res = sqrt(res / fluid_cells) / p0;

        if(res < eps){break;}
    }
    cudaDeviceSynchronize(); 
    //Retrieve P from device
    err = cudaMemcpy(p2, d_p, sizeof(double) * MatrixSize, cudaMemcpyDeviceToHost); 
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector P from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Put p2 back in p
    for(int i = 0; i<MatrixSize; i++){
        p[i] = p2[i];
    }
    
    //FREE BIRD (and also memory)
    free(rhs2);
    free(flag2);
    free(p2);
    return res;
}



/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity() {   
    for (int i = 1; i < imax-2; i++) {
        for (int j = 1; j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i*flag_size_y+j] & C_F) && (flag[(i+1)*flag_size_y+j] & C_F)) {
                u[i][j] = f[i][j] - (p[(i+1)*p_size_y+j] - p[i*p_size_y+j]) * del_t / delx;
            }
        }
    }
    
    for (int i = 1; i < imax-1; i++) {
        for (int j = 1; j < jmax-2; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i*flag_size_y+j] & C_F) && (flag[i*flag_size_y+j+1] & C_F)) {
                v[i][j] = g[i][j] - (p[i*p_size_y+j+1] - p[i*p_size_y+j]) * del_t / dely;
            }
        }
    }
}


/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval() {
    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        double umax = 1.0e-10;
        double vmax = 1.0e-10; 
        
        for (int i = 0; i < imax+2; i++) {
            for (int j = 1; j < jmax+2; j++) {
                umax = fmax(fabs(u[i][j]), umax);
            }
        }

        for (int i = 1; i < imax+2; i++) {
            for (int j = 0; j < jmax+2; j++) {
                vmax = fmax(fabs(v[i][j]), vmax);
            }
        }

        double deltu = delx / umax;
        double deltv = dely / vmax; 
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

        if (deltu < deltv) {
            del_t = fmin(deltu, deltRe);
        } else {
            del_t = fmin(deltv, deltRe);
        }
        del_t = tau * del_t; /* multiply by safety factor */
    }
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
    clock_t start,stop;
    clock_t tempStart, tempStop;
    clock_t loopsStart, loopsStop;
    double loops_time;
    double total_time;
    //double tent_veloc_time;
    //double rhs_time;
    double poisson_time;
    //double update_veloc_time;
    //double boundary_cond_time;
    start = clock();
    set_defaults();
    parse_args(argc, argv);
    setup();
    if (verbose) print_opts();

    allocate_arrays();
    problem_set_up();
    double res;


    double *d_p, *d_rhs;
    char *d_flag;
    double *d_res;

    //MEMORY ALLOCATIONS FOR DEVICE
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **) &d_p, sizeof(double) * MatrixSize);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector P (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_res, sizeof(double));
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector RHS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &d_flag, sizeof(char) * MatrixSize);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector FLAG (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &d_rhs, sizeof(double) * MatrixSize);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector RHS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Device allocations done!\n");
    


    
    /* Main loop */
    int iters = 0;
    double t;
    loopsStart = clock();
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        //tempStart = clock();
        compute_tentative_velocity();
        ///tempStop = clock();
        //tent_veloc_time = (double) (tempStop - tempStart) /CLOCKS_PER_SEC;

        //tempStart = clock();
        compute_rhs();
        //tempStop = clock();
        //rhs_time = (double) (tempStop - tempStart) /CLOCKS_PER_SEC;

        tempStart = clock();
        res = poisson(d_p, d_flag, d_rhs, d_res);
        tempStop = clock();
        poisson_time = (double) (tempStop - tempStart) /CLOCKS_PER_SEC;

        //tempStart = clock();
        update_velocity();
        //tempStop = clock();
        //update_veloc_time = (double) (tempStop - tempStart) /CLOCKS_PER_SEC;

        //tempStart = clock();
        apply_boundary_conditions();
        //tempStop = clock();
        //boundary_cond_time = (double) (tempStop - tempStart) /CLOCKS_PER_SEC;

        if ((iters % output_freq == 0)) {
            loopsStop = clock();
            loops_time = (double) (loopsStop - loopsStart) /CLOCKS_PER_SEC;
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);
            printf("Time: %lf, PoissonTime: %lf\n", loops_time, poisson_time);
            //printf("Time: %lf, TentVeloc: %lf, RHSTime: %lf, PoissonTime: %lf, UpdateVeloc: %lf, BoundaryCond: %lf\n", 
            //loops_time, tent_veloc_time, rhs_time, poisson_time, update_veloc_time, boundary_cond_time);
            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t+del_t);
            loopsStart = clock();
        }
    } /* End of main loop */

    stop = clock();
    total_time = (double) (stop - start) /CLOCKS_PER_SEC;
    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("FULL TIME: %lf", total_time);
    printf("Simulation complete.\n");

    if (!no_output)
        write_result(iters, t);

    free_arrays();
    cudaFree(d_p);
    cudaFree(d_rhs);
    cudaFree(d_flag);
    cudaFree(d_res);
    return 0;
}

