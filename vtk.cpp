#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "vtk.h"
#include "data.h"

char checkpoint_basename[1024];
char result_filename[1024];

/**
 * @brief Set the default basename for file output to out/vortex
 * 
 */
void set_default_base() {
    set_basename("out/vortex");
}

/**
 * @brief Set the basename for file output
 * 
 * @param base Basename string
 */
void set_basename(char *base) {
    checkpoint_basename[0] = '\0';
    result_filename[0] = '\0';
    sprintf(checkpoint_basename, "%s-%%d.vtk", base);
    sprintf(result_filename, "%s.vtk", base);
}

/**
 * @brief Get the basename for file output
 * 
 * @return char* Basename string
 */
char *get_basename() {
    return checkpoint_basename;
}

/**
 * @brief Write a checkpoint VTK file (with the iteration number in the filename)
 * 
 * @param iteration The current iteration number
 * @return int Return whether the write was successful
 */
int write_checkpoint(int iters, double t) { 
    char filename[1024];
    sprintf(filename, checkpoint_basename, iters);
    return write_vtk(filename, iters, t);
}

/**
 * @brief Write the final output to a VTK file
 * 
 * @return int Return whether the write was successful
 */
int write_result(int iters, double t) {
    return write_vtk(result_filename, iters, t);
}

/**
 * @brief Write a VTK file with the current state of the simulation
 * 
 * @param filename The filename to write out
 * @return int Return whether the write was successful
 */
int write_vtk(char* filename, int iters, double t) {
    FILE * f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error");
        return -1;
    }

    // Write the VTK header information
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "Vortex Output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");

    // Write out data for the simulation time and step number
    fprintf(f, "FIELD FieldData 2\n");
    fprintf(f, "TIME 1 1 double\n");
    fprintf(f, "%lf\n", t);
    fprintf(f, "CYCLE 1 1 int\n");
    fprintf(f, "%d\n", iters);

    // Write out the dimensions of the grid
    fprintf(f, "DIMENSIONS %d %d 1\n", u_size_x, u_size_y);
    fprintf(f, "ORIGIN 0 0 0\n");
    fprintf(f, "SPACING 1 1 1\n");

    // Write out the u variable
    int points = u_size_x * u_size_y;
    fprintf(f, "POINT_DATA %d\n", points);
    fprintf(f, "SCALARS u double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < u_size_y; j++) {
        for (int i = 0; i < u_size_x; i++)
            fprintf(f, "%.12e ", u[i][j]);
        fprintf(f, "\n");
    }

    // Write out the v variable
    fprintf(f, "\nSCALARS v double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < v_size_y; j++) {
        for (int i = 0; i < v_size_x; i++)
            fprintf(f, "%.12e ", v[i][j]);
        fprintf(f, "\n");
    } 

    // Write out the p variable
    fprintf(f, "\nSCALARS p double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < p_size_y; j++) {
        for (int i = 0; i < p_size_x; i++)
            fprintf(f, "%.12e ", p[i*p_size_y+j]);
        fprintf(f, "\n");
    } 

    // Write out the flag variable
    fprintf(f, "\nSCALARS flag int 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < flag_size_y; j++) {
        for (int i = 0; i < flag_size_x; i++)
            fprintf(f, "%d ", flag[i*flag_size_y+j]);
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}