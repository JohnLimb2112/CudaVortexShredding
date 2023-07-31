#ifndef ARGS_H
#define ARGS_H

extern int verbose;
extern int no_output;
extern int output_freq;
extern int enable_checkpoints;
extern int fixed_dt;

void parse_args(int argc, char *argv[]);
void print_opts();

#endif