#!/bin/sh

# exit on error
set -e
# turn on command echoing
set -v

# create output folder for initial run
mkdir -p out_init
# start the initial run
../../build/partmc urban_plume_init.spec &> run_init.log

# run the postprocessing 
../../build/s_variables_processes &> s_variables_processes.log
../../build/s_mixing_state_processes &> s_mixing_state_processes.log

