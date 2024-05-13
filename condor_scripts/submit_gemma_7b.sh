#!/bin/bash

# File to hold the submit script
submit_file="submit_jb_calc.sub"

# Start of the submit file
cat > $submit_file <<EOF
# Executable
executable = /home/apanfilov/jailbreak_filter/llm-adaptive-attacks/condor_scripts/jb_calc.sh

# Arguments template
arguments = "\$(index)"

# Logs
error = log/job.\$(Cluster).\$(Process).err
output = log/job.\$(Cluster).\$(Process).out
log = log/job.\$(Cluster).\$(Process).log

# Compute requests
request_memory = 32GB
request_cpus = 4
request_gpus = 1
# request_memory = 150GB
# request_cpus = 4
# request_gpus = 4
# requirements = (CUDACapability >= 8)

# Queue jobs
EOF

# Step size for the loop
step=1

# Starting and ending index for your example (from 0 to 1)
start_loop=0
end_loop=1

# Generate the queue section
for i in $(seq $start_loop $step $end_loop)
do
    echo "queue index from (" >> $submit_file
    echo "    $i " >> $submit_file
    echo ")" >> $submit_file
done
