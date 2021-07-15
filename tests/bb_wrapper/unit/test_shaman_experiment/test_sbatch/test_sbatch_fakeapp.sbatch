#!/bin/bash
#SBATCH --nodelist=pic52
#SBATCH --exclusive
#SBATCH --job-name=fantastic_euler
fakeapp-nompi -d /fs1/roberts -R -N 5166 -w 4k -W 4k -S 14M -L 307 -l 0k -Z 100G &
wait;