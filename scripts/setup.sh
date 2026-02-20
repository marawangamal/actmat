# Tamia cluster setup
module load httpproxy
source $SCRATCH/tangent_task_arithmetic/.venv/bin/activate
cd $SCRATCH/tangent_task_arithmetic
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export OPENCLIP_CACHEDIR=$SCRATCH/openclip