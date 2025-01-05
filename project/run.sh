#!/bin/bash


MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1 main.py 