#!/bin/sh
# Eikonal PINN hyperparameter sweep


# Activation functions
for SEED in {0..5}
do
	python main.py -u 30 -H 10 -a Tanh       -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ReLU       -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a LeakyReLU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU        -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a Swish      -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
done


# Network size
for SEED in {0..5}
do
	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 10  -H 30 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 50  -H 6 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 100 -H 3 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
done


# Randomized vs fixed training samples
for SEED in {0..5}
do
	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25    -b 256 -p 1. -i 10. -S ${SEED}
done


# Number of training samples
for SEED in {0..5}
do
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.01 -R -b 256 -p 1. -i 10. -S ${SEED}
	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.10 -R -b 256 -p 1. -i 10. -S ${SEED}
	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.50 -R -b 256 -p 1. -i 10. -S ${SEED}
done


# Batch size
for SEED in {0..5}
do
 	python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 64 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 128 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 256 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 512 -p 1. -i 10. -S ${SEED}
  python main.py -u 30 -H 10 -a ELU  -r 0.0001 -e 1000 -s 0.25 -R -b 0 -p 1. -i 10. -S ${SEED}
done
