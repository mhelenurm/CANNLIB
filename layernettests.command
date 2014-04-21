gcc -o layernettests.out layernettests.c neural/activation.c neural/neuron.c neural/layered_network.c neural/random.c -O2
./layernettests.out
rm layernettests.out
