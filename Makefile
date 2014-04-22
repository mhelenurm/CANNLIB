CC=gcc
CFLAGS=-O2

layernettestsmake: layernettests.c neural/activation.c neural/neuron.c neural/layered_network.c neural/random.c bitmap/colors.c bitmap/bitmap.c
	$(CC) -o a.out layernettests.c neural/activation.c neural/neuron.c neural/layered_network.c neural/random.c bitmap/colors.c bitmap/bitmap.c $(CFLAGS)

activationtestsmake: activationtests.c neural/activation.c
	$(CC) -o a.out activationtests.c neural/activation.c

neurontestsmake: neurontests.c neural/activation.c neural/neuron.c
	$(CC) -o a.out neurontests.c neural/activation.c neural/neuron.c

layer: layernettestsmake run clean

activation: activationtestsmake run clean

neuron: neurontestsmake run clean

run:
	./a.out

.PHONY: clean

clean:
	rm a.out
