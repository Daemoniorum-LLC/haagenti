# HCT Reference Decoder Makefile

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm

.PHONY: all clean test

all: test_decoder

test_decoder: test_decoder.c hct_decoder.h
	$(CC) $(CFLAGS) -o $@ test_decoder.c $(LDFLAGS)

test: test_decoder
	./test_decoder

clean:
	rm -f test_decoder
