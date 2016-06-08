# Build mode. 
# 0: debug mode
# 1: release mode with optimiziation enabled
BUILD_MODE = 0


CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -fPIC
ifeq ("$(BUILD_MODE)", "0")
  CFLAGS += -g -ftrapv
endif
ifeq ("$(BUILD_MODE)", "1")
  CFLAGS += -O3 -msse -msse2 -msse3 -mfpmath=sse
endif
LIBS = blas/blas.a -pthread
SHVER = 1
#LIBS = -lblas


all: train

lib: linear.o tron.o blas/blas.a
	$(CXX) -shared -dynamiclib linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: tron.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	cd blas;	make clean
	cd matlab;	make clean
	rm -f *~ tron.o linear.o train predict liblinear.so.$(SHVER)
