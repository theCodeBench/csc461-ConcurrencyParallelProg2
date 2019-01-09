SOURCE = prog2.cu

OBJS = $(SOURCE:.cpp=.o)

FILES = prog2.cu lodepng.cpp

#GNU C/C++ Compiler
GCC = g++

# NVCC compiler
LINK = nvcc

# Libraries
LIBS = -Xcompiler -fopenmp

# Compiler flags
CFLAGS = -std=c++11

# Targets include all and clean

all : prog2

prog2: $(OBJS)
	$(LINK) -o $@ $(FILES) $(CFLAGS) $(LIBS)
		

debug: CXXFLAGS += -DDEBUG -g
debug: main

clean: rm prog2

help:
	@echo "	make all   - builds the main target"
	@echo "	make       - same as make all"
	@echo "	make clean - remove prog2"
%.d: %.cpp
	@set -e; /usr/bin/rm -rf $@;$(GCC) -MM $< $(CXXFLAGS) > $@
