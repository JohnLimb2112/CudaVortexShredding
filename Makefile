CC=nvcc
CFLAGS=-O3 
LIBFLAGS=-lm -lcurand

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o boundary.o vortex.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories vortex

obj/vortex.o: vortex.cu
	$(CC) -c -o $@ $< $(CFLAGS)

obj/%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS) 

vortex: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f vortex

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

