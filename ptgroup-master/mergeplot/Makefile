CC=g++
CFLAGS=-c -g -Wall `root-config --cflags`
LDFLAGS=`root-config --glibs`
GFORT=ccylinder.o intersept.o targcell.o stcounter.o -lgfortran
SOURCES=mergeplot.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=MergePlot
 all: $(SOURCES) $(EXECUTABLE)
 $(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(GFORT)
 .cc.o:
	$(CC) $(CFLAGS) $< -o $@ $(CFLAGS)
 clean:
	rm ./*.o ./MergePlot


