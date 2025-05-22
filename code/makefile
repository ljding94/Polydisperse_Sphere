Compile=g++ -g -std=c++2a -O3 # improve run time e.g. 10s to 4s
calc_Iq: calc_Iq.o
	$(Compile) -o calc_Iq calc_Iq.o
calc_Iq.o: calc_Iq.cpp
	$(Compile) -o calc_Iq.o -c calc_Iq.cpp
clean:
	rm -f calc_Iq.o