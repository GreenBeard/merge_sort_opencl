build: main.c
	gcc -g main.c -l OpenCL

clean:
	$(RM) ./a.out
