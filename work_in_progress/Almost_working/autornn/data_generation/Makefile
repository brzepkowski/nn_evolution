FC	= gfortran
FFLAGS        = -O3

# Among the 3 possibilities below, uncomment the appropriate
# case for your environment and comment the others.

# case 1: works when LAPACK and BLAS are installed.
#OBJLIBS =
#LIBS    = -llapack -lblas

# case 2: works when LAPACK is not installed but BLAS is.
#LIBS    = -lblas
#OBJLIBS = lapack.o

# case 3: works when neither LAPACK nor BLAS are installed.
OBJLIBS = blas.o lapack.o
LIBS    =

#############
#############

EXPOKIT   = expokit.o
#EXPOKIT   = dgpadm.o dspadm.o zgpadm.o zhpadm.o                    \
#            dgchbv.o dschbv.o zgchbv.o          dnchbv.o znchbv.o  \
#            dgexpv.o dsexpv.o zgexpv.o zhexpv.o                    \
#            dgphiv.o dsphiv.o zgphiv.o zhphiv.o

MATAID    = mataid.o
#MATAID    = dgmatv.o zgmatv.o \
#            dgcnvr.o zgcnvr.o loadhb.o

OBJS      = clock.o $(EXPOKIT) $(MATAID) $(OBJLIBS)

generate_evolution_operator: $(OBJS) generate_evolution_operator.o
	$(FC) -o generate_evolution_operator generate_evolution_operator.o $(OBJS) $(LIBS)

generate_evolution_operator.o:
	$(FC) $(FFLAGS) -c generate_evolution_operator.f90
