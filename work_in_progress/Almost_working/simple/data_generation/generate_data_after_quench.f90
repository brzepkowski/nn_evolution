program generate_data_after_quench
  use lapack95
  implicit none

  integer(8) :: N, hil_size
  real(8),dimension(:,:), allocatable :: hamil
  real(8),dimension(:), allocatable :: ener,v
  real(8) :: h
  integer(8) :: i,j,k,l
  integer :: info

  CHARACTER(100) :: N_string, timestep_string

  ! Get the arguments passed in the command line
  CALL get_command_argument(1, N_string)
  ! Then convert them to reals
  READ(N_string,*) N

  hil_size=2**N
  allocate(hamil(0:hil_size-1,0:hil_size-1))
  allocate(ener(0:hil_size-1))
  allocate(v(0:hil_size-1))

  hamil=0._8
  h=0._8

  do k = 0,hil_size-1
     do i = 0,N-1
        !j = mod(i+1,N) ! P.B.C.
        if (i < N-1) then
           j = i + 1! O.B.C
           if (btest(k,i) .eqv. btest(k,j)) then
              hamil(k,k) = hamil(k,k) + 0.25
           else
              hamil(k,k) = hamil(k,k) - 0.25
              l = ieor(k,2**i+2**j)
              hamil(k,l) = 0.5
           endif
           hamil(k,k) = hamil(k,k) - h*popcnt(k) ! popcnt return the count of number of bits set to 1 in number k (spin 1/2*2...)
        endif
     enddo
  enddo

  open(10, file = 'HQ.dat', form = 'unformatted')
  write (10) hamil
  close(10)

  ! print *, "[FORTRAN] HQ: "
  ! do i=0,hil_size-1
  !  do j=0,hil_size-1
  !     write (*,'(E)', advance="no") hamil(i,j)
  !  enddo
  !  write (*, *) ""
  ! enddo

  ! ! Diagonalize the hamiltonian HQ
  ! call syevd(hamil,ener,'V',info=info)
  !
  ! ! print *, "[FORTRAN] eig_HQ: "
  ! ! do i=0,hil_size-1
  ! !  do j=0,hil_size-1
  ! !     write (*,'(E)', advance="no") hamil(i,j)
  ! !  enddo
  ! !  write (*, *) ""
  ! ! enddo
  !
  ! open(10, file = 'eigenvectors_HQ.dat', form = 'unformatted')
  ! write (10) hamil
  ! close(10)

end
