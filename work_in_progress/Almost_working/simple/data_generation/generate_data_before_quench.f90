program generate_data_before_quench
  use lapack95
  implicit none
  integer(8) :: N, hil_size
   real(8),dimension(:,:), allocatable :: hamil
   complex,dimension(:,:), allocatable :: U ! Evolution operator, which we will also save in a file
   real(8),dimension(:), allocatable :: ener,v
   real(8) :: h, timestep
   integer(8) :: i,j,k,l
   integer :: info

   CHARACTER(100) :: N_string

   ! Get the arguments passed in the command line
   CALL get_command_argument(1, N_string)
   ! Then convert them to reals
   READ(N_string,*) N

   hil_size=2**N
   allocate(hamil(0:hil_size-1,0:hil_size-1))
   allocate(U(0:hil_size-1,0:hil_size-1))
   allocate(ener(0:hil_size-1))
   allocate(v(0:hil_size-1))

  hamil=0._8
  h=1._8

  do k = 0,hil_size-1
     do i = 0,N-1
        !j = mod(i+1,N) ! P.B.C.
        if (i < N-1) then
           j = i + 1 ! O.B.C
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

  open(10, file = 'H.dat', form = 'unformatted')
  write (10) hamil
  close(10)

  ! print *, "[FORTRAN] H: "
  !
  ! do i=0,hil_size-1
  !  do j=0,hil_size-1
  !     write (*,'(E)', advance="no") hamil(i,j)
  !  enddo
  !  write (*, *) ""
  ! enddo

  call syevd(hamil,ener,'V',info=info)

  ! print *, "[FORTRAN] eig_H: "
  !
  ! do i=0,hil_size-1
  !  do j=0,hil_size-1
  !     write (*,'(E)', advance="no") hamil(i,j)
  !  enddo
  !  write (*, *) ""
  ! enddo
  !
  ! ! print *, "eigenvectors_H", hamil

  ! print *, ener

  open(1, file = 'eigenvalues_H.dat', form = 'unformatted')
  write (1) ener
  close(1)

  open(1, file = 'eigenvectors_H.dat', form = 'unformatted')
  write (1) hamil
  close(1)

end
