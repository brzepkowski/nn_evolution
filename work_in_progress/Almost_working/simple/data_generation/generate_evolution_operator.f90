!###############################################################################
!#########  Implementation of matrix exponent using expokit library  ###########
!###############################################################################
module exponent
! double precision kind constant
! integer, parameter :: dp = kind(1.d0)

contains
   ! Calculate exp(t*H) for an N-by-N matrix H using Expokit.
   function expm(t, H) result(expH)
     real(8), intent(in) :: t
     complex(8), dimension(:,:), intent(in) :: H
     complex(8), dimension(size(H,1),size(H,2)) :: expH

     ! Expokit variables
     external :: ZGPADM
     integer, parameter :: ideg = 6
     complex(8), dimension(4*size(H,1)*size(H,2) + ideg + 1) :: wsp
     integer, dimension(size(H,1))  :: iwsp
     integer :: iexp, ns, iflag, n

     ! print *, "________________________"
     ! print *, "H"
     ! print *, H
     ! print *, "________________________"

     if (size(H,1) /= size(H,2)) then
        stop 'expm: matrix must be square'
     end if

     n = size(H,1)
     call ZGPADM(ideg, n, t, H, n, wsp, size(wsp,1), iwsp, iexp, ns, iflag)
     expH = reshape(wsp(iexp:iexp+n*n-1), shape(expH))
   end function expm
end module exponent

program generate_evolution_operator
  use exponent
  implicit none

  integer(8) :: N, hil_size
  real(8),dimension(:,:), allocatable :: hamil
  ! Evolution operator, which we will also save in a file
  complex(8),dimension(:,:), allocatable :: U, U1
  real(8),dimension(:,:), allocatable :: U_real, U_imag
  real(8) :: timestep
  complex(8) :: imaginary
  real(8) :: t

  CHARACTER(100) :: N_string, timestep_string

  ! Get the arguments passed in the command line
  CALL get_command_argument(1, N_string)
  CALL get_command_argument(2, timestep_string)
  ! Then convert them to reals
  READ(N_string,*) N
  READ(timestep_string,*) timestep

  hil_size=2**N
  allocate(hamil(0:hil_size-1,0:hil_size-1))
  allocate(U(0:hil_size-1,0:hil_size-1))
  allocate(U_real(0:hil_size-1,0:hil_size-1))
  allocate(U_imag(0:hil_size-1,0:hil_size-1))

  open(10, file = 'HQ.dat', form = 'unformatted')
  read (10) hamil
  close(10)

  imaginary = cmplx(0.0,-1.0)
  U1 = (imaginary*timestep)*hamil
  t = 1.0
  U = expm(t, U1)

  ! print *, "________________________"
  ! print *, "U:"
  ! print *, U
  ! print *, "________________________"


  ! ! Generate and save the evolution operator
  U_real = real(U)
  U_imag = aimag(U)

  ! print *, "________________________"
  ! print *, "U_real:"
  ! print *, U_real
  ! print *, "________________________"
  !
  ! print *, "________________________"
  ! print *, "U_imag:"
  ! print *, U_imag
  ! print *, "________________________"

  open(10, file = 'U_real.dat', form = 'unformatted')
  write (10) U_real
  close(10)

  open(10, file = 'U_imag.dat', form = 'unformatted')
  write (10) U_imag
  close(10)

end
