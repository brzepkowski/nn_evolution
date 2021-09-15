program generate_correlation_operators
      implicit none

      CHARACTER(100) :: arg0, arg1, arg2

      integer(8) :: N, hil_size
			integer(8) :: N1, N2
      complex, dimension (:,:), allocatable :: S_corr
      character(len=:), allocatable :: filename_real, filename_imag
      real(8), dimension(:,:), allocatable :: S_corr_real, S_corr_imag
			integer(8) :: i,j,k,l

      ! Get the arguments passed in the command line
      CALL get_command_argument(1, arg0)
      ! Then convert them to integers
      READ(arg0,*) N
      hil_size=2**N

	! Allocate memory for the used arrays
	allocate(S_corr(0:hil_size-1,0:hil_size-1))
	allocate(S_corr_real(0:hil_size-1,0:hil_size-1))
  allocate(S_corr_imag(0:hil_size-1,0:hil_size-1))

	do i = 0,N-2
		do j = i+1, N-1
			! Fill S_corr with zeros
			S_corr = 0.0
			! Generate correlation operator <S+_i S-_j> + <S-_i S+_j> + <Sz_i Sz_j>
			do k = 0,hil_size-1
			   ! Add non-zero elements into the array
         if (btest(k,i) .eqv. btest(k,j)) then
            S_corr(k,k) = S_corr(k,k) + 0.25
         else
            S_corr(k,k) = S_corr(k,k) - 0.25
            l = ieor(k,2**i+2**j)
            S_corr(k,l) = 0.5
         endif
      enddo

			N1 = N - 1 - j
			N2 = N - 1 - i

			! Save correlation operator to appropriate file
			if (N1 < 10 .and. N2 < 10) then
				allocate(character(len=15) :: filename_real)
				write(filename_real, "(A7,I1,A1,I1,A5)") "S_corr_", N1, "_", N2, "_real"
				allocate(character(len=15) :: filename_imag)
				write(filename_imag, "(A7,I1,A1,I1,A5)") "S_corr_", N1, "_", N2, "_imag"
			else if (N1 >= 10 .and. N2 < 10) then
				allocate(character(len=16) :: filename_real)
				write(filename_real, "(A7,I2,A1,I1,A5)") "S_corr_", N1, "_", N2, "_real"
				allocate(character(len=16) :: filename_imag)
				write(filename_imag, "(A7,I2,A1,I1,A5)") "S_corr_", N1, "_", N2, "_imag"
			else if (N1 < 10 .and. N2 >= 10) then
				allocate(character(len=16) :: filename_real)
				write(filename_real, "(A7,I1,A1,I2,A5)") "S_corr_", N1, "_", N2, "_real"
				allocate(character(len=16) :: filename_imag)
				write(filename_imag, "(A7,I1,A1,I2,A5)") "S_corr_", N1, "_", N2, "_imag"
			else if (N1 >= 10 .and. N2 >= 10) then
				allocate(character(len=17) :: filename_real)
				write(filename_real, "(A7,I2,A1,I2,A5)") "S_corr_", N1, "_", N2, "_real"
				allocate(character(len=17) :: filename_imag)
				write(filename_imag, "(A7,I2,A1,I2,A5)") "S_corr_", N1, "_", N2, "_imag"
			endif

			S_corr_real = real(S_corr)
			S_corr_imag = aimag(S_corr)

			open(10, file = filename_real, form = 'unformatted')
			write (10) S_corr_real
			close(10)

			open(10, file = filename_imag, form = 'unformatted')
			write (10) S_corr_imag
			close(10)

			deallocate(filename_real)
			deallocate(filename_imag)

	   enddo
	enddo
end
