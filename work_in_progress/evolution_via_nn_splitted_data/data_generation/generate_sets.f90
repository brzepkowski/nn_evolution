!###############################################################################
!#########  Implementation of matrix exponent using expokit library  ###########
!###############################################################################
module subprograms

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

     if (size(H,1) /= size(H,2)) then
        stop 'expm: matrix must be square'
     end if

     n = size(H,1)
     call ZGPADM(ideg, n, t, H, n, wsp, size(wsp,1), iwsp, iexp, ns, iflag)
     expH = reshape(wsp(iexp:iexp+n*n-1), shape(expH))
   end function expm


   ! Eigenvectors passed to this function are STORED AS COLUMNS of the matrix
   function generate_linearly_combined_vector(eigenvectors) result(vector)
      real(8), dimension(:,:), intent(in) :: eigenvectors   ! input
      real(8), dimension(size(eigenvectors,1)) :: vector ! output
      real(8) :: r

      vector = 0._8

      ! Generate linearly combined vector
      do i = 1, size(eigenvectors, 2)
        call random_number(r)
        ! Adding eigenvector (stored as column), multiplied by random "r"
        vector = vector + (r*eigenvectors(:,i))
      end do

      vector = vector / NORM2(vector)

    end function generate_linearly_combined_vector


    subroutine generate_set(eigenvectors, target_set, set_size, total_timesteps)
      real(8), dimension(:,:), intent(in) :: eigenvectors
      complex(8), dimension(:,:), intent(inout) :: target_set
      integer(8), intent(in) :: set_size, total_timesteps

      real(8), dimension(size(eigenvectors,1)) :: vector
      real(8) :: r
      integer(8) :: i, j


      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Generating initial linearly combined set of vectors
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      do i = 1, set_size
        vector = 0._8

        ! Generate linearly combined vector
        do j = 1, size(eigenvectors, 2)
          call random_number(r)
          ! Adding eigenvector (stored as column), multiplied by random "r"
          vector = vector + (r*eigenvectors(:,j))
        end do

        ! Normalize the vector
        vector = vector / NORM2(vector)

        ! Add obtained vector as new ROW to the training set
        target_set(i,:) = vector(:)
        if (MOD(i, 10) == 0) then
          print *, "i: ", i, " / ", set_size
        end if
      end do

    end subroutine generate_set

end module subprograms

program generate_sets
  use subprograms
  implicit none

  ! Launch parameters
  integer(8) :: N, set_number
  real(8) :: timestep, total_time
  CHARACTER(4) :: N_string, timestep_string, total_time_string
  CHARACTER(4) :: set_number_string

  ! Hamiltonian, eigenvectors and evolution operator
  real(8),dimension(:,:), allocatable :: hamil_q
  real(8),dimension(:,:), allocatable :: eigenvectors_H
  complex(8),dimension(:,:), allocatable :: U, U1

  ! All allocatable sets
  complex(8),dimension(:,:), allocatable :: training_set_input, training_set_output
  real(8),dimension(:,:), allocatable :: training_set_input_reshaped, training_set_output_reshaped
  complex(8),dimension(:,:), allocatable :: validation_set_input, validation_set_output
  real(8),dimension(:,:), allocatable :: validation_set_input_reshaped, validation_set_output_reshaped
  complex(8),dimension(:,:), allocatable :: testing_set_input, testing_set_output
  real(8),dimension(:,:), allocatable :: testing_set_input_reshaped, testing_set_output_reshaped

  ! Other variables
  integer(8) :: hil_size, total_timesteps
  integer(8) :: i, j, k, batch_beginning
  integer(8) :: MAX_NUM_OF_ELEMENTS
  integer(8) :: TRAINING_SET_SIZE, VALIDATION_SET_SIZE
  complex(8) :: imaginary
  real(8) :: t
  character(len=:), allocatable :: filename

  ! Get the arguments passed in the command line
  CALL get_command_argument(1, N_string)
  CALL get_command_argument(2, timestep_string)
  CALL get_command_argument(3, total_time_string)
  CALL get_command_argument(4, set_number_string)

  print *, "total_time_string: ", total_time_string
  print *, "set_number_string: ", set_number_string

  ! Then convert them to reals
  READ(N_string,*) N
  READ(timestep_string,*) timestep
  READ(total_time_string,*) total_time
  READ(set_number_string,*) set_number

  if (set_number >= 100) then
    print *, "Too many sets! Stopping execution."
    stop
  end if

  MAX_NUM_OF_ELEMENTS = 209715200
  hil_size=2**N
  total_timesteps = total_time / timestep
  TRAINING_SET_SIZE = MAX_NUM_OF_ELEMENTS / (total_timesteps * hil_size * 2)
  VALIDATION_SET_SIZE = MAX_NUM_OF_ELEMENTS / (total_timesteps * hil_size * 2 * 4) ! We want this set to be 4 times smaller, than the training set

  ! We don't want these sets to be too big comparing to the Hilbert space size of the system under study
  if (TRAINING_SET_SIZE > 5*hil_size) then
    TRAINING_SET_SIZE = 5*hil_size
  end if
  if (VALIDATION_SET_SIZE > 2*hil_size) then
    VALIDATION_SET_SIZE = 2*hil_size
  end if

  print *, "total_time: ", total_time
  print *, "timestep: ", timestep
  print *, "total_timesteps: ", total_timesteps

  allocate(hamil_q(1:hil_size,1:hil_size))
  allocate(U(1:hil_size,1:hil_size))
  allocate(eigenvectors_H(1:hil_size,1:hil_size))

  allocate(training_set_input(1:(TRAINING_SET_SIZE*total_timesteps),1:hil_size)) ! This set consists only of linear combinations of eigenvectors of H
  allocate(training_set_output(1:(TRAINING_SET_SIZE*total_timesteps),1:hil_size))

  training_set_input = 0._8
  training_set_output = 0._8

  open(10, file = 'eigenvectors_H.dat', form = 'unformatted')
  read (10) eigenvectors_H
  close(10)

  open(10, file = 'HQ.dat', form = 'unformatted')
  read (10) hamil_q
  close(10)

  imaginary = cmplx(0.0,-1.0)
  U1 = (imaginary*timestep)*hamil_q
  t = 1.0
  U = expm(t, U1)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Generating TRAINING set
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  print *, "GENERATING INITIAL TRAINING SET..."
  call generate_set(eigenvectors_H, training_set_input, TRAINING_SET_SIZE, total_timesteps)
  print *, "DONE!"

  ! Time evolution of linearly combined vectors in TRAINING set
  print *, "EVOLVING TRAINING SET..."
  batch_beginning = 1
  do i = 1, total_timesteps
    print *, "i: ", i, " / ", total_timesteps
    do j = batch_beginning, batch_beginning + TRAINING_SET_SIZE - 1
      ! Multiply U by vector
      do k = 1, hil_size
        training_set_output(j,k) = SUM(U(k,:) * training_set_input(j,:))
      end do
      if (i < total_timesteps) then
        training_set_input(j+TRAINING_SET_SIZE,:) = training_set_output(j,:)
      end if
    end do
    batch_beginning = batch_beginning + TRAINING_SET_SIZE
  end do
  print *, "DONE!"

  ! Reshaping obtained vectors to format understandable by neural network
  ! (2 times longer vectors with only real values)
  allocate(training_set_input_reshaped(1:(TRAINING_SET_SIZE*total_timesteps),1:2*hil_size))
  training_set_input_reshaped = 0._8

  print *, "RESHAPING TRAINING_INPUT SET..."
  do i = 1, TRAINING_SET_SIZE*total_timesteps
    do j = 1, hil_size
      training_set_input_reshaped(i, (2*j)-1) = REAL(training_set_input(i, j))
      training_set_input_reshaped(i, 2*j) = AIMAG(training_set_input(i, j))
    end do
  end do
  print *, "DONE!"

  if (set_number < 10) then
    allocate(character(len=20) :: filename)
    write(filename, "(A15,I1,A4)") "training_input_", set_number, ".dat"
  else if (set_number >= 10 .and. set_number < 100) then
    allocate(character(len=21) :: filename)
    write(filename, "(A15,I2,A4)") "training_input_", set_number, ".dat"
  end if

  ! Save training inputs to a file
  open(10, file = filename, form = 'unformatted')
  write (10) training_set_input_reshaped
  close(10)
  deallocate(filename)

  deallocate(training_set_input)
  deallocate(training_set_input_reshaped)
  allocate(training_set_output_reshaped(1:(TRAINING_SET_SIZE*total_timesteps),1:2*hil_size))
  training_set_output_reshaped = 0._8

  print *, "RESHAPING TRAINING_OUTPUT SET..."
  do i = 1, TRAINING_SET_SIZE*total_timesteps
    do j = 1, hil_size
      training_set_output_reshaped(i, (2*j)-1) = REAL(training_set_output(i, j))
      training_set_output_reshaped(i, 2*j) = AIMAG(training_set_output(i, j))
    end do
  end do
  print *, "DONE!"

  if (set_number < 10) then
    allocate(character(len=21) :: filename)
    write(filename, "(A16,I1,A4)") "training_output_", set_number, ".dat"
  else if (set_number >= 10 .and. set_number < 100) then
    allocate(character(len=22) :: filename)
    write(filename, "(A16,I2,A4)") "training_output_", set_number, ".dat"
  end if

  ! Save training outputs to a file
  open(10, file = filename, form = 'unformatted')
  write (10) training_set_output_reshaped
  close(10)
  deallocate(filename)

  deallocate(training_set_output)
  deallocate(training_set_output_reshaped)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Generating VALIDATION set
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  allocate(validation_set_input(1:(VALIDATION_SET_SIZE*total_timesteps),1:hil_size)) ! This set consists only of linear combinations of eigenvectors of H
  allocate(validation_set_output(1:(VALIDATION_SET_SIZE*total_timesteps),1:hil_size))

  validation_set_input = 0._8
  validation_set_output = 0._8

  print *, "GENERATING INITIAL VALIDATION SET..."
  call generate_set(eigenvectors_H, validation_set_input, VALIDATION_SET_SIZE, total_timesteps)
  print *, "DONE!"

  ! Time evolution of linearly combined vectors in VALIDATION set
  print *, "EVOLVING VALIDATION SET..."
  batch_beginning = 1
  do i = 1, total_timesteps
    print *, "i: ", i, " / ", total_timesteps
    do j = batch_beginning, batch_beginning + VALIDATION_SET_SIZE - 1
      ! Multiply U by vector
      do k = 1, hil_size
        validation_set_output(j,k) = SUM(U(k,:) * validation_set_input(j,:))
      end do
      if (i < total_timesteps) then
        validation_set_input(j+VALIDATION_SET_SIZE,:) = validation_set_output(j,:)
      end if
    end do
    batch_beginning = batch_beginning + VALIDATION_SET_SIZE
  end do
  print *, "DONE!"

  ! Reshaping obtained vectors to format understandable by neural network
  ! (2 times longer vectors with only real values)
  allocate(validation_set_input_reshaped(1:(VALIDATION_SET_SIZE*total_timesteps),1:2*hil_size))
  validation_set_input_reshaped = 0._8

  print *, "RESHAPING VALIDATION_INPUT SET..."
  do i = 1, VALIDATION_SET_SIZE*total_timesteps
    do j = 1, hil_size
      validation_set_input_reshaped(i, (2*j)-1) = REAL(validation_set_input(i, j))
      validation_set_input_reshaped(i, 2*j) = AIMAG(validation_set_input(i, j))
    end do
  end do
  print *, "DONE!"

  if (set_number < 10) then
    allocate(character(len=22) :: filename)
    write(filename, "(A17,I1,A4)") "validation_input_", set_number, ".dat"
  else if (set_number >= 10 .and. set_number < 100) then
    allocate(character(len=23) :: filename)
    write(filename, "(A17,I2,A4)") "validation_input_", set_number, ".dat"
  end if

  ! Save validation inputs to a file
  open(10, file = filename, form = 'unformatted')
  write (10) validation_set_input_reshaped
  close(10)
  deallocate(filename)

  deallocate(validation_set_input)
  deallocate(validation_set_input_reshaped)
  allocate(validation_set_output_reshaped(1:(VALIDATION_SET_SIZE*total_timesteps),1:2*hil_size))
  validation_set_output_reshaped = 0._8

  print *, "RESHAPING VALIDATION_OUTPUT SET..."
  do i = 1, VALIDATION_SET_SIZE*total_timesteps
    do j = 1, hil_size
      validation_set_output_reshaped(i, (2*j)-1) = REAL(validation_set_output(i, j))
      validation_set_output_reshaped(i, 2*j) = AIMAG(validation_set_output(i, j))
    end do
  end do
  print *, "DONE!"

  if (set_number < 10) then
    allocate(character(len=23) :: filename)
    write(filename, "(A18,I1,A4)") "validation_output_", set_number, ".dat"
  else if (set_number >= 10 .and. set_number < 100) then
    allocate(character(len=24) :: filename)
    write(filename, "(A18,I2,A4)") "validation_output_", set_number, ".dat"
  end if

  ! Save validation outputs to a file
  open(10, file = filename, form = 'unformatted')
  write (10) validation_set_output_reshaped
  close(10)
  deallocate(filename)

  deallocate(validation_set_output)
  deallocate(validation_set_output_reshaped)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Generating TESTING set
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (set_number == 0) then
    allocate(testing_set_input(1:(hil_size*total_timesteps),1:hil_size)) ! This set consists only of eigenvectors of H
    allocate(testing_set_output(1:(hil_size*total_timesteps),1:hil_size))

    testing_set_input = 0._8
    testing_set_output = 0._8

    ! Copy eigenvectors to the testing set
    print *, "GENERATING INITIAL TESTING SET..."
    do i = 1, hil_size
      testing_set_input(i, :) = eigenvectors_H(:, i)
    end do
    print *, "DONE!"

    ! Time evolution of eigenvectors in TESTING set
    print *, "EVOLVING TESTING SET..."
    batch_beginning = 1
    do i = 1, total_timesteps
      print *, "i: ", i, " / ", total_timesteps
      do j = batch_beginning, batch_beginning + hil_size - 1
        ! Multiply U by vector
        do k = 1, hil_size
          testing_set_output(j,k) = SUM(U(k,:) * testing_set_input(j,:))
        end do
        if (i < total_timesteps) then
          testing_set_input(j+hil_size,:) = testing_set_output(j,:)
        end if
      end do
      batch_beginning = batch_beginning + hil_size
    end do
    print *, "DONE!"

    ! Reshaping obtained vectors to format understandable by neural network
    ! (2 times longer vectors with only real values)
    allocate(testing_set_input_reshaped(1:(hil_size*total_timesteps),1:2*hil_size))
    testing_set_input_reshaped = 0._8

    print *, "RESHAPING TESTING_INPUT SET..."
    do i = 1, hil_size*total_timesteps
      do j = 1, hil_size
        testing_set_input_reshaped(i, (2*j)-1) = REAL(testing_set_input(i, j))
        testing_set_input_reshaped(i, 2*j) = AIMAG(testing_set_input(i, j))
      end do
    end do
    print *, "DONE!"

    ! Save testing inputs to a file
    open(10, file = 'testing_input.dat', form = 'unformatted')
    write (10) testing_set_input_reshaped
    close(10)

    deallocate(testing_set_input)
    deallocate(testing_set_input_reshaped)
    allocate(testing_set_output_reshaped(1:(hil_size*total_timesteps),1:2*hil_size))
    testing_set_output_reshaped = 0._8

    print *, "RESHAPING TESTING_OUTPUT SET..."
    do i = 1, hil_size*total_timesteps
      do j = 1, hil_size
        testing_set_output_reshaped(i, (2*j)-1) = REAL(testing_set_output(i, j))
        testing_set_output_reshaped(i, 2*j) = AIMAG(testing_set_output(i, j))
      end do
    end do
    print *, "DONE!"

    ! Save testing outputs to a file
    open(10, file = 'testing_output.dat', form = 'unformatted')
    write (10) testing_set_output_reshaped
    close(10)

    deallocate(testing_set_output)
    deallocate(testing_set_output_reshaped)
  end if

end
