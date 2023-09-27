program Nconductors_2
    use fd1d_tl
    use matrixUtils
    implicit none
    
    integer :: numArgs
    character(100) :: arg1,  arg2
    
    integer :: zSteps , tSteps
    real :: shieldR = 5e-3, wireR = 1e-3, dToCenter = 2e-3
    real :: eps, mu
    real, dimension(1,1) :: l,c
    integer :: i,k, dim
    real, dimension(:,:), allocatable :: voltage, current 
    integer :: io

    numArgs = command_argument_count()
    call get_command_argument(1,arg1) 
    call get_command_argument(2,arg2) 
    read(arg1,*) zSteps
    read(arg2,*) tSteps

    dim = size(l,1)
    eps = 8.85e-12
    mu = 4*pi*1e-7

    !allocate(voltage(dim,0:zSteps))
    !allocate(current(dim,0:zSteps-1))

    l(1,1) = (mu/(2*pi))*log(shieldR/wireR)
    c(1,1) = 2*pi*eps/log(shieldR/wireR)
    !pul L and C for coaxial given shield and wire radius
    call TL_Nconductors(zSteps, tSteps, l, c, voltage, current)
    
    open(newunit=io, file="../../../logs/2conductors_output.txt", action="write")
    do k = 0, zSteps-1
        do i = 1, dim
            write (io, "(4f20.10)", advance="no") voltage(i, k), current(i, k)
        enddo
        write(io, *)
    enddo
    close(io)

    
end program Nconductors_2 