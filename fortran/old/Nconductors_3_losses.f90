program Nconductors_3
    use fd1d_tl
    use matrixUtils
    implicit none
    
    integer :: numArgs
    character(100) :: arg1,  arg2, arg3
    
    integer :: zSteps, tSteps, losses
    real :: shieldR = 5e-3, wireR = 1e-3, dToCenter = 2e-3
    real :: eps, mu, sigma
    real, dimension(2,2) :: l, c, r, g
    integer :: i,k, dim
    real, dimension(:,:), allocatable :: voltage, current 
    integer :: io
    
    numArgs = command_argument_count()
    call get_command_argument(1,arg1) 
    call get_command_argument(2,arg2) 
    call get_command_argument(3,arg3) 
    read(arg1,*) zSteps
    read(arg2,*) tSteps
    read(arg3,*) losses

    !zSteps = 200
    !tSteps = 1000
    
    dim = size(l,1)
    eps = 8.85e-12
    mu = 4*pi*1e-7
    sigma = 1e-2
    
    l = 0
    !pul L and C for coaxial given shield and wire radius
    l(1,1) = (mu/(2*pi))*log((shieldR**2-dToCenter**2)/(shieldR*wireR))
    l(2,2) = l(1,1)
    l(1,2) = (mu/(2*pi))*log((dToCenter/shieldR)&
                        *sqrt((dToCenter**4+shieldR**4+2*dToCenter**2*shieldR**2)&
                             /(dToCenter**4+dToCenter**4+2*dToCenter**4)))
    l(2,1) = l(1,2)

    c = mu*eps*inv(l)

    r = 0
    g = 0
    if (losses == 1) then
        g = (sigma/eps)*c
        r(1,1) = 100
        r(2,2) = 100
    end if
    
    call TL_Nconductors_losses(zSteps, tSteps, l, c, r, g, voltage, current)
    open(newunit=io, file="../../../logs/3conductors_output_losses.txt", action="write")
    do k = 0, zSteps-1
        do i = 1, dim
            write (io, "(4f20.10)", advance="no") voltage(i, k), current(i, k)
        enddo
        write(io, *)
    enddo
    close(io)

    
end program Nconductors_3 