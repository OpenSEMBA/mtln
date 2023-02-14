program coaxial
    use fd1d_tl
    implicit none
    
    integer :: numArgs
    character(100) :: arg1,  arg2
    
    integer :: zSteps , tSteps
    real :: shieldR, wireR
    real :: eps, mu
    real, dimension :: l,c
    
    numArgs = command_argument_count()
    call get_command_argument(1,arg1) 
    call get_command_argument(2,arg2) 
    read(arg1,*) zSteps
    read(arg2,*) tSteps
    
    shieldR = 5e-3
    wireR = 1e-3
    eps = 8.85e-12
    mu = 4*pi*1e-7

    !pul L and C for coaxial given shield and wire radius
    shieldR = 5e-3
    wireR = 1e-3
    l = (mu/(2*pi))*log(shieldR/wireR)
    c = 2*pi*eps/log(shieldR/wireR)

    call TL_2conductors(zSteps, tSteps, l, c)
    
    
end program coaxial