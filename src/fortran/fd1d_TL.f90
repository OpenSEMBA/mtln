module fd1d_tl
    implicit none   
    
    real :: deltaZ, deltaT
    real :: pi = 4.d0*datan(1.d0)
 
contains 
    subroutine TL_2conductors(zSteps, tSteps, l, c)    
    
    integer :: zSteps, tSteps, n, k
    real, allocatable :: voltage(:), current(:)
    real :: l, c

    real :: vSource, vLoad, rSource, rLoad
 
    integer :: io
    
    allocate(voltage(0:zSteps))
    allocate(current(0:zSteps-1))
    
    
    deltaZ = 1e-3
    deltaT = 0.95*deltaZ*sqrt(l*c)
    
    vSource = 25
    vLoad = 0
    rSource = sqrt(l/c)
    rLoad = sqrt(l/c)
    
    voltage = 0
    current = 0
    
    do n = 1, tSteps
        
        voltage(0) = (1/(deltaZ*rSource*c/deltaT+1))*((deltaZ*rSource*c/deltaT-1)*voltage(0) - 2*rSource*current(0) + 2*vSource)

        do k = 1, zSteps-1
            voltage(k) = voltage(k) - (deltaT/(deltaZ*c)) * (current(k)-current(k-1))
        end do

        voltage(zSteps) = (1/(deltaZ*rLoad*c/deltaT+1))*((deltaZ*rLoad*c/deltaT-1)*voltage(zSteps) + 2*rLoad*current(zSteps-1) + 2*vLoad)

        
        do k = 0, zSteps-1
            current(k) = current(k) - (deltaT/(deltaZ*l)) * (voltage(k+1) - voltage(k))
        end do

    enddo
    
    open(newunit=io, file="../../../logs/f_output.txt", action="write")
    do k = 0, zSteps-1
        write(io, *) voltage(k), " ", current(k)
    enddo
    close(io)
    
    end subroutine TL_2conductors
    
end module fd1d_tl