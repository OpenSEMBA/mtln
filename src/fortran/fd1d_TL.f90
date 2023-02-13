module fd1d_tl
    implicit none   
    
    real :: deltaZ, deltaT
    real :: pi = 4.d0*datan(1.d0)
 
contains 
    subroutine TL_2conductors(zSteps, tSteps, l, c)    
    
    integer :: zSteps, tSteps, n, k
    real, dimension(:), allocatable :: voltage, current
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
        enddo

        voltage(zSteps) = (1/(deltaZ*rLoad*c/deltaT+1))*((deltaZ*rLoad*c/deltaT-1)*voltage(zSteps) + 2*rLoad*current(zSteps-1) + 2*vLoad)

        
        do k = 0, zSteps-1
            current(k) = current(k) - (deltaT/(deltaZ*l)) * (voltage(k+1) - voltage(k))
        enddo

    enddo
    
    open(newunit=io, file="../../../logs/f_output.txt", action="write")
    do k = 0, zSteps-1
        write(io, *) voltage(k), " ", current(k)
    enddo
    close(io)
    
    end subroutine TL_2conductors
    
!   Time-domain TL solver for N conductors 
    subroutine TL_Nconductors(zSteps, tSteps, l, c)    
    use matrixUtils
    
    real, intent(in), dimension(:,:)  :: l, c
    integer :: zSteps, tSteps, n, k, dim, i
    real, dimension(:,:), allocatable :: voltage, current
    real, dimension(:,:), allocatable :: lInv, cInv, Id
    real, dimension(:,:), allocatable :: A1, A2, B1, B2
    
    real, dimension(:), allocatable :: vSource, vLoad
    real, dimension(:,:), allocatable :: rSource, rLoad
    
    real :: maxV = 0.0, lineVel = 0.0
    
    integer :: io
    
    dim = size(l,1)
    
    allocate(voltage(dim,0:zSteps))
    allocate(current(dim,0:zSteps-1))
    allocate(lInv(dim,dim))
    allocate(cInv(dim,dim))

    allocate(A1(dim,dim))
    allocate(A2(dim,dim))
    allocate(B1(dim,dim))
    allocate(B2(dim,dim))

    allocate(Id(dim, dim))
    Id = 0
    forall(i = 1:dim) Id(i,i) = 1
    
    allocate(vSource(dim))
    allocate(vLoad(dim))
    allocate(rSource(dim,dim))
    allocate(rLoad(dim,dim))
    
    vSource(:) = 25
    vLoad(:) = 0
    rSource = 0
    rLoad = 0
    
    do i = 1, dim
        lineVel = 1.0/sqrt(l(i,i)*c(i,i))
        if (lineVel > maxV) then
            maxV = lineVel
        endif
        
        rSource(i,i) = sqrt(l(i,i)/c(i,i))
        rLoad(i,i)   = sqrt(l(i,i)/c(i,i))

    enddo
    
    deltaZ = 1e-3
    deltaT = 0.95*deltaZ/maxV
    
    lInv = inv(l)
    cInv = inv(c)
   
    A1 = deltaZ*matmul(rSource, c)/deltaT - Id
    A2 = deltaZ*matmul(rLoad, c)/deltaT - Id
    B1 = inv(deltaZ*matmul(rSource, c)/deltaT+Id)
    B2 = inv(deltaZ*matmul(rLoad, c)/deltaT+Id)
    
    voltage(:,:) = 0
    current(:,:) = 0
    
    do n = 1, tSteps
        
        voltage(:,0) = matmul(B1, matmul(A1, voltage(:,0)) - 2*matmul(rSource, current(:,0)) + 2*vSource)

        do k = 1, zSteps-1
            voltage(:,k) = voltage(:,k) - (deltaT/deltaZ)*matmul(cInv,current(:,k)-current(:,k-1))
        enddo

        voltage(:,zSteps) = matmul(B2, matmul(A2,voltage(:,zSteps)) +2*matmul(rLoad, current(:,zSteps-1)) + 2*vLoad)

        
        do k = 0, zSteps-1
            current(:,k) = current(:,k) - (deltaT/deltaZ)*matmul(lInv, voltage(:,k+1) - voltage(:,k))
        enddo

    enddo
    
    open(newunit=io, file="../../../logs/3conductors_output.txt", action="write")
    do k = 0, zSteps-1
        do i = 1,2
            write (io, "(4f20.10)", advance="no") voltage(i, k), current(i, k)
        enddo
        write(io, *)
    enddo
    close(io)
    
    end subroutine TL_Nconductors
    
end module fd1d_tl