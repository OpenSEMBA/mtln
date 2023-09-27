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

        voltage(zSteps) = (1/(deltaZ*rLoad*c/deltaT+1)) &
                          *((deltaZ*rLoad*c/deltaT-1)*voltage(zSteps) &
                          + 2*rLoad*current(zSteps-1) + 2*vLoad)

        
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
    
!   Time-domain TL solver for N conductors without losses
    subroutine TL_Nconductors(zSteps, tSteps, l, c, voltage, current)    
    use matrixUtils
    
    real, intent(in), dimension(:,:)  :: l, c
    real, dimension(:,:), allocatable :: voltage, current
    integer :: zSteps, tSteps, n, k, dim, i
    real, dimension(:,:), allocatable :: lInv, cInv, Id
    real, dimension(:,:), allocatable :: A1, A2, B1, B2
    
    real, dimension(:), allocatable :: vSource, vLoad
    real, dimension(:,:), allocatable :: rSource, rLoad
    
    real :: maxV = 0.0, lineVel = 0.0
    
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
  
    end subroutine TL_Nconductors

    !   Time-domain TL solver for N conductors with losses
    subroutine TL_Nconductors_losses(zSteps, tSteps, l, c, r, g, voltage, current)    
    use matrixUtils
    
    real, intent(in), dimension(:,:)  :: l, c, r, g
    real, dimension(:,:), allocatable :: voltage, current
    integer :: zSteps, tSteps, n, k, dim, i
    real, dimension(:,:), allocatable :: lInv, cInv, Id
    !real, dimension(:,:), allocatable :: A1, A2, B1, B2
    real, dimension(:,:), allocatable :: S1, S2, L1, L2, CG1, CG2, LR1, LR2
    
    real, dimension(:), allocatable :: vSource, vLoad
    real, dimension(:,:), allocatable :: rSource, rLoad
    
    real :: maxV = 0.0, lineVel = 0.0
    
    dim = size(l,1)
    
    allocate(voltage(dim,0:zSteps))
    allocate(current(dim,0:zSteps-1))
    
    !allocate(lInv(dim,dim))
    !allocate(cInv(dim,dim))
    
    allocate(S1(dim,dim))
    allocate(S2(dim,dim))
    allocate(L1(dim,dim))
    allocate(L2(dim,dim))
    allocate(CG1(dim,dim))
    allocate(CG2(dim,dim))
    allocate(LR1(dim,dim))
    allocate(LR2(dim,dim))
    !allocate(A1(dim,dim))
    !allocate(A2(dim,dim))
    !allocate(B1(dim,dim))
    !allocate(B2(dim,dim))
    S1 = 0
    S2 = 0
    L1 = 0
    L2 = 0
    CG1 = 0
    CG2 = 0
    LR1 = 0
    LR2 = 0
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
    
    call getTerminalMatrices(c, g, rSource, deltaZ, deltaT, S1, S2)
    call getTerminalMatrices(c, g, rLoad, deltaZ, deltaT, L1, L2)
    
    call getUpdateMatrices(c, g, deltaZ, deltaT, CG1, CG2)
    call getUpdateMatrices(l, r, deltaZ, deltaT, LR1, LR2)
    
    !lInv = inv(l)
    !cInv = inv(c)
    !
    !A1 = deltaZ*matmul(rSource, c)/deltaT - Id
    !A2 = deltaZ*matmul(rLoad, c)/deltaT - Id
    !B1 = inv(deltaZ*matmul(rSource, c)/deltaT+Id)
    !B2 = inv(deltaZ*matmul(rLoad, c)/deltaT+Id)
    
    voltage(:,:) = 0
    current(:,:) = 0
    
    do n = 1, tSteps
        
        !voltage(:,0) = matmul(B1, matmul(A1, voltage(:,0)) - 2*matmul(rSource, current(:,0)) + 2*vSource)
    
        voltage(:,0) = matmul(S1, matmul(S2, voltage(:,0)) - 2*matmul(rSource, current(:,0)) + 2*vSource)
        
        do k = 1, zSteps-1
            !voltage(:,k) = voltage(:,k) - (deltaT/deltaZ)*matmul(cInv,current(:,k)-current(:,k-1))
            voltage(:,k) = matmul(CG2, voltage(:,k)) - matmul(CG1,current(:,k)-current(:,k-1))
        enddo
    
        voltage(:,zSteps) = matmul(L1, matmul(L2, voltage(:,zSteps)) + 2*matmul(rLoad, current(:,zSteps-1)) + 2*vLoad)
        !voltage(:,zSteps) = matmul(B2, matmul(A2,voltage(:,zSteps)) +2*matmul(rLoad, current(:,zSteps-1)) + 2*vLoad)
    
        
        do k = 0, zSteps-1
            current(:,k) = matmul(LR2, current(:,k)) - matmul(LR1, voltage(:,k+1) - voltage(:,k))
            !current(:,k) = current(:,k) - (deltaT/deltaZ)*matmul(lInv, voltage(:,k+1) - voltage(:,k))
        enddo
    
    enddo
    
    end subroutine TL_Nconductors_losses
    
end module fd1d_tl