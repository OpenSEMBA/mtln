module matrixUtils
implicit none

contains    
    function inv(A) result(Ainv)
    real,intent(in) :: A(:,:)
    real            :: Ainv(size(A,1),size(A,2))
    real            :: work(size(A,1))            ! work array for LAPACK
    integer         :: n,info,ipiv(size(A,1))     ! pivot indices

    ! Store A in Ainv to prevent it from being overwritten by LAPACK
    Ainv = A
    n = size(A,1)
    ! SGETRF computes an LU factorization of a general M-by-N matrix A
    ! using partial pivoting with row interchanges.
    call SGETRF(n,n,Ainv,n,ipiv,info)
    if (info.ne.0) stop 'Matrix is numerically singular!'
    ! SGETRI computes the inverse of a matrix using the LU factorization
    ! computed by SGETRF.
    call SGETRI(n,Ainv,n,ipiv,work,n,info)
    if (info.ne.0) stop 'Matrix inversion failed!'
    end function inv
    
    
    subroutine getUpdateMatrices(M, N, dZ, dT, U1, U2)
        real, intent(in), dimension(:,:)  :: M, N
        real, intent(in) :: dZ, dT
        real, dimension(:,:), allocatable :: A
        real, intent(out), dimension(:,:), allocatable :: U1, U2
        integer :: dim
        
        dim = size(M,1)
        allocate(U1(dim,dim))
        allocate(U2(dim,dim))
        allocate(A(dim,dim))
        
        U1 = inv(dZ*M/dT + dZ*N/2)
        A = dZ*M/dT - dZ*N/2
        U2 = matmul(U1,A)
    
    end subroutine getUpdateMatrices
    
    subroutine getTerminalMatrices(M, N, R, dz, dT, T1, T2)
        real, intent(in), dimension(:,:) :: M, N, R
        real, intent(in) :: dZ, dT
        real, dimension(:,:), allocatable :: Id
        real, intent(out), dimension(:,:), allocatable :: T1, T2
        integer :: dim, i
        
        dim = size(M,1)
        allocate(T1(dim,dim))
        allocate(T2(dim,dim))
        allocate(Id(dim, dim))
        Id = 0
        forall(i = 1:dim) Id(i,i) = 1
        
        T1 = inv((dZ/dT)*matmul(R, M) + (dZ/2)*matmul(R, N) + Id)
        T2 = (dZ/dT)*matmul(R,M) - (dZ/2)*matmul(R, N) - Id
        
    end subroutine getTerminalMatrices
    

        
    
end module matrixUtils