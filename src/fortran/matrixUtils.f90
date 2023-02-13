module matrixUtils

contains    
    function inv(A) result(Ainv)
    implicit none
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
    
end module matrixUtils