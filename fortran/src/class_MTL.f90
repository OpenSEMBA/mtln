module class_MTL
    use utils, only: linspace

    implicit none
    private

    type, public :: Probe
        real :: position
        integer :: conductor
        character(len=7) :: type
        real, dimension(:), allocatable :: t, v, i
    end type Probe

    type, public :: MTL
        integer :: nx, number_of_conductors
        real :: time, timestep
        real :: length
        real, dimension(:), allocatable :: x
        real, dimension(:,:), allocatable :: l, c
        real, dimension(:,:), allocatable :: Zs, Zl
        real, dimension(:,:), allocatable :: v, i
        type(Probe), dimension(:), allocatable :: probes

        ! allocate(x(nx + 1))
        ! call linspace(from=0, to=length, x)

    ! contains
    !     procedure :: init => mtl_init
    
    end type MTL

! contains
!     subroutine mtl_init(this) result(area)
!         class(MTL), intent(in) :: this
!         real :: area
!         area = 1
!     end subroutine mtl_init



end module class_MTL