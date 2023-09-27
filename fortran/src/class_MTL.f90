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


     contains
        procedure :: init => mtl_init
        procedure :: get_max_timestep => mtl_get_max_timestep
        procedure :: get_phase_velocities => mtl_get_phase_velocities
    
    end type MTL

    contains
        function mtl_get_phase_velocities(this) result(phase_velocities)
            class(MTL), intent(in) :: this
            real, dimension(:), allocatable :: phase_velocities

            allocate(phase_velocities(2))
           

        end function mtl_get_phase_velocities

        function mtl_get_max_timestep(this) result(max_timestep)
            class(MTL), intent(in) :: this
            real :: dx, max_timestep
            dx = this%x(1) - this%x(0)
            max_timestep = dx/MAXVAL(this%get_phase_velocities())
            

        end function mtl_get_max_timestep


        subroutine mtl_init(this)
            class(MTL), intent(inout) :: this

            allocate(this%x(this%nx + 1))
            call linspace(from=0.0, to=this%length, array = this%x)

            allocate(this%v(this%number_of_conductors,size(this%x)))
            allocate(this%i(this%number_of_conductors,size(this%x)-1))


        end subroutine   mtl_init



end module class_MTL