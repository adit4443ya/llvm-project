! LINEAR on an allocatable: privatized via omp.private; the linear value is
! copied into the private copy at the start of each iteration and back at the
! end.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: omp.private {type = private} @[[PRIV_I32:.*simd_linear_allocatableEa_private_box_heap_i32]] : !fir.box<!fir.heap<i32>> init {
! CHECK:   fir.allocmem i32
! CHECK: } dealloc {
! CHECK:   fir.freemem

! CHECK-LABEL: func.func @_QPsimd_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         omp.simd linear(%[[ADDR]] : !fir.heap<i32> = %{{.*}} : i32
! CHECK-SAME:    linear_var_types([i32
! CHECK-SAME:    private(@[[PRIV_I32]] %[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           omp.loop_nest
! CHECK:             %[[PRIV_A:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFsimd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:             %[[LIN:.*]] = fir.load %[[ADDR]] : !fir.heap<i32>
! CHECK:             hlfir.assign %[[LIN]] to %[[PRIV_A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             hlfir.assign %{{.*}} to %[[PRIV_A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             %[[PBOX:.*]] = fir.load %[[PRIV_A]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:             %[[PADDR:.*]] = fir.box_addr %[[PBOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:             %[[PVAL:.*]] = fir.load %[[PADDR]] : !fir.heap<i32>
! CHECK:             fir.store %[[PVAL]] to %[[ADDR]] : !fir.heap<i32>
! CHECK-NEXT:        omp.yield
subroutine simd_linear_allocatable()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp simd linear(a)
  do i = 1, 2
    a = 2
  end do
  !$omp end simd
end subroutine

! CHECK-LABEL: func.func @_QPdo_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         omp.parallel
! CHECK:           %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           omp.wsloop linear(%[[ADDR]] : !fir.heap<i32> = %{{.*}} : i32
! CHECK-SAME:      private(@{{.*}}do_linear_allocatableEa_private_box_heap_i32 %[[A]]#0 -> %[[ARG:[^,]*]],
! CHECK:             omp.loop_nest
! CHECK:               %[[PRIV_A:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdo_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:               %[[LIN:.*]] = fir.load %[[ADDR]] : !fir.heap<i32>
! CHECK:               hlfir.assign %[[LIN]] to %[[PRIV_A]]#0 realloc
! CHECK:               fir.store %{{.*}} to %[[ADDR]] : !fir.heap<i32>
! CHECK-NEXT:          omp.yield
subroutine do_linear_allocatable()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp parallel do linear(a:2)
  do i = 1, 10
    a = a + 2
  end do
  !$omp end parallel do
end subroutine

! Composite: the linear operand moves to omp.wsloop, the private copy is on
! omp.simd.
! CHECK-LABEL: func.func @_QPdo_simd_linear_allocatable
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_simd_linear_allocatableEa") fortran_attrs<allocatable>
! CHECK:         %[[ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:         omp.wsloop linear(%[[ADDR]] : !fir.heap<i64> = %{{.*}} : i64
! CHECK:           omp.simd
! CHECK-SAME:      private(@{{.*}}do_simd_linear_allocatableEa_private_box_heap_i64 %[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.box<!fir.heap<i64>>>)
! CHECK:             omp.loop_nest
! CHECK:               %[[PRIV_A:.*]]:2 = hlfir.declare %[[ARG]]
! CHECK:               %[[LIN:.*]] = fir.load %[[ADDR]] : !fir.heap<i64>
! CHECK:               hlfir.assign %[[LIN]] to %[[PRIV_A]]#0 realloc : i64
! CHECK:               fir.store %{{.*}} to %[[ADDR]] : !fir.heap<i64>
! CHECK-NEXT:          omp.yield
subroutine do_simd_linear_allocatable()
  integer(8), allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp do simd linear(a)
  do i = 1, 10
    a = a + 1
  end do
  !$omp end do simd
end subroutine

! The original descriptor is used after the loop.
! CHECK-LABEL: func.func @_QPoriginal_used_after_loop
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QForiginal_used_after_loopEa") fortran_attrs<allocatable>
! CHECK:         omp.simd linear
! CHECK:           omp.yield
! CHECK:         }
! CHECK:         hlfir.assign %{{.*}} to %[[A]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
subroutine original_used_after_loop()
  integer, allocatable :: a
  integer :: i
  allocate(a)
  a = 0
  !$omp simd linear(a)
  do i = 1, 2
    a = 2
  end do
  !$omp end simd
  a = 5
end subroutine
