! Test cases that violate OpenMP 5.2 standards
! These should produce semantic errors in Flang

!===-------------------------------------------------------------------===!
! Test 1: Linear clause stride must be integer constant (OpenMP 5.2 §2.9.2)
!===-------------------------------------------------------------------===!
subroutine test_linear_non_integer_stride()
  integer :: i
  real :: stride_val = 2.5
  
  !$omp simd linear(i:stride_val)  ! ERROR: stride must be integer constant
  do i = 1, 10
  end do
end subroutine

!===-------------------------------------------------------------------===!
! Test 2: COPYPRIVATE and NOWAIT are mutually exclusive (OpenMP 5.2 §2.19.6.3)
!===-------------------------------------------------------------------===!
subroutine test_copyprivate_nowait()
  integer :: x
  
  !$omp parallel
    !$omp single copyprivate(x) nowait  ! ERROR: Cannot use both
    x = 42
    !$omp end single
  !$omp end parallel
end subroutine

!===-------------------------------------------------------------------===!
! Test 3: SAFELEN must be positive (OpenMP 5.2 §2.9.3.8)
!===-------------------------------------------------------------------===!
subroutine test_safelen_invalid()
  integer :: i, arr(100)
  
  !$omp simd safelen(0)    ! ERROR: Must be positive
  do i = 1, 100; arr(i) = i; end do
  
  !$omp simd safelen(-5)   ! ERROR: Must be positive  
  do i = 1, 100; arr(i) = i * 2; end do
end subroutine

!===-------------------------------------------------------------------===!
! Test 4: Workshare requires PURE/ELEMENTAL procedures (OpenMP 5.2 §2.10.3)
!===-------------------------------------------------------------------===!
subroutine impure_sub(x)
  integer :: x
  x = x + 1
end subroutine

subroutine test_workshare()
  integer :: arr(10)
  
  !$omp parallel workshare
    arr = 0
    call impure_sub(arr(1))  ! ERROR: Must be PURE or ELEMENTAL
  !$omp end parallel workshare
end subroutine

!===-------------------------------------------------------------------===!
! Test 5: Cray pointee cannot be in DSA list (OpenMP 5.2 §2.21.1.1)
!===-------------------------------------------------------------------===!
subroutine test_cray_pointee_dsa()
  integer :: target_var
  pointer (ptr, target_var)
  
  !$omp parallel shared(target_var)  ! ERROR: Pointee not allowed in DSA
    target_var = 42
  !$omp end parallel
end subroutine

!===-------------------------------------------------------------------===!
! Test 6: DEFAULT(NONE) with Cray pointer (OpenMP 5.2 §2.21.1.1)
!===-------------------------------------------------------------------===!
subroutine test_cray_default_none()
  integer :: target_var
  pointer (ptr, target_var)
  
  !$omp parallel default(none)  ! ERROR: Must specify pointer, not pointee
    target_var = 42
  !$omp end parallel
end subroutine

!===-------------------------------------------------------------------===!
! Test 7: Atomic with type conversion (OpenMP 5.2 §2.19.7.2)
!===-------------------------------------------------------------------===!
subroutine test_atomic_type_mismatch()
  integer :: int_var
  real :: real_var = 3.14
  
  !$omp atomic write
    int_var = real_var  ! ERROR: Type mismatch
  !$omp end atomic
end subroutine

! Related bugs: #111354, #73486, #109089, #111358, #121028, 
!               #133232, #92346, #108516, #119172, #122097, #73102
