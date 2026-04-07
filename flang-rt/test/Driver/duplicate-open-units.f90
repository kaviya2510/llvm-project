! UNSUPPORTED: offload-cuda
!
! Exercise opening the same external file on two units (runtime behavior).
! Uses a unique filename to avoid clashes with parallel lit runs in the exec root.

! RUN: rm -f test_file.txt
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: File opened successfully on unit 10.
! CHECK: Written content: hello
! CHECK: Attempting to open the same file on a different unit (11)...
! CHECK: File opened successfully on unit 11.
! CHECK: Read content on unit 11: hello
program file_op
  implicit none
  character(len=20) :: line
  character(len=*), parameter :: filename = 'test_file.txt'

  open(unit=10, file=filename, status='replace', action='readwrite')
  print *, 'File opened successfully on unit 10.'
  write(unit=10, fmt='(A)') 'hello'
  flush(unit=10)
  print *, 'Written content: hello'

  print *, 'Attempting to open the same file on a different unit (11)...'
  open(unit=11, file=filename, status='old', action='read')
  print *, 'File opened successfully on unit 11.'
  rewind(unit=11)
  read(unit=11, fmt='(A)') line
  print '(A,A)', 'Read content on unit 11: ', trim(line)

  close(unit=11)
  close(unit=10)
end program file_op
