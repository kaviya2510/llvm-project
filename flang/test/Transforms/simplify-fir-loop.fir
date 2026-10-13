// Test forwarding a DO loop's induction variable past its memory slot.

// RUN: fir-opt %s --simplify-fir-loop | FileCheck %s

// The spill and the reload of the loop variable both go away, and the body
// refers to the induction variable instead. Nothing reads the slot afterwards,
// so the store holding the value the variable has after the loop goes too.
// CHECK-LABEL: func.func @forward_iv
// CHECK-NOT:     fir.store
// CHECK-NOT:     fir.load %{{[0-9]+}} : !fir.ref<i32>
// CHECK:         fir.do_loop %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:      fir.convert %[[IV]] : (i32) -> i64
func.func @forward_iv(%n: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %n : !fir.ref<i32>
  fir.do_loop %iv = %c1_i32 to %2 step %c1_i32 : i32 {
    fir.store %iv to %1 : !fir.ref<i32>
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
  }
  %5 = fir.convert %2 : (i32) -> index
  %6 = arith.addi %5, %c1 : index
  %7 = fir.convert %6 : (index) -> i32
  fir.store %7 to %1 : !fir.ref<i32>
  return
}

// Both levels of a nest are forwarded, including the inner variable's final
// store that lowering places in the outer loop's body. Only pure arithmetic is
// left between the inner loop and the end of the outer body, so a later DCE
// leaves the nest perfectly nested.
// CHECK-LABEL: func.func @forward_nest
// CHECK-NOT:     fir.store
// CHECK:         fir.do_loop %[[J:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:      fir.do_loop %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:        fir.convert %[[I]] : (i32) -> i64
// CHECK-NEXT:        fir.convert %[[J]] : (i32) -> i64
// CHECK-NEXT:      }
// CHECK-NOT:     fir.store
func.func @forward_nest(%n: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.alloca i32 <{bindc_name = "j", uniq_name = "j"}>
  %3 = fir.declare %2 uniq_name("j") : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.load %n : !fir.ref<i32>
  fir.do_loop %j = %c1_i32 to %4 step %c1_i32 : i32 {
    fir.store %j to %3 : !fir.ref<i32>
    fir.do_loop %i = %c1_i32 to %4 step %c1_i32 : i32 {
      fir.store %i to %1 : !fir.ref<i32>
      %5 = fir.load %1 : !fir.ref<i32>
      %6 = fir.convert %5 : (i32) -> i64
      %7 = fir.load %3 : !fir.ref<i32>
      %8 = fir.convert %7 : (i32) -> i64
    }
    %9 = fir.convert %4 : (i32) -> index
    %10 = arith.addi %9, %c1 : index
    %11 = fir.convert %10 : (index) -> i32
    fir.store %11 to %1 : !fir.ref<i32>
  }
  return
}

// A read that still has a reader keeps its stores, so the value the variable
// holds after the loop stays observable.
// CHECK-LABEL: func.func @keep_final_store
// CHECK:         fir.do_loop
// CHECK:         fir.store %{{.+}} to %[[SLOT:.+]] : !fir.ref<i32>
// CHECK-NEXT:    fir.load %[[SLOT]] : !fir.ref<i32>
func.func @keep_final_store(%n: !fir.ref<i32>) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %n : !fir.ref<i32>
  fir.do_loop %iv = %c1_i32 to %2 step %c1_i32 : i32 {
    fir.store %iv to %1 : !fir.ref<i32>
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
  }
  %5 = fir.convert %2 : (i32) -> index
  %6 = arith.addi %5, %c1 : index
  %7 = fir.convert %6 : (index) -> i32
  fir.store %7 to %1 : !fir.ref<i32>
  %8 = fir.load %1 : !fir.ref<i32>
  return %8 : i32
}

// A second write inside the loop means a read need not see the induction
// variable, so the spill stays.
// CHECK-LABEL: func.func @reject_second_store
// CHECK:         fir.do_loop %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:      fir.store %[[IV]] to
// CHECK:           fir.store %c0_i32 to
// CHECK:           fir.load
func.func @reject_second_store(%n: !fir.ref<i32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %n : !fir.ref<i32>
  fir.do_loop %iv = %c1_i32 to %2 step %c1_i32 : i32 {
    fir.store %iv to %1 : !fir.ref<i32>
    fir.store %c0_i32 to %1 : !fir.ref<i32>
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
  }
  return
}

// A read the spill does not dominate observes an earlier iteration, so the
// spill stays.
// CHECK-LABEL: func.func @reject_read_before_spill
// CHECK:         fir.do_loop %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:      fir.load
// CHECK:           fir.store %[[IV]] to
func.func @reject_read_before_spill(%n: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %n : !fir.ref<i32>
  fir.do_loop %iv = %c1_i32 to %2 step %c1_i32 : i32 {
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
    fir.store %iv to %1 : !fir.ref<i32>
  }
  return
}

// The DO statement names a dummy argument. Reads inside the loop are still
// forwarded, but the caller observes the variable after the call, so the store
// describing the value it holds once the loop finishes has to stay.
// CHECK-LABEL: func.func @keep_dummy_final_store
// CHECK:         fir.do_loop %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NOT:       fir.store
// CHECK:           fir.convert %[[IV]]
// CHECK:         }
// CHECK:         fir.store %{{.+}} to %[[SLOT:.+]] : !fir.ref<i32>
func.func @keep_dummy_final_store(%i: !fir.ref<i32>, %n: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.declare %i uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.load %n : !fir.ref<i32>
  fir.do_loop %iv = %c1_i32 to %1 step %c1_i32 : i32 {
    fir.store %iv to %0 : !fir.ref<i32>
    %2 = fir.load %0 : !fir.ref<i32>
    %3 = fir.convert %2 : (i32) -> i64
  }
  %4 = arith.addi %1, %c1_i32 : i32
  fir.store %4 to %0 : !fir.ref<i32>
  return
}

// The slot's address escapes, so a reader this pass cannot see may observe it
// and the spill stays.
// CHECK-LABEL: func.func @reject_escaping_slot
// CHECK:         fir.do_loop %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} : i32 {
// CHECK-NEXT:      fir.store %[[IV]] to
// CHECK:           fir.load
func.func @reject_escaping_slot(%n: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 <{bindc_name = "i", uniq_name = "i"}>
  %1 = fir.declare %0 uniq_name("i") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %n : !fir.ref<i32>
  fir.call @escape(%1) : (!fir.ref<i32>) -> ()
  fir.do_loop %iv = %c1_i32 to %2 step %c1_i32 : i32 {
    fir.store %iv to %1 : !fir.ref<i32>
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
  }
  return
}
func.func private @escape(!fir.ref<i32>)
