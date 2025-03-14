//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>
// UNSUPPORTED: c++03, c++11, c++14

// template<class Container>
//   stack(Container) -> stack<typename Container::value_type, Container>;
//
// template<class Container, class Allocator>
//   stack(Container, Allocator) -> stack<typename Container::value_type, Container>;
//
// template<ranges::input_range R>
//   stack(from_range_t, R&&) -> stack<ranges::range_value_t<R>>; // since C++23
//
// template<ranges::input_range R, class Allocator>
//   stack(from_range_t, R&&, Allocator)
//     -> stack<ranges::range_value_t<R>, deque<ranges::range_value_t<R>, Allocator>>; // since C++23

#include <array>
#include <stack>
#include <deque>
#include <vector>
#include <list>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <climits> // INT_MAX

#include "deduction_guides_sfinae_checks.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

int main(int, char**) {
  //  Test the explicit deduction guides
  {
    std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::stack stk(v);

    static_assert(std::is_same_v<decltype(stk), std::stack<int, std::vector<int>>>, "");
    assert(stk.size() == v.size());
    assert(stk.top() == v.back());
  }

  {
    std::list<long, test_allocator<long>> l{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::stack stk(l, test_allocator<long>(0, 2)); // different allocator
    static_assert(std::is_same_v<decltype(stk)::container_type, std::list<long, test_allocator<long>>>, "");
    static_assert(std::is_same_v<decltype(stk)::value_type, long>, "");
    assert(stk.size() == 10);
    assert(stk.top() == 19);
    //  I'd like to assert that we've gotten the right allocator in the stack, but
    //  I don't know how to get at the underlying container.
  }

  //  Test the implicit deduction guides

  {
    //  We don't expect this one to work - no way to implicitly get value_type
    //  std::stack stk(std::allocator<int>()); // stack (allocator &)
  }

  {
    std::stack<A> source;
    std::stack stk(source); // stack(stack &)
    static_assert(std::is_same_v<decltype(stk)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(stk)::container_type, std::deque<A>>, "");
    assert(stk.size() == 0);
  }

  {
    typedef short T;
    typedef test_allocator<T> Alloc;
    typedef std::list<T, Alloc> Cont;
    typedef test_allocator<int> ConvertibleToAlloc;
    static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                  !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

    {
      Cont cont;
      std::stack stk(cont, Alloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      Cont cont;
      std::stack stk(cont, ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      Cont cont;
      std::stack stk(std::move(cont), Alloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      Cont cont;
      std::stack stk(std::move(cont), ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }
  }

  {
    typedef short T;
    typedef test_allocator<T> Alloc;
    typedef std::list<T, Alloc> Cont;
    typedef test_allocator<int> ConvertibleToAlloc;
    static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                  !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

    {
      std::stack<T, Cont> source;
      std::stack stk(source, Alloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      std::stack<T, Cont> source;
      std::stack stk(source, ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      std::stack<T, Cont> source;
      std::stack stk(std::move(source), Alloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }

    {
      std::stack<T, Cont> source;
      std::stack stk(std::move(source), ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(stk), std::stack<T, Cont>>);
    }
  }

#if TEST_STD_VER >= 23
  {
    typedef short T;
    typedef test_allocator<T> Alloc;
    std::list<T> a;
    {
      std::stack s(a.begin(), a.end());
      static_assert(std::is_same_v<decltype(s), std::stack<T>>);
    }
    {
      std::stack s(a.begin(), a.end(), Alloc());
      static_assert(std::is_same_v<decltype(s), std::stack<T, std::deque<T, Alloc>>>);
    }
  }

  {
    {
      std::stack c(std::from_range, std::array<int, 0>());
      static_assert(std::is_same_v<decltype(c), std::stack<int>>);
    }

    {
      using Alloc = test_allocator<int>;
      std::stack c(std::from_range, std::array<int, 0>(), Alloc());
      static_assert(std::is_same_v<decltype(c), std::stack<int, std::deque<int, Alloc>>>);
    }
  }
#endif

  ContainerAdaptorDeductionGuidesSfinaeAway<std::stack, std::stack<int>>();

  return 0;
}
