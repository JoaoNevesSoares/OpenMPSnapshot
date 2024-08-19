
#ifndef ASIO_EXECUTION_SENDER_HPP
#define ASIO_EXECUTION_SENDER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/as_invocable.hpp"
#include "asio/execution/detail/void_receiver.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/receiver.hpp"

#include "asio/detail/push_options.hpp"

#if defined(ASIO_HAS_ALIAS_TEMPLATES) \
&& defined(ASIO_HAS_VARIADIC_TEMPLATES) \
&& defined(ASIO_HAS_DECLTYPE) \
&& !defined(ASIO_MSVC) || (_MSC_VER >= 1910)
# define ASIO_HAS_DEDUCED_EXECUTION_IS_TYPED_SENDER_TRAIT 1
#endif 

namespace asio {
namespace execution {
namespace detail {

namespace sender_base_ns { struct sender_base {}; }

template <typename S, typename = void>
struct sender_traits_base
{
typedef void asio_execution_sender_traits_base_is_unspecialised;
};

template <typename S>
struct sender_traits_base<S,
typename enable_if<
is_base_of<sender_base_ns::sender_base, S>::value
>::type>
{
};

template <typename S, typename = void, typename = void, typename = void>
struct has_sender_types : false_type
{
};

#if defined(ASIO_HAS_DEDUCED_EXECUTION_IS_TYPED_SENDER_TRAIT)

template <
template <
template <typename...> class Tuple,
template <typename...> class Variant
> class>
struct has_value_types
{
typedef void type;
};

template <
template <
template <typename...> class Variant
> class>
struct has_error_types
{
typedef void type;
};

template <typename S>
struct has_sender_types<S,
typename has_value_types<S::template value_types>::type,
typename has_error_types<S::template error_types>::type,
typename conditional<S::sends_done, void, void>::type> : true_type
{
};

template <typename S>
struct sender_traits_base<S,
typename enable_if<
has_sender_types<S>::value
>::type>
{
template <
template <typename...> class Tuple,
template <typename...> class Variant>
using value_types = typename S::template value_types<Tuple, Variant>;

template <template <typename...> class Variant>
using error_types = typename S::template error_types<Variant>;

ASIO_STATIC_CONSTEXPR(bool, sends_done = S::sends_done);
};

#endif 

template <typename S>
struct sender_traits_base<S,
typename enable_if<
!has_sender_types<S>::value
&& detail::is_executor_of_impl<S,
as_invocable<void_receiver, S> >::value
>::type>
{
#if defined(ASIO_HAS_DEDUCED_EXECUTION_IS_TYPED_SENDER_TRAIT) \
&& defined(ASIO_HAS_STD_EXCEPTION_PTR)

template <
template <typename...> class Tuple,
template <typename...> class Variant>
using value_types = Variant<Tuple<>>;

template <template <typename...> class Variant>
using error_types = Variant<std::exception_ptr>;

ASIO_STATIC_CONSTEXPR(bool, sends_done = true);

#endif 
};

} 

#if defined(GENERATING_DOCUMENTATION)
typedef unspecified sender_base;
#else 
typedef detail::sender_base_ns::sender_base sender_base;
#endif 

template <typename S>
struct sender_traits
#if !defined(GENERATING_DOCUMENTATION)
: detail::sender_traits_base<S>
#endif 
{
};

namespace detail {

template <typename S, typename = void>
struct has_sender_traits : true_type
{
};

template <typename S>
struct has_sender_traits<S,
typename enable_if<
is_same<
typename asio::execution::sender_traits<
S>::asio_execution_sender_traits_base_is_unspecialised,
void
>::value
>::type> : false_type
{
};

} 



template <typename T>
struct is_sender :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
conditional<
detail::has_sender_traits<typename remove_cvref<T>::type>::value,
is_move_constructible<typename remove_cvref<T>::type>,
false_type
>::type
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
ASIO_CONSTEXPR const bool is_sender_v = is_sender<T>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T>
ASIO_CONCEPT sender = is_sender<T>::value;

#define ASIO_EXECUTION_SENDER ::asio::execution::sender

#else 

#define ASIO_EXECUTION_SENDER typename

#endif 

template <typename S, typename R>
struct can_connect;


template <typename T, typename R>
struct is_sender_to :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
integral_constant<bool,
is_sender<T>::value
&& is_receiver<R>::value
&& can_connect<T, R>::value
>
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename R>
ASIO_CONSTEXPR const bool is_sender_to_v =
is_sender_to<T, R>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename R>
ASIO_CONCEPT sender_to = is_sender_to<T, R>::value;

#define ASIO_EXECUTION_SENDER_TO(r) \
::asio::execution::sender_to<r>

#else 

#define ASIO_EXECUTION_SENDER_TO(r) typename

#endif 


template <typename T>
struct is_typed_sender :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
integral_constant<bool,
is_sender<T>::value
&& detail::has_sender_types<
sender_traits<typename remove_cvref<T>::type> >::value
>
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
ASIO_CONSTEXPR const bool is_typed_sender_v = is_typed_sender<T>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T>
ASIO_CONCEPT typed_sender = is_typed_sender<T>::value;

#define ASIO_EXECUTION_TYPED_SENDER \
::asio::execution::typed_sender

#else 

#define ASIO_EXECUTION_TYPED_SENDER typename

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#include "asio/execution/connect.hpp"

#endif 