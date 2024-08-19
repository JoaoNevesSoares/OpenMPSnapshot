
#ifndef ASIO_TRAITS_PREFER_FREE_HPP
#define ASIO_TRAITS_PREFER_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_DECLTYPE) \
&& defined(ASIO_HAS_NOEXCEPT) \
&& defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_PREFER_FREE_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename Property, typename = void>
struct prefer_free_default;

template <typename T, typename Property, typename = void>
struct prefer_free;

} 
namespace detail {

struct no_prefer_free
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(ASIO_HAS_DEDUCED_PREFER_FREE_TRAIT)

template <typename T, typename Property, typename = void>
struct prefer_free_trait : no_prefer_free
{
};

template <typename T, typename Property>
struct prefer_free_trait<T, Property,
typename void_type<
decltype(prefer(declval<T>(), declval<Property>()))
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
prefer(declval<T>(), declval<Property>()));

ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
prefer(declval<T>(), declval<Property>())));
};

#else 

template <typename T, typename Property, typename = void>
struct prefer_free_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<Property, typename decay<Property>::type>::value,
no_prefer_free,
traits::prefer_free<
typename decay<T>::type,
typename decay<Property>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename Property, typename>
struct prefer_free_default :
detail::prefer_free_trait<T, Property>
{
};

template <typename T, typename Property, typename>
struct prefer_free :
prefer_free_default<T, Property>
{
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 