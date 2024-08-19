
#ifndef BOOST_ASIO_IP_ADDRESS_V6_RANGE_HPP
#define BOOST_ASIO_IP_ADDRESS_V6_RANGE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/ip/address_v6_iterator.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

template <typename> class basic_address_range;


template <> class basic_address_range<address_v6>
{
public:
typedef basic_address_iterator<address_v6> iterator;

basic_address_range() BOOST_ASIO_NOEXCEPT
: begin_(address_v6()),
end_(address_v6())
{
}

explicit basic_address_range(const iterator& first,
const iterator& last) BOOST_ASIO_NOEXCEPT
: begin_(first),
end_(last)
{
}

basic_address_range(const basic_address_range& other) BOOST_ASIO_NOEXCEPT
: begin_(other.begin_),
end_(other.end_)
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
basic_address_range(basic_address_range&& other) BOOST_ASIO_NOEXCEPT
: begin_(BOOST_ASIO_MOVE_CAST(iterator)(other.begin_)),
end_(BOOST_ASIO_MOVE_CAST(iterator)(other.end_))
{
}
#endif 

basic_address_range& operator=(
const basic_address_range& other) BOOST_ASIO_NOEXCEPT
{
begin_ = other.begin_;
end_ = other.end_;
return *this;
}

#if defined(BOOST_ASIO_HAS_MOVE)
basic_address_range& operator=(
basic_address_range&& other) BOOST_ASIO_NOEXCEPT
{
begin_ = BOOST_ASIO_MOVE_CAST(iterator)(other.begin_);
end_ = BOOST_ASIO_MOVE_CAST(iterator)(other.end_);
return *this;
}
#endif 

iterator begin() const BOOST_ASIO_NOEXCEPT
{
return begin_;
}

iterator end() const BOOST_ASIO_NOEXCEPT
{
return end_;
}

bool empty() const BOOST_ASIO_NOEXCEPT
{
return begin_ == end_;
}

iterator find(const address_v6& addr) const BOOST_ASIO_NOEXCEPT
{
return addr >= *begin_ && addr < *end_ ? iterator(addr) : end_;
}

private:
iterator begin_;
iterator end_;
};

typedef basic_address_range<address_v6> address_v6_range;

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 