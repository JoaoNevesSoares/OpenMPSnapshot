#ifndef BOOST_SMART_PTR_DETAIL_SPINLOCK_STD_ATOMIC_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SPINLOCK_STD_ATOMIC_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/yield_k.hpp>
#include <boost/config.hpp>
#include <atomic>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using std::atomic spinlock")

#endif

namespace boost
{

namespace detail
{

class spinlock
{
public:

std::atomic_flag v_;

public:

bool try_lock() BOOST_NOEXCEPT
{
return !v_.test_and_set( std::memory_order_acquire );
}

void lock() BOOST_NOEXCEPT
{
for( unsigned k = 0; !try_lock(); ++k )
{
boost::detail::yield( k );
}
}

void unlock() BOOST_NOEXCEPT
{
v_ .clear( std::memory_order_release );
}

public:

class scoped_lock
{
private:

spinlock & sp_;

scoped_lock( scoped_lock const & );
scoped_lock & operator=( scoped_lock const & );

public:

explicit scoped_lock( spinlock & sp ) BOOST_NOEXCEPT: sp_( sp )
{
sp.lock();
}

~scoped_lock() 
{
sp_.unlock();
}
};
};

} 
} 

#define BOOST_DETAIL_SPINLOCK_INIT { ATOMIC_FLAG_INIT }

#endif 