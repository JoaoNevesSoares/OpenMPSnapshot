
#ifndef BOOST_ASIO_DETAIL_TIMER_QUEUE_PTIME_HPP
#define BOOST_ASIO_DETAIL_TIMER_QUEUE_PTIME_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_BOOST_DATE_TIME)

#include <boost/asio/time_traits.hpp>
#include <boost/asio/detail/timer_queue.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

struct forwarding_posix_time_traits : time_traits<boost::posix_time::ptime> {};

template <>
class timer_queue<time_traits<boost::posix_time::ptime> >
: public timer_queue_base
{
public:
typedef boost::posix_time::ptime time_type;

typedef boost::posix_time::time_duration duration_type;

typedef timer_queue<forwarding_posix_time_traits>::per_timer_data
per_timer_data;

BOOST_ASIO_DECL timer_queue();

BOOST_ASIO_DECL virtual ~timer_queue();

BOOST_ASIO_DECL bool enqueue_timer(const time_type& time,
per_timer_data& timer, wait_op* op);

BOOST_ASIO_DECL virtual bool empty() const;

BOOST_ASIO_DECL virtual long wait_duration_msec(long max_duration) const;

BOOST_ASIO_DECL virtual long wait_duration_usec(long max_duration) const;

BOOST_ASIO_DECL virtual void get_ready_timers(op_queue<operation>& ops);

BOOST_ASIO_DECL virtual void get_all_timers(op_queue<operation>& ops);

BOOST_ASIO_DECL std::size_t cancel_timer(
per_timer_data& timer, op_queue<operation>& ops,
std::size_t max_cancelled = (std::numeric_limits<std::size_t>::max)());

BOOST_ASIO_DECL void move_timer(per_timer_data& target,
per_timer_data& source);

private:
timer_queue<forwarding_posix_time_traits> impl_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/timer_queue_ptime.ipp>
#endif 

#endif 

#endif 