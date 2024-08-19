

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/EventCpu.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/kernel/TaskKernelCpuOmp2Blocks.hpp>
#    include <alpaka/queue/QueueCpuBlocking.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/queue/cpu/ICpuQueue.hpp>
#    include <alpaka/test/event/EventHostManualTrigger.hpp>
#    include <alpaka/test/queue/Queue.hpp>
#    include <alpaka/wait/Traits.hpp>

#    include <omp.h>

#    include <atomic>
#    include <mutex>

namespace alpaka
{
namespace cpu::detail
{
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
#    endif
class QueueCpuOmp2CollectiveImpl final : public cpu::ICpuQueue
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
{
public:
QueueCpuOmp2CollectiveImpl(DevCpu const& dev) noexcept : m_dev(dev), m_uCurrentlyExecutingTask(0u)
{
}
QueueCpuOmp2CollectiveImpl(QueueCpuOmp2CollectiveImpl const&) = delete;
auto operator=(QueueCpuOmp2CollectiveImpl const&) -> QueueCpuOmp2CollectiveImpl& = delete;
void enqueue(EventCpu& ev) final
{
alpaka::enqueue(*this, ev);
}
void wait(EventCpu const& ev) final
{
alpaka::wait(*this, ev);
}

public:
DevCpu const m_dev; 
std::mutex mutable m_mutex;
std::atomic<uint32_t> m_uCurrentlyExecutingTask;
};
} 

class QueueCpuOmp2Collective final
: public concepts::Implements<ConceptCurrentThreadWaitFor, QueueCpuOmp2Collective>
{
public:
QueueCpuOmp2Collective(DevCpu const& dev)
: m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuOmp2CollectiveImpl>(dev))
, m_spBlockingQueue(std::make_shared<QueueCpuBlocking>(dev))
{
dev.registerQueue(m_spQueueImpl);
}
auto operator==(QueueCpuOmp2Collective const& rhs) const -> bool
{
return m_spQueueImpl == rhs.m_spQueueImpl && m_spBlockingQueue == rhs.m_spBlockingQueue;
}
auto operator!=(QueueCpuOmp2Collective const& rhs) const -> bool
{
return !((*this) == rhs);
}

public:
std::shared_ptr<cpu::detail::QueueCpuOmp2CollectiveImpl> m_spQueueImpl;
std::shared_ptr<QueueCpuBlocking> m_spBlockingQueue;
};

namespace trait
{
template<>
struct DevType<QueueCpuOmp2Collective>
{
using type = DevCpu;
};
template<>
struct GetDev<QueueCpuOmp2Collective>
{
ALPAKA_FN_HOST static auto getDev(QueueCpuOmp2Collective const& queue) -> DevCpu
{
return queue.m_spQueueImpl->m_dev;
}
};

template<>
struct EventType<QueueCpuOmp2Collective>
{
using type = EventCpu;
};

template<typename TTask>
struct Enqueue<QueueCpuOmp2Collective, TTask>
{
ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, TTask const& task) -> void
{
if(::omp_in_parallel() != 0)
{
while(!empty(*queue.m_spBlockingQueue))
{
}
queue.m_spQueueImpl->m_uCurrentlyExecutingTask += 1u;

#    pragma omp single nowait
task();

queue.m_spQueueImpl->m_uCurrentlyExecutingTask -= 1u;
}
else
{
std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
alpaka::enqueue(*queue.m_spBlockingQueue, task);
}
}
};

template<>
struct Empty<QueueCpuOmp2Collective>
{
ALPAKA_FN_HOST static auto empty(QueueCpuOmp2Collective const& queue) -> bool
{
return queue.m_spQueueImpl->m_uCurrentlyExecutingTask == 0u && alpaka::empty(*queue.m_spBlockingQueue);
}
};

template<>
struct Enqueue<cpu::detail::QueueCpuOmp2CollectiveImpl, EventCpu>
{
ALPAKA_FN_HOST static auto enqueue(cpu::detail::QueueCpuOmp2CollectiveImpl&, EventCpu&) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    pragma omp barrier
}
};
template<>
struct Enqueue<QueueCpuOmp2Collective, EventCpu>
{
ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, EventCpu& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

if(::omp_in_parallel() != 0)
{
while(!empty(*queue.m_spBlockingQueue))
{
}
#    pragma omp barrier
}
else
{
alpaka::enqueue(*queue.m_spBlockingQueue, event);
}
}
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct Enqueue<QueueCpuOmp2Collective, TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
private:
using Task = TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>;

public:
ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, Task const& task) -> void
{
if(::omp_in_parallel() != 0)
{
while(!empty(*queue.m_spBlockingQueue))
{
}
queue.m_spQueueImpl->m_uCurrentlyExecutingTask += 1u;
task();
queue.m_spQueueImpl->m_uCurrentlyExecutingTask -= 1u;
}
else
{
std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
alpaka::enqueue(*queue.m_spBlockingQueue, task);
}
}
};

template<>
struct Enqueue<QueueCpuOmp2Collective, test::EventHostManualTriggerCpu<>>
{
ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective&, test::EventHostManualTriggerCpu<>&) -> void
{
}
};

template<>
struct CurrentThreadWaitFor<QueueCpuOmp2Collective>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueCpuOmp2Collective const& queue) -> void
{
if(::omp_in_parallel() != 0)
{
while(!empty(*queue.m_spBlockingQueue))
{
}
#    pragma omp barrier
}
else
{
std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
wait(*queue.m_spBlockingQueue);
}
}
};


template<>
struct WaiterWaitFor<cpu::detail::QueueCpuOmp2CollectiveImpl, EventCpu>
{
ALPAKA_FN_HOST static auto waiterWaitFor(cpu::detail::QueueCpuOmp2CollectiveImpl&, EventCpu const&) -> void
{
#    pragma omp barrier
}
};
template<>
struct WaiterWaitFor<QueueCpuOmp2Collective, EventCpu>
{
ALPAKA_FN_HOST static auto waiterWaitFor(QueueCpuOmp2Collective& queue, EventCpu const& event) -> void
{
if(::omp_in_parallel() != 0)
{
while(!empty(*queue.m_spBlockingQueue))
{
}
wait(queue);
}
else
wait(*queue.m_spBlockingQueue, event);
}
};
} 
namespace test::trait
{
template<>
struct IsBlockingQueue<QueueCpuOmp2Collective>
{
static constexpr bool value = true;
};
} 
} 

#    include <alpaka/event/EventCpu.hpp>

#endif