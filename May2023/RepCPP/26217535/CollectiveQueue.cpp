

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/alpaka.hpp>
#    include <alpaka/test/queue/Queue.hpp>
#    include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>
#    include <alpaka/test/queue/QueueTestFixture.hpp>

#    include <catch2/catch_test_macros.hpp>

#    include <vector>

struct QueueCollectiveTestKernel
{
template<typename TAcc>
auto operator()(TAcc const& acc, int* resultsPtr) const -> void
{
size_t threadId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
std::this_thread::sleep_for(std::chrono::milliseconds(200u * threadId));
resultsPtr[threadId] = static_cast<int>(threadId);
}
};

TEST_CASE("queueCollective", "[queue]")
{
using Dim = alpaka::DimInt<1>;
using Idx = size_t;

using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
using Dev = alpaka::Dev<Acc>;

using Queue = alpaka::QueueCpuOmp2Collective;
using Pltf = alpaka::Pltf<Dev>;

auto dev = alpaka::getDevByIdx<Pltf>(0u);
Queue queue(dev);

std::vector<int> results(4, -1);

using Vec = alpaka::Vec<Dim, Idx>;
Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
Vec const blocksPerGrid(results.size());

using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
WorkDiv const workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

#    pragma omp parallel num_threads(static_cast <int>(results.size()))
{
alpaka::exec<Acc>(queue, workDiv, QueueCollectiveTestKernel{}, results.data());

alpaka::wait(queue);
}

for(size_t i = 0; i < results.size(); ++i)
{
REQUIRE(static_cast<int>(i) == results.at(i));
}
}

TEST_CASE("TestCollectiveMemcpy", "[queue]")
{
using Dim = alpaka::DimInt<1>;
using Idx = size_t;

using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
using Dev = alpaka::Dev<Acc>;

using Queue = alpaka::QueueCpuOmp2Collective;
using Pltf = alpaka::Pltf<Dev>;

auto dev = alpaka::getDevByIdx<Pltf>(0u);
Queue queue(dev);

std::vector<int> results(4, -1);

using Vec = alpaka::Vec<Dim, Idx>;
Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
Vec const blocksPerGrid(results.size());

using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
WorkDiv const workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

#    pragma omp parallel num_threads(static_cast <int>(results.size()))
{
int threadId = omp_get_thread_num();

auto dst = alpaka::createView(dev, results.data() + threadId, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));
auto src = alpaka::createView(dev, &threadId, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));

size_t sleep_ms = (results.size() - static_cast<uint32_t>(threadId)) * 100u;
std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

alpaka::memcpy(queue, dst, src, Vec(static_cast<Idx>(1u)));

alpaka::wait(queue);
}

uint32_t numFlippedValues = 0u;
uint32_t numNonIntitialValues = 0u;
for(size_t i = 0; i < results.size(); ++i)
{
if(static_cast<int>(i) == results.at(i))
numFlippedValues++;
if(results.at(i) != -1)
numNonIntitialValues++;
}
REQUIRE(numFlippedValues == 1u);
REQUIRE(numNonIntitialValues == 1u);
}

#endif