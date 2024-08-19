#pragma once

#include <faabric/proto/faabric.pb.h>
#include <faabric/snapshot/SnapshotRegistry.h>
#include <faabric/state/State.h>
#include <faabric/util/memory.h>
#include <faabric/util/queue.h>
#include <faabric/util/snapshot.h>
#include <storage/FileSystem.h>
#include <threads/ThreadState.h>
#include <wasm/WasmCommon.h>
#include <wasm/WasmEnvironment.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <sys/uio.h>
#include <thread>
#include <tuple>

namespace wasm {

enum ThreadRequestType
{
UNSET = 0,
PTHREAD = 1,
OPENMP = 2,
};

bool isWasmPageAligned(int32_t offset);

class WasmModule
{
public:
WasmModule();

explicit WasmModule(int threadPoolSizeIn);

virtual ~WasmModule();

virtual void reset(faabric::Message& msg, const std::string& snapshotKey);

void bindToFunction(faabric::Message& msg, bool cache = true);

int32_t executeTask(int threadPoolIdx,
int msgIdx,
std::shared_ptr<faabric::BatchExecuteRequest> req);

virtual int32_t executeFunction(faabric::Message& msg);

bool isBound();

std::string getBoundUser();

std::string getBoundFunction();

virtual void flush();

uint32_t getArgc();

uint32_t getArgvBufferSize();

virtual void writeArgvToMemory(uint32_t wasmArgvPointers,
uint32_t wasmArgvBuffer);

virtual void writeWasmEnvToMemory(uint32_t envPointers, uint32_t envBuffer);

WasmEnvironment& getWasmEnvironment();

storage::FileSystem& getFileSystem();

ssize_t captureStdout(const struct ::iovec* iovecs, int iovecCount);

ssize_t captureStdout(const void* buffer);

std::string getCapturedStdout();

void clearCapturedStdout();

uint32_t getCurrentBrk();

virtual void setMemorySize(size_t nBytes);

uint32_t growMemory(size_t nBytes);

uint32_t shrinkMemory(size_t nBytes);

uint32_t mmapMemory(size_t nBytes);

virtual uint32_t mmapFile(uint32_t fp, size_t length);

void unmapMemory(uint32_t offset, size_t nBytes);

uint32_t createMemoryGuardRegion(uint32_t wasmOffset);

virtual uint32_t mapSharedStateMemory(
const std::shared_ptr<faabric::state::StateKeyValue>& kv,
long offset,
uint32_t length);

virtual uint8_t* wasmPointerToNative(uint32_t wasmPtr);

virtual size_t getMemorySizeBytes();

virtual size_t getMaxMemoryPages();

virtual uint8_t* getMemoryBase();

std::shared_ptr<faabric::util::SnapshotData> getSnapshotData();

std::span<uint8_t> getMemoryView();

std::string snapshot(bool locallyRestorable = true);

void restore(const std::string& snapshotKey);

void queuePthreadCall(threads::PthreadCall call);

int awaitPthreadCall(faabric::Message* msg, int pthreadPtr);

std::vector<uint32_t> getThreadStacks();

std::shared_ptr<std::mutex> getPthreadMutex(uint32_t id);

std::shared_ptr<std::mutex> getOrCreatePthreadMutex(uint32_t id);

void addMergeRegionForNextThreads(
uint32_t wasmPtr,
size_t regionSize,
faabric::util::SnapshotDataType dataType,
faabric::util::SnapshotMergeOperation mergeOp);

std::vector<faabric::util::SnapshotMergeRegion> getMergeRegions();

void clearMergeRegions();

virtual int32_t executeOMPThread(int threadPoolIdx,
uint32_t stackTop,
faabric::Message& msg);

virtual int32_t executePthread(int threadPoolIdx,
uint32_t stackTop,
faabric::Message& msg);

virtual void printDebugInfo();

protected:
std::shared_mutex moduleMutex;

std::atomic<uint32_t> currentBrk = 0;

std::string boundUser;
std::string boundFunction;
bool _isBound = false;

storage::FileSystem filesystem;

WasmEnvironment wasmEnvironment;

int stdoutMemFd = 0;
ssize_t stdoutSize = 0;

int threadPoolSize = 0;
std::vector<uint32_t> threadStacks;

unsigned int argc;
std::vector<std::string> argv;
size_t argvBufferSize;

std::vector<threads::PthreadCall> queuedPthreadCalls;
std::unordered_map<int32_t, uint32_t> pthreadPtrsToChainedCalls;
std::vector<std::pair<uint32_t, int32_t>> lastPthreadResults;
std::vector<faabric::util::SnapshotMergeRegion> mergeRegions;

std::shared_mutex pthreadLocksMx;
std::unordered_map<uint32_t, std::shared_ptr<std::mutex>> pthreadLocks;

std::shared_mutex sharedMemWasmPtrsMutex;
std::unordered_map<std::string, uint32_t> sharedMemWasmPtrs;

int getStdoutFd();

void prepareArgcArgv(const faabric::Message& msg);

virtual void doBindToFunction(faabric::Message& msg, bool cache);

virtual bool doGrowMemory(uint32_t pageChange);

faabric::snapshot::SnapshotRegistry& reg;

void snapshotWithKey(const std::string& snapKey);

void ignoreThreadStacksInSnapshot(const std::string& snapKey);

void createThreadStacks();
};

size_t getNumberOfWasmPagesForBytes(size_t nBytes);

uint32_t roundUpToWasmPageAligned(uint32_t nBytes);

size_t getPagesForGuardRegion();


class WasmExitException : public std::exception
{
public:
explicit WasmExitException(int exitCode)
: exitCode(exitCode)
{}

int exitCode;
};

}