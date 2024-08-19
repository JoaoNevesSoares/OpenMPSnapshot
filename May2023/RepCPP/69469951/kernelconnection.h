


#pragma once


#include <string>
#include <vector>
#include <map>

#include "paraverkerneltypes.h"

class Timeline;
class Histogram;
class Trace;
class RecordList;
class ProgressController;
class Filter;

class TraceOptions;
class TraceCutter;
class TraceFilter;
class TraceSoftwareCounters;
class TraceShifter;
class TraceEditSequence;
class EventDrivenCutter;
class EventTranslator;

typedef std::pair< TEventType, TEventValue > TTypeValuePair;

enum class UserMessageID
{
MessageCFGNoneEvents = 0,
MessageCFGZeroObjects,
MessageCFGMultipleValues,
MessageCFGSomeEvents,
UserMessageSize
};

static const std::string userMessages[ static_cast<size_t>( UserMessageID::UserMessageSize ) ] =
{
"None of the events specified in the filter appear in the trace.",
"Some timeline has 0 objects selected at some level.",
"Some of the events specified in the filter have multiple instances. All of them will be included.",
"Some of the events specified in the filter doesn't appear in the trace."
};

class KernelConnection
{
public:
virtual ~KernelConnection() {}

virtual bool checkTraceSize( const std::string& filename, TTraceSize maxSize ) const = 0;
virtual TTraceSize getTraceSize( const std::string& filename ) const = 0;
virtual Trace *newTrace( const std::string& whichFile, bool noLoad, ProgressController *progress, TTraceSize traceSize = 0 ) const = 0;
virtual std::string getPCFFileLocation( const std::string& traceFile ) const = 0;
virtual std::string getROWFileLocation( const std::string& traceFile ) const = 0;
virtual Timeline *newSingleWindow() const = 0;
virtual Timeline *newSingleWindow( Trace *whichTrace ) const = 0;
virtual Timeline *newDerivedWindow() const = 0;
virtual Timeline *newDerivedWindow( Timeline *window1, Timeline * window2 ) const = 0;
virtual Histogram *newHistogram() const = 0;
virtual ProgressController *newProgressController() const = 0;
virtual Filter *newFilter( Filter *concreteFilter ) const = 0;
virtual TraceEditSequence *newTraceEditSequence() const = 0;

virtual std::string getToolID( const std::string &toolName ) const = 0;
virtual std::string getToolName( const std::string &toolID ) const = 0;
virtual TraceOptions *newTraceOptions() const = 0;
virtual TraceCutter *newTraceCutter( TraceOptions *options,
const std::vector< TEventType > &whichTypesWithValuesZero ) const = 0;
virtual TraceFilter *newTraceFilter( char *trace_in,
char *trace_out,
TraceOptions *options,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable,
ProgressController *progress = nullptr ) const = 0;
virtual TraceSoftwareCounters *newTraceSoftwareCounters( char *trace_in,
char *trace_out,
TraceOptions *options,
ProgressController *progress = nullptr ) const = 0;
virtual TraceShifter *newTraceShifter( std::string traceIn,
std::string traceOut,
std::string shiftTimesFile,
TWindowLevel shiftLevel,
ProgressController *progress = nullptr ) const = 0;
virtual EventDrivenCutter *newEventDrivenCutter( std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress = nullptr ) const = 0;
virtual EventTranslator *newEventTranslator( std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress = nullptr ) const = 0;

virtual void getAllStatistics( std::vector<std::string>& onVector ) const = 0;
virtual void getAllFilterFunctions( std::vector<std::string>& onVector ) const = 0;
virtual void getAllSemanticFunctions( TSemanticGroup whichGroup,
std::vector<std::string>& onVector ) const = 0;

virtual bool userMessage( UserMessageID messageID ) const = 0;

virtual bool isTraceFile( const std::string &filename ) const = 0;
virtual void copyPCF( const std::string& name, const std::string& traceToLoad ) const = 0;
virtual void copyROW( const std::string& name, const std::string& traceToLoad ) const = 0;
virtual void getNewTraceName( char *name,
char *new_trace_name,
std::string action,
bool saveNewNameInfo = true ) = 0;
virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::string& traceFilterID,
const bool commitName = false ) const = 0;

virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::vector< std::string >& traceFilterID,
const bool commitName = false ) const = 0;

virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::string& outputPath,
const std::vector< std::string >& traceFilterID,
const bool commitName = false ) const = 0;

virtual bool isFileReadable( const std::string& filename,
const std::string& message,
const bool verbose = true,
const bool keepOpen = true,
const bool exitProgram = true ) const = 0;

virtual void commitNewTraceName( const std::string& newTraceName ) const = 0;


virtual std::string getPathSeparator() const = 0;
virtual void setPathSeparator( const std::string& whichPath ) = 0;
virtual std::string getDistributedCFGsPath() const = 0;
virtual std::string getParaverUserDir() const = 0;

protected:

private:

};


