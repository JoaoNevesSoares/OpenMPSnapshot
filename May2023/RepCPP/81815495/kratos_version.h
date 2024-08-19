
#include <string>

#include "includes/kratos_export_api.h"


#pragma once

namespace Kratos {

#ifndef KRATOS_MAJOR_VERSION
#define KRATOS_MAJOR_VERSION 9
#endif

#ifndef KRATOS_MINOR_VERSION
#define KRATOS_MINOR_VERSION 0
#endif

#define KRATOS_VERSION_EQ(MAJOR,MINOR) \
((KRATOS_MAJOR_VERSION == (MAJOR)) && (KRATOS_MINOR_VERSION == (MINOR)))

#define KRATOS_VERSION_ KRATOS_VERSION_EQ

#define KRATOS_VERSION_LT(MAJOR,MINOR)                                  \
(KRATOS_MAJOR_VERSION < (MAJOR) || (KRATOS_MAJOR_VERSION == (MAJOR) &&   \
(KRATOS_MINOR_VERSION < (MINOR) )))

#define KRATOS_VERSION_LE(MAJOR,MINOR) \
(KRATOS_VERSION_LT(MAJOR,MINOR) || KRATOS_VERSION_EQ(MAJOR,MINOR))

#define KRATOS_VERSION_GT(MAJOR,MINOR) (0 == KRATOS_VERSION_LE(MAJOR,MINOR))

#define KRATOS_VERSION_GE(MAJOR,MINOR) (0 == KRATOS_VERSION_LT(MAJOR,MINOR))

constexpr int GetMajorVersion() {
return KRATOS_MAJOR_VERSION;
}

constexpr int GetMinorVersion() {
return KRATOS_MINOR_VERSION;
}

KRATOS_API_EXPORT std::string GetPatchVersion();
KRATOS_API_EXPORT std::string GetCommit();
KRATOS_API_EXPORT std::string GetBranchName();
KRATOS_API_EXPORT std::string GetBuildType();
KRATOS_API_EXPORT std::string GetVersionString();
KRATOS_API_EXPORT std::string GetOSName();
KRATOS_API_EXPORT std::string GetCompiler();

} 