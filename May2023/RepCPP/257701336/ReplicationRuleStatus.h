

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ReplicationRuleStatus
{
NOT_SET,
Enabled,
Disabled
};

namespace ReplicationRuleStatusMapper
{
AWS_S3_API ReplicationRuleStatus GetReplicationRuleStatusForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForReplicationRuleStatus(ReplicationRuleStatus value);
} 
} 
} 
} 