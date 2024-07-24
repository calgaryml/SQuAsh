#pragma once

enum class EGrid2dMode
{
    BATCH_UNIT,
    BATCH_UNIT_TG
};

enum class ForwardImplementations
{
    GPU_BatchUnit_Fan,
    GPU_BatchUnit_TG_Fan,
};

enum class BackwardFtrImplementations
{
    GPU_BatchUnit_Fan
};

enum class BackwardWgtImplementations
{
    GPU_FanUnit_Batch
};