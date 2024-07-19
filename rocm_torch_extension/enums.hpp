#pragma once
enum class EGrid2dMode
{
    BATCH_UNIT,
    BATCH_UNIT_TG, //!< Scalar weight access and coalesced access over batches, transposed (unit, batch) block order
};