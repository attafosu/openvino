// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_broadcast_utils.h"

#include "cpu_memcpy.h"
#include "ie_parallel.hpp"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

VectorDims TileBroadcastCommon::calculateDenseStrides(const VectorDims &dims) {
    VectorDims strides(dims.size(), 1);

    for (int i = strides.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    return strides;
}

void TileBroadcastCommon::fillOptimizedDimsAndSrcStrides(const VectorDims& srcBlockedDims, const VectorDims& blockedRepeats,
        VectorDims& optimizedDims, VectorDims& optimizedSrcStrides) {
    optimizedDims.clear();
    optimizedSrcStrides.clear();
    VectorDims srcBlockedStrides = calculateDenseStrides(srcBlockedDims);

    for (int i = 0; i < srcBlockedDims.size(); i++) {
        optimizedDims.push_back(blockedRepeats[i]);
        optimizedDims.push_back(srcBlockedDims[i]);
        optimizedSrcStrides.push_back(0);
        optimizedSrcStrides.push_back(srcBlockedStrides[i]);
    }

    int i = 1;
    while (i < optimizedDims.size() - 1) {
        if (optimizedDims[i] == 1) {
            optimizedDims[i + 1] *= optimizedDims[i - 1];
            optimizedDims.erase(optimizedDims.begin() + i - 1, optimizedDims.begin() + i + 1);
            optimizedSrcStrides.erase(optimizedSrcStrides.begin() + i - 1, optimizedSrcStrides.begin() + i + 1);
        } else {
            i++;
        }
    }

    if (optimizedDims[0] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.begin());
        optimizedSrcStrides.erase(optimizedSrcStrides.begin());
    }

    if (optimizedDims[optimizedDims.size() - 1] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.end() - 1);
        optimizedSrcStrides.erase(optimizedSrcStrides.end() - 1);
    }
}

bool TileBroadcastCommon::canBeExecutedInBlockedLayout(VectorDims srcBlockedDims, VectorDims blockedRepeats,
        const size_t elemsInBlock) {
    if (srcBlockedDims.empty() || blockedRepeats.empty() || elemsInBlock == 0lu || srcBlockedDims[1] == Shape::UNDEFINED_DIM ||
            (blockedRepeats[1] != 1 && srcBlockedDims[1] % elemsInBlock != 0))
        return false;

    srcBlockedDims[1] = div_up(srcBlockedDims[1], elemsInBlock);
    srcBlockedDims.push_back(elemsInBlock);
    blockedRepeats.push_back(1);

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    return optimizedDims.size() <= maxNDims;
}

bool TileBroadcastCommon::canBeExecutedInNSPCLayout(VectorDims srcBlockedDims, VectorDims blockedRepeats) {
    srcBlockedDims.push_back(srcBlockedDims[1]);
    srcBlockedDims.erase(srcBlockedDims.begin() + 1);
    blockedRepeats.push_back(blockedRepeats[1]);
    blockedRepeats.erase(blockedRepeats.begin() + 1);

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    return optimizedDims.size() <= maxNDims;
}

std::vector<NodeDesc> TileBroadcastCommon::getSupportedConfigs(const MKLDNNNode *node) {
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    auto precision = node->getOriginalInputPrecisionAtPort(0);
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    const auto& srcDims = node->getInputShapeAtPort(0).getDims();
    const auto& inDataShape = node->getInputShapeAtPort(0);
    size_t outDataShapeRank = node->getOutputShapeAtPort(0).getRank();

    NodeConfig config;
    if (repeats.size() != outDataShapeRank && !repeats.empty())
        IE_THROW() << node->getTypeStr() << " node with name " << node->getName() << " has incorrect Repeats vector."
                "Repeats rank must be equal to output shape rank. Repeats rank: " << repeats.size() << ", output shape rank: " << outDataShapeRank;

    config.dynBatchSupport = false;
    config.inConfs.resize(node->getParentEdges().size());
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = constMap[0];
    config.inConfs[1].inPlace = -1;
    config.inConfs[1].constant = constMap[1];
    config.inConfs[1].desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::I32, node->getInputShapeAtPort(1));
    if (config.inConfs.size() == 3) {
        config.inConfs[2].inPlace = -1;
        config.inConfs[2].constant = constMap[2];
        config.inConfs[2].desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::I32, node->getInputShapeAtPort(2));
    }

    config.outConfs.resize(node->getChildEdges().size());

    auto pushDesc = [&](mkldnn::memory::format_tag inFormat, mkldnn::memory::format_tag outFormat) {
        config.inConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(node->getInputShapeAtPort(0), dataType, inFormat);
        for (int i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = std::make_shared<DnnlBlockedMemoryDesc>(node->getOutputShapeAtPort(0), dataType, outFormat);
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    };

    if (!repeats.empty() && inDataShape.getRank() == outDataShapeRank && (outDataShapeRank == 4 || outDataShapeRank == 5)) {
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 16)) {
            if (outDataShapeRank == 4) {
                pushDesc(mkldnn::memory::format_tag::nChw16c, mkldnn::memory::format_tag::nChw16c);
            } else {
                pushDesc(mkldnn::memory::format_tag::nCdhw16c, mkldnn::memory::format_tag::nCdhw16c);
            }
        }
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 8)) {
            if (outDataShapeRank == 4) {
                pushDesc(mkldnn::memory::format_tag::nChw8c, mkldnn::memory::format_tag::nChw8c);
            } else {
                pushDesc(mkldnn::memory::format_tag::nCdhw8c, mkldnn::memory::format_tag::nCdhw8c);
            }
        }
        if (canBeExecutedInNSPCLayout(srcDims, repeats)) {
            if (outDataShapeRank == 4) {
                pushDesc(mkldnn::memory::format_tag::nhwc, mkldnn::memory::format_tag::nhwc);
            } else {
                pushDesc(mkldnn::memory::format_tag::ndhwc, mkldnn::memory::format_tag::ndhwc);
            }
        }
    }

    auto inFmt = MKLDNNExtensionUtils::GetPlainFormatByRank(inDataShape.getRank());
    auto outFmt = MKLDNNExtensionUtils::GetPlainFormatByRank(outDataShapeRank);
    if (inFmt == mkldnn::memory::format_tag::undef || outFmt == mkldnn::memory::format_tag::undef) {
        config.inConfs[0].desc = std::make_shared<CpuBlockedMemoryDesc>(precision, node->getInputShapeAtPort(0));
        for (int i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = std::make_shared<CpuBlockedMemoryDesc>(precision, node->getOutputShapeAtPort(i));
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    } else {
        pushDesc(inFmt, outFmt);
    }

    return supportedPrimitiveDescriptors;
}

bool TileBroadcastCommon::prepareOptimizedParams(const MKLDNNNode *node, VectorDims& srcBlockedDims, VectorDims& dstBlockedDims) {
    while (srcBlockedDims.size() < dstBlockedDims.size()) {
        srcBlockedDims.insert(srcBlockedDims.begin(), 1);
    }

    VectorDims blockedRepeats = repeats;
    // for nC(d)hw16c and nC(d)hw8c layouts
    while (blockedRepeats.size() < dstBlockedDims.size()) {
        blockedRepeats.push_back(1);
    }
    // for NSPC layouts
    if (node->getBaseMemDescAtInputPort(0)->hasLayoutType(LayoutType::nspc) && one_of(node->getBaseMemDescAtInputPort(0)->getShape().getRank(), 4, 5)) {
        blockedRepeats.push_back(blockedRepeats[1]);
        blockedRepeats.erase(blockedRepeats.begin() + 1);
    }

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    if (optimizedDims.size() > maxNDims)
        return false;

    while (optimizedDims.size() < maxNDims) {
        optimizedDims.insert(optimizedDims.begin(), 1);
        optimizedSrcStrides.insert(optimizedSrcStrides.begin(), 1);
    }

    VectorDims optimizedDstStrides = calculateDenseStrides(optimizedDims);

    size_t dataSize = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision().size();
    for (int i = 0; i < optimizedDims.size(); i++) {
        optimizedSrcStrides[i] *= dataSize;
        optimizedDstStrides[i] *= dataSize;
    }

    optimizedParams.dims = optimizedDims;
    optimizedParams.srcStrides = optimizedSrcStrides;
    optimizedParams.dstStrides = optimizedDstStrides;
    optimizedParams.copySize = optimizedDims[5] * dataSize;

    return true;
}

void TileBroadcastCommon::optimizedExecute(const MKLDNNMemoryPtr& srcMemory, const MKLDNNMemoryPtr& dstMemory) {
    auto srcData = reinterpret_cast<const char *>(srcMemory->GetPtr());
    auto dstData = reinterpret_cast<char *>(dstMemory->GetPtr());

    if (optimizedParams.srcStrides[5] == 0) {
        parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                [&](int i0, int i1, int i2, int i3, int i4) {
            auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                                                 i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                                                 i4 * optimizedParams.srcStrides[4]);
            auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                                           i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                                           i4 * optimizedParams.dstStrides[4]);
            for (int i = 0; i < optimizedParams.dims[5]; i++) {
                cpu_memcpy(dstData2 + i * optimizedParams.dstStrides[5], srcData2, optimizedParams.dstStrides[5]);
            }
        });
    } else {
        parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                [&](int i0, int i1, int i2, int i3, int i4) {
            auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                                                 i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                                                 i4 * optimizedParams.srcStrides[4]);
            auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                                           i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                                           i4 * optimizedParams.dstStrides[4]);
            cpu_memcpy(dstData2, srcData2, optimizedParams.copySize);
        });
    }
}