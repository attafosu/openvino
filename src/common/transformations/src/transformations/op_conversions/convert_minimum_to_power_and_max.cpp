// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <vector>

#include "itt.hpp"

ov::pass::ConvertMinimum::ConvertMinimum() {
    MATCHER_SCOPE(ConvertMinimum);
    auto minimum = ngraph::pattern::wrap_type<opset1::Minimum>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto minimum = std::dynamic_pointer_cast<ov::opset1::Minimum>(m.get_match_root());
        if (!minimum || transformation_callback(minimum) || !minimum->get_output_element_type(0).is_signed()) {
            return false;
        }

        /*
         * Decompose Minimum operation to Mul(-1)---->Maximum-->Mul(-1)
         *                                Mul(-1)--'
         */

        auto neg_0 = std::make_shared<ov::opset1::Multiply>(
            minimum->input(0).get_source_output(),
            opset1::Constant::create(minimum->get_input_element_type(0), Shape{}, {-1}));

        auto neg_1 = std::make_shared<ov::opset1::Multiply>(
            minimum->input(1).get_source_output(),
            opset1::Constant::create(minimum->get_input_element_type(1), Shape{}, {-1}));

        auto max = std::make_shared<ov::opset1::Maximum>(neg_0, neg_1);

        auto neg_2 =
            std::make_shared<ov::opset1::Multiply>(max,
                                                   opset1::Constant::create(max->get_element_type(), Shape{}, {-1}));

        neg_2->set_friendly_name(minimum->get_friendly_name());
        ngraph::copy_runtime_info(minimum, {neg_0, neg_1, max, neg_2});
        ngraph::replace_node(minimum, neg_2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(minimum, matcher_name);
    this->register_matcher(m, callback);
}
