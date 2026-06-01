from .registration import register_converter as register_converter

def get_default_tree_classifier_attribute_pairs(): ...
def get_default_tree_regressor_attribute_pairs(): ...
def add_node(
    attr_pairs,
    is_classifier,
    tree_id,
    tree_weight,
    node_id,
    feature_id,
    mode,
    value,
    true_child_id,
    false_child_id,
    weights,
    weight_id_bias,
    leaf_weights_are_counts,
) -> None: ...
def add_tree_to_attribute_pairs(
    attr_pairs, is_classifier, tree, tree_id, tree_weight, weight_id_bias, leaf_weights_are_counts
) -> None: ...
