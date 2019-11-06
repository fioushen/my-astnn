def flatten_tree(tree):
    res = []
    flatten_tree_helper(tree, res)
    return res


def flatten_tree_helper(tree, res):
    for node in tree:
        if not isinstance(node, list):
            res.append(node)
        else:
            flatten_tree_helper(node, res)
