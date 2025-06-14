from unittest import TestCase
from src.models.bb_node import BranchBoundNode, BranchBoundNodeList


class TestBranchBoundNode(TestCase):

    def test_init(self):
        y_bound = {
            0: [1, 2],
            1: [10, 20]
        }

        node = BranchBoundNode(y_bound)

        self.assertEqual(y_bound, node.bound)

        y_bound[2] = [100, 200]

        self.assertNotEqual(y_bound, node.bound)
    
    def test_partition(self):

        y_bound = {
            0: [1, 2],
            1: [10, 20]
        }

        node = BranchBoundNode(y_bound)

        node.partition(0)

        bound_1 = {
            0: [1, 1.5],
            1: [10, 20]
        }

        bound_2 = {
            0: [1.5, 2],
            1: [10, 20]
        }

        self.assertEqual(bound_1, node.left.bound)
        self.assertEqual(bound_2, node.right.bound)

        node.partition(1)

        bound_1 = {
            0: [1, 2],
            1: [10, 15]
        }

        bound_2 = {
            0: [1, 2],
            1: [15, 20]
        }

        self.assertEqual(bound_1, node.left.bound)
        self.assertEqual(bound_2, node.right.bound)
    
class TestBranchBoundNodeList(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bound = {
            0: [1, 2],
            1: [10, 20]
        } 
        cls.root = BranchBoundNode(cls.bound)

    def test_init(self):

        node_list = BranchBoundNodeList(self.root)

        node_list.root = self.root

    def test_get_node(self):

        node = self.root

        node_list = BranchBoundNodeList(node)

        self.assertEqual(node_list.get_node(0), node)

        node.partition(0)

        node_list.add_node(node.left)
        node_list.add_node(node.right)
        left_idx = node_list.add_node(node.left)
        right_idx = node_list.add_node(node.right)

        self.assertEqual(node_list.get_node(left_idx), node.left)
        self.assertEqual(node_list.get_node(right_idx), node.right)
    
    def test_add_node(self):

        node = self.root

        node_list = BranchBoundNodeList(node)

        node.partition(0)

        new_idx = node_list.add_node(node.left)
        self.assertEqual(new_idx, 1)
        self.assertEqual(node_list.get_node(1), node.left)

        new_idx = node_list.add_node(node.right)
        self.assertEqual(new_idx, 2)
        self.assertEqual(node_list.get_node(2), node.right)
    
    def test_delete_node(self):

        node = self.root

        node_list = BranchBoundNodeList(node)

        node.partition(0)

        _ = node_list.add_node(node.left)
        _ = node_list.add_node(node.right)

        node_list.delete_node(2)

        with self.assertRaises(KeyError):
            node_list.get_node(2)
        

    def test_fathom_nodes(self):

        node = self.root

        node_list = BranchBoundNodeList(node)

        node.partition(0)

        _ = node_list.add_node(node.left)
        _ = node_list.add_node(node.right)

        node.lbd = 2
        node.left.lbd = 3
        node.right.lbd = 5

        ubd = 4

        node_list.fathom_nodes(ubd)

        self.assertEqual(len(node_list.active_nodes), 2)

        node_list.get_node(0)
        node_list.get_node(1)
        with self.assertRaises(KeyError):
            node_list.get_node(2)

    def test_is_empty(self):
        
        node = self.root

        node_list = BranchBoundNodeList(node)

        self.assertFalse(node_list.is_empty())

        node_list.delete_node(0)
        self.assertTrue(node_list.is_empty())
    
    def test_find_min_lbd(self):
        node = self.root

        node_list = BranchBoundNodeList(node)

        node.partition(0)

        _ = node_list.add_node(node.left)
        _ = node_list.add_node(node.right)

        node.lbd = 2
        node.left.lbd = 3
        node.right.lbd = 5

        min_idx, min_lbd = node_list.find_min_lbd()

        self.assertEqual(min_idx, 0)
        self.assertEqual(min_lbd, 2)
    
    def test_get_all_bounds(self):
        node = self.root

        node_list = BranchBoundNodeList(node)

        node.partition(0)

        _ = node_list.add_node(node.left)
        _ = node_list.add_node(node.right)

        bound = self.bound
        bound_1 = {
            0: [1, 1.5],
            1: [10, 20]
        }
        bound_2 = {
            0: [1.5, 2],
            1: [10, 20]
        }

        self.assertEqual(node_list.get_all_bounds(), [bound, bound_1, bound_2])
        

