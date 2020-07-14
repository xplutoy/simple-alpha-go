from graphviz import Digraph


# from  mcts import *

class TreeToImage:
    def __init__(self, root):
        self.node_list = []
        self.dot = Digraph(comment='MCTS search')
        self.current_node = root
        self.current_tag = 'root'
        self.expand_image(root, self.current_tag)
        self.render()

    def add_node(self, node_url, node_tag):
        self.dot.node(node_url, node_tag)

    def add_edge(self, start_node_url, end_node_url):
        self.dot.edge(start_node_url, end_node_url, constraint='true')

    def expand_image(self, node, tag):
        current_node = node
        current_tag = tag
        if not current_node.is_leaf():
            for i in range(len(current_node.edges)):
                tag = str(current_node.child[i].index) + '_'+str(current_node.level)  + '_' + \
                      str(
                    current_node.edges[i])
                self.add_node(tag, tag)
                self.add_edge(current_tag, tag)
                self.expand_image(current_node.child[i], tag)

    def render(self):
        self.dot.view()
        # print(self.dot.source)
        # 保存source到文件，并提供Graphviz引擎
        # self.dot.render('test-output/gobang.gv', view=True)

# if __name__ == '__main__':
#     root = mcts.Node()
#     edges = []
#     for i in range(3):
#         edge = [[1, i], 3, 3]
#         edges.append(edge)
#     root.edges = edges
#     node1 = mcts.Node()
#     node2 = mcts.Node()
#     node3 = mcts.Node()
#     root.add_child(node1)
#     root.add_child(node2)
#     root.add_child(node3)
#     node4 = mcts.Node()
#     node5 = mcts.Node()
#     node6 = mcts.Node()
#     node7 = mcts.Node()
#     node8 = mcts.Node()
#     edges2 = []
#     for i in range(3):
#         edge = [[2, i], 3, 3]
#         edges2.append(edge)
#     node1.edges = edges2
#     node1.add_child(node4)
#     node1.add_child(node5)
#     node1.add_child(node6)
#     edges3 = []
#     for i in range(2):
#         edge = [[3, i], 3, 3]
#         edges3.append(edge)
#     node3.edges = edges3
#     node3.add_child(node7)
#     node3.add_child(node8)
#
#     tree = TreeToImage(root)
