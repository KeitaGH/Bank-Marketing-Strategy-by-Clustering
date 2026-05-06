from graphviz import Digraph

dot = Digraph(format='png')


dot.attr(rankdir='TB', size='10,12')
dot.attr('node', shape='box', style='rounded,filled',
         color='black', fillcolor='#4F4F4F',
         fontcolor='white', fontsize='16',
         width='2.8', height='1')

dot.attr('edge', color='black', penwidth='1.5')


dot.node('A', 'Dataset')
dot.node('B', 'Data Preprocessing')
dot.node('C', 'Feature Selection')


dot.node('D1', 'K-Means')
dot.node('D2', 'DBSCAN')


dot.node('E', 'Cluster Analysis')
dot.node('F', 'Model Evaluation\n(Silhouette & DBI)')
dot.node('G', 'Marketing Recommendation')
dot.node('H', 'Final Model Selection')


dot.edge('A', 'B')
dot.edge('B', 'C')


dot.edge('C', 'D1')
dot.edge('C', 'D2')


dot.edge('D1', 'E')
dot.edge('D2', 'E')


dot.edge('E', 'F')


dot.edge('F', 'G')


dot.edge('G', 'H')


with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('D1')
    s.node('D2')

# Render
dot.render('block_diagram_final', view=True)