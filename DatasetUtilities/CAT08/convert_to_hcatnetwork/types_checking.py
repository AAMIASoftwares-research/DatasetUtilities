import hcatnetwork

graph_save_path = f"C:\\Users\\lecca\\Desktop\\GraphsCAT08\\dataset{0:02d}_0.5mm.GML"


print("Graph types checking: \n\n")
graph_reloaded_05mm = hcatnetwork.io.load_graph(graph_save_path, output_type=hcatnetwork.graph.SimpleCenterlineGraph)
for graph_feature in graph_reloaded_05mm.graph:
    print(f"{graph_feature}: {graph_reloaded_05mm.graph[graph_feature]}, type: {type(graph_reloaded_05mm.graph[graph_feature])}, expected type: {graph_reloaded_05mm._attributes_types_dict[graph_feature]}")
for node in graph_reloaded_05mm.nodes.values():
    for node_feature in node:
        print(f"{node_feature}: {node[node_feature]}, type: {type(node[node_feature])}, expected type: {graph_reloaded_05mm._simple_centerline_node_attributes_types_dict[node_feature]}")
    break
for edge in graph_reloaded_05mm.edges.values():
    for edge_feature in edge:
        print(f"{edge_feature}: {edge[edge_feature]}, type: {type(edge[edge_feature])}, expected type: {graph_reloaded_05mm._simple_centerline_edge_attributes_types_dict[edge_feature]}")
    break


hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph_reloaded_05mm)