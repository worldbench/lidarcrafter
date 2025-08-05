from pyvis.network import Network
import networkx as nx

def generate_graph(dataset, graph_dict):
    G = nx.MultiDiGraph()
    
    box_names = list(graph_dict['gt_names'])
    box_names = dataset.get_unique_names(box_names)
    for i, name in enumerate(box_names):
        G.add_node(i, 
                   label=name, 
                   title=f"<b>{name}</b>",
                   group=name,
                   value=1)

    triples = graph_dict['gt_box_relationships']
    for idx, (src, rel_idx, dst) in enumerate(triples):
        src, dst = int(src), int(dst)
        rel = dataset.relationships[rel_idx]
        G.add_edge(src, dst, key=idx, label=rel, title=rel, font={"align": "middle"})

    net = Network(
        height="700px", width="100%", 
        directed=True,      # 有向图
        notebook=False,      # 如果在 notebook 环境显示可改为 True
        bgcolor="#1e1e2f",   # 深色背景
        font_color="white",  # 全局字体白色
    )
    
    # 5. 导入 NetworkX 图
    net.from_nx(G)

    for node in net.nodes:
        nid = node["id"]
        deg = G.degree(nid)
        node["size"] = 20 + deg * 5  # 基础 15，度越大越大
        node["font"] = {"color": "white", "strokeWidth": 0}

    # 统一边样式（箭头、平滑）
    net.set_edge_smooth("dynamic")  # 平滑边
    for edge in net.edges:
        edge["arrows"] = "to"
        # edge["color"] = {"color": "#aaaaff", "highlight": "#ffffff", "hover": "#ffffff"}
        edge["font"] = {"color": "white", "size": 18, "align": "top", "strokeWidth": 0}

    # 6. 调整力导向布局参数，节点间距、弹簧长度等，可根据数据规模再微调
    net.repulsion(
        node_distance=200,    # 节点间理想距离
        central_gravity=0.1,  # 向中心的引力
        spring_length=200,    # 边的弹簧长度
        spring_strength=0.08,
        damping=0.09
    )

    # 7. 显示控制面板，可筛选 physics、nodes、edges，便于实时调参
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    # 8. 输出 HTML
    net.show("dynamic_graph.html", notebook=False)
