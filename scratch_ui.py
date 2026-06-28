import networkx as nx
import plotly.graph_objects as go

def generate_network_map(reports):
    G = nx.Graph()
    for i in range(1, 67):
        G.add_node(i)
    # create some edges
    for i in range(1, 66):
        G.add_edge(i, i+1)
        if i % 10 == 0:
            if i + 10 <= 66:
                G.add_edge(i, i+10)
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for r in reports:
        x, y = pos[r.link_id]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(r.congestion_color)
        node_text.append(f"Link {r.link_id}<br>Health: {r.health_score}<br>{r.congestion_level}")
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
            line_width=2))
            
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Traffic Network Map',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
             ))
    return fig
