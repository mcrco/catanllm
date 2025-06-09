import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from catanatron import Color, RandomPlayer, Game
from catanatron.state import State
from catanatron.models.map import LandTile, Port, PORT_DIRECTION_TO_NODEREFS
from catanatron.models.enums import SETTLEMENT, CITY, NodeRef
from catanatron.models.coordinate_system import Direction

NODE_REF_TO_COORDS = {
    NodeRef.NORTHWEST: (0, 1),
    NodeRef.NORTH: (np.sqrt(3)/2, 0.5),
    NodeRef.NORTHEAST: (np.sqrt(3)/2, -0.5),
    NodeRef.SOUTHEAST: (0, -1),
    NodeRef.SOUTH: (-np.sqrt(3)/2, -0.5),
    NodeRef.SOUTHWEST: (-np.sqrt(3)/2, 0.5),
}

RESOURCE_TO_COLOR = {
    "WOOD": 'forestgreen',
    "BRICK": 'firebrick',
    "SHEEP": 'lightgreen',
    "WHEAT": 'goldenrod',
    "ORE": 'slategray',
    None: 'khaki'
}

PLAYER_COLOR_TO_MPL_COLOR = {
    Color.RED: 'red',
    Color.BLUE: 'blue',
    Color.ORANGE: 'orange',
    Color.WHITE: 'white',
}

def axial_to_cartesian(q, r, size=1):
    x = size * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
    y = size * (3. / 2. * r)
    return x, y

def plot_board(state: State, output_path=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    hex_size=1

    node_id_to_coords = {}
    for coord, tile in state.board.map.tiles.items():
        q, r, s = coord
        x_hex, y_hex = axial_to_cartesian(q, r, hex_size)
        for node_ref, node_id in tile.nodes.items():
            if node_id in state.board.map.node_production and node_id not in node_id_to_coords:
                offset_x, offset_y = NODE_REF_TO_COORDS[node_ref]
                node_id_to_coords[node_id] = (x_hex + offset_x * hex_size, y_hex + offset_y * hex_size)

    # Write node ids on top of each node
    for node_id, (nx, ny) in node_id_to_coords.items():
        ax.text(nx, ny + 0.13, str(node_id), ha='center', va='bottom', fontsize=7, color='black', zorder=20, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # Plot tiles
    for coord, tile in state.board.map.tiles.items():
        q, r, s = coord
        x, y = axial_to_cartesian(q, r, hex_size)
        
        if isinstance(tile, LandTile):
            # Draw hexagon
            hexagon = Polygon([
                (x + hex_size * np.cos(np.deg2rad(angle)), y + hex_size * np.sin(np.deg2rad(angle)))
                for angle in range(30, 360, 60)
            ], closed=True, ec='black', fc=RESOURCE_TO_COLOR.get(getattr(tile, 'resource', None), 'beige'))
            ax.add_patch(hexagon)

            # Add tile info
            if tile.resource:
                ax.text(x, y + 0.2, tile.resource, ha='center', va='center', fontsize=8, weight='bold')
            if tile.number:
                ax.text(x, y, str(tile.number), ha='center', va='center', fontsize=12, weight='bold',
                        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=2))
            
            if state.board.robber_coordinate == coord:
                ax.text(x, y - 0.3, "R", ha='center', va='center', fontsize=20, weight='bold', color='black')

        elif isinstance(tile, Port):
            # Do NOT draw the hexagon for the port tile.
            # Instead, highlight the two port nodes.
            a_noderef, b_noderef = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            node_ids = [tile.nodes[a_noderef], tile.nodes[b_noderef]]
            coords = [node_id_to_coords[nid] for nid in node_ids]
            for cx, cy in coords:
                ax.plot(cx, cy, 'o', markersize=24, color='deepskyblue', markeredgecolor='navy', markeredgewidth=2, zorder=10)
            # Add port text label between the two nodes
            mx, my = np.mean([coords[0][0], coords[1][0]]), np.mean([coords[0][1], coords[1][1]])
            if tile.resource is None:
                port_text = '3:1'
            else:
                port_text = f'2:1 {tile.resource.title()}'
            ax.text(mx, my, port_text, ha='center', va='center', fontsize=10, weight='bold', color='navy', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="navy", lw=1), zorder=11)

    # Plot roads
    for edge, color in state.board.roads.items():
        if edge[0] < edge[1]: # Avoid drawing roads twice
            node1_coords = node_id_to_coords.get(edge[0])
            node2_coords = node_id_to_coords.get(edge[1])
            if node1_coords and node2_coords:
                ax.plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]], 
                        color=PLAYER_COLOR_TO_MPL_COLOR.get(color, 'black'), linewidth=5)

    # Plot buildings
    for node_id, (color, building_type) in state.board.buildings.items():
        coords = node_id_to_coords.get(node_id)
        if coords:
            player_color = PLAYER_COLOR_TO_MPL_COLOR.get(color, 'black')
            if building_type == SETTLEMENT:
                ax.plot(coords[0], coords[1], 's', markersize=12, color=player_color, markeredgecolor='black')
            elif building_type == CITY:
                ax.plot(coords[0], coords[1], '^', markersize=16, color=player_color, markeredgecolor='black')

    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE)
    ]
    game = Game(players)
    
    # Play a few moves to get a more interesting board
    for _ in range(10):
        if not game.winning_color():
            game.play_tick()

    plot_board(game.state, 'catan_board.png')
    print("Board image saved to catan_board.png")