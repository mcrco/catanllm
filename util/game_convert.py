"""
Convert Catanatron game state to natural language for LLM processing.
"""

from catanatron import Game, ActionType
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.models.map import Port, PORT_DIRECTION_TO_NODEREFS
from catanatron.state_functions import (
    get_player_freqdeck,
    get_visible_victory_points,
)
from catanatron.models.board import STATIC_GRAPH


def game_to_natural_language(game: Game, current_player_color, verbose = False) -> str:
    """
    Convert a Catanatron game instance to natural language description.

    Args:
        game: The Catanatron game instance
        current_player_color: The color of the current player making decisions
        verbose: Whether to include detailed information about the game state
    Returns:
        str: Natural language description of the game state
    """
    state = game.state
    description = []
    
    # Game overview
    description.append("=== CATAN GAME STATE ===")
    description.append(f"Turn: {state.num_turns}")
    description.append(f"Current player: {current_player_color.name}")
    description.append("")

    # Player states
    description.append("=== PLAYER STATUS ===")
    for i, color in enumerate(state.colors):
        is_current = color == current_player_color
        marker = ">>> " if is_current else "    "

        # Victory points
        visible_vp = get_visible_victory_points(state, color)

        # Resources
        freqdeck = get_player_freqdeck(state, color)
        total_resources = sum(freqdeck)
        # Development cards
        player_index = state.color_to_index[current_player_color]
        dev_cards = []
        dev_cards_count = 0
        for dev_card_name in DEVELOPMENT_CARDS:
            card_key = f"P{player_index}_{dev_card_name}_IN_HAND"
            if card_key in state.player_state and state.player_state[card_key] > 0:
                dev_cards.append(f"{dev_card_name}: {state.player_state[card_key]}")
                dev_cards_count += state.player_state[card_key]

        # Resource breakdown (only show for current player)
        if is_current:
            resources_detail = f" (Wood: {freqdeck[0]}, Brick: {freqdeck[1]}, Sheep: {freqdeck[2]}, Wheat: {freqdeck[3]}, Ore: {freqdeck[4]})"
            if dev_cards:
                resources_detail += f" | Dev Cards: {', '.join(dev_cards)}"
        else:
            resources_detail = ""

        description.append(
            f"{marker}{color.name}: {visible_vp} VP, {total_resources} resources{resources_detail}, {dev_cards_count} dev cards"
        )

    description.append("")
    if verbose:
        # Board coordinate system explanation
        description.append("=== BOARD COORDINATE SYSTEM EXPLANATION ===")
        description.append("SPATIAL RELATIONSHIPS:")
        description.append("- Tiles: Identified by cube coordinates (x,y,z) where x+y+z=0")
        description.append("- Nodes: Intersection points between tiles, identified by integer IDs (0-53)")
        description.append("- Edges/Roads: Connections between two nodes, represented as (node_id1, node_id2)")
        description.append("- Each tile has 6 nodes at its corners and 6 edges connecting those nodes")
        description.append("- Adjacent tiles share nodes and edges")
        description.append("- Ports are located on specific edges facing the ocean")
        description.append("")

    # Full board layout
    description.append("=== BOARD LAYOUT ===")
    description.append(f"Robber position: {state.board.robber_coordinate}")
    description.append("")

    # Hexagon tiles with resources and numbers
    description.append("--- HEXAGON TILES ---")
    for coordinate, tile in sorted(state.board.map.land_tiles.items()):
        if tile.resource is None:
            tile_desc = f"Tile {tile.id}: DESERT (no resources)"
        else:
            tile_desc = f"Tile {tile.id}: {tile.resource} with number {tile.number}"
        
        if verbose:
            # Show which nodes belong to this tile
            tile_nodes = [tile.nodes[noderef] for noderef in tile.nodes.keys()]
            tile_nodes.sort()
            tile_desc += f" | corner nodes: {tile_nodes}"
        
        description.append(tile_desc)
    description.append("")

    # Ports
    description.append("--- PORTS ---")
    for _, tile in sorted(state.board.map.tiles.items()):
        if isinstance(tile, Port):
            port_type = "3:1 Generic" if tile.resource is None else f"2:1 {tile.resource}"
            a_noderef, b_noderef = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            port_nodes = [tile.nodes[a_noderef], tile.nodes[b_noderef]]
            port_nodes.sort()
            description.append(f"Port at {port_nodes[0]}: {port_type}")
            description.append(f"Port at {port_nodes[1]}: {port_type}")
    description.append("")


    if verbose:
        # All nodes and their states
        description.append("--- NODES (Intersection Points) ---")
        for node_id in sorted(state.board.map.land_nodes):
            building_info = state.board.buildings.get(node_id, None)
            if building_info:
                color, building_type = building_info
                building_desc = f"{color.name} {building_type}"
            else:
                building_desc = "Empty"
            
            # Find neighboring nodes (connected by roads)
            neighboring_nodes = list(STATIC_GRAPH.neighbors(node_id))
            neighboring_nodes.sort()
            
            # Find which tiles this node belongs to
            adjacent_tiles = state.board.map.adjacent_tiles.get(node_id, [])
            tile_ids = [str(tile.id) for tile in adjacent_tiles]
            
            # Find existing roads from this node
            connected_roads = []
            for neighbor_id in neighboring_nodes:
                edge = tuple(sorted([node_id, neighbor_id]))
                if edge in state.board.roads:
                    road_color = state.board.roads[edge].name
                    connected_roads.append(f"road to node {neighbor_id} ({road_color})")
            
            neighbor_info = f"neighbors: {neighboring_nodes}" if neighboring_nodes else "no neighbors"
            road_info = f"roads: {', '.join(connected_roads)}" if connected_roads else "no roads"
            
            description.append(f"Node {node_id}: {building_desc} | {neighbor_info} | {road_info} | adjacent to tile IDs: {', '.join(tile_ids)}")
        description.append("")
    else:
        pass

    # All roads
    description.append("--- ROADS (Edge Connections) ---")
    if state.board.roads:
        for edge, color in sorted(state.board.roads.items()):
            node1, node2 = edge
            if node1 < node2:
                description.append(f"Road: {color.name} connecting nodes {node1} ↔ {node2}")
    else:
        description.append("No roads built yet")
    description.append("")

    # Possible building locations
    description.append("--- POSSIBLE BUILDING LOCATIONS ---")
    empty_nodes = [node_id for node_id in sorted(state.board.map.land_nodes) 
                   if node_id not in state.board.buildings]
    
    description.append(f"Available nodes for settlements: {len(empty_nodes)} nodes")
    description.append(f"Empty nodes: {empty_nodes}")
    
    # Show available road connections
    available_edges = []
    for node1 in state.board.map.land_nodes:
        for node2 in STATIC_GRAPH.neighbors(node1):
            if node1 < node2:  # Avoid duplicates
                edge = (node1, node2)
                if edge not in state.board.roads:
                    available_edges.append(edge)
    
    description.append(f"Available road connections: {len(available_edges)} possible roads")
    if len(available_edges) <= 20 or verbose:  # Only show if not too many
        description.append(f"Available road edges: {sorted(available_edges)}")
    description.append("")

    # Buildings summary
    description.append("=== BUILDINGS SUMMARY ===")
    for color in state.colors:
        settlements = []
        cities = []
        roads_count = 0
        
        for node_id, (building_color, building_type) in state.board.buildings.items():
            if building_color == color:
                if building_type == "SETTLEMENT":
                    settlements.append(node_id)
                elif building_type == "CITY":
                    cities.append(node_id)
        
        for edge, road_color in state.board.roads.items():
            if road_color == color:
                roads_count += 1
        
        settlement_list = f" at nodes {settlements}" if settlements else ""
        city_list = f" at nodes {cities}" if cities else ""
        
        description.append(f"{color.name}: {len(settlements)} settlements{settlement_list}, {len(cities)} cities{city_list}, {roads_count} roads")

    description.append("")

    # Current game phase
    if verbose:
        description.append("=== CURRENT SITUATION ===")
        if state.is_discarding:
            description.append("Player must discard cards (rolled 7)")
        elif state.is_moving_knight:
            description.append("Player must move the robber")
        elif state.is_road_building:
            description.append(f"Player is in road building phase ({state.free_roads_available} roads remaining)")
        else:
            description.append("Normal turn phase")

    return "\n".join(description)


def format_playable_actions(playable_actions) -> str:
    """
    Format playable actions into natural language.

    Args:
        playable_actions: List of Action tuples from Catanatron

    Returns:
        str: Natural language description of available actions
    """
    if not playable_actions:
        return "No actions available."

    description = []
    description.append("=== AVAILABLE ACTIONS ===")

    for i, action in enumerate(playable_actions):
        # Action is a namedtuple with (color, action_type, value)
        if hasattr(action, 'action_type'):
            action_type = action.action_type
            action_value = action.value
        else:
            # Fallback for tuple format
            action_type = action[1] if len(action) > 1 else action[0]
            action_value = action[2] if len(action) > 2 else None

        # Convert action to readable format
        if action_type == ActionType.ROLL:
            desc = "Roll dice"
        elif action_type == ActionType.END_TURN:
            desc = "End turn"
        elif action_type == ActionType.BUILD_ROAD:
            if action_value and len(action_value) == 2:
                node1, node2 = action_value
                desc = f"Build road connecting nodes {node1} ↔ {node2}"
            else:
                desc = f"Build road at edge {action_value}"
        elif action_type == ActionType.BUILD_SETTLEMENT:
            desc = f"Build settlement at node {action_value}"
            # Add information about adjacent tiles if possible
            if action_value is not None:
                desc += f" (node {action_value})"
        elif action_type == ActionType.BUILD_CITY:
            desc = f"Build city at node {action_value}"
            # Add information about adjacent tiles if possible  
            if action_value is not None:
                desc += f" (node {action_value})"
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            desc = "Buy development card"
        elif action_type == ActionType.PLAY_KNIGHT_CARD and action_value is not None:
            desc = f"Play knight card, move robber to {action_value[0]}, steal from {action_value[1] if action_value[1] else 'no one'}"
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY and action_value is not None:
            if len(action_value) == 2:
                desc = f"Play Year of Plenty card for {action_value[0]} and {action_value[1]}"
            else:
                desc = f"Play Year of Plenty card for {action_value[0]}"
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            desc = "Play Road Building card"
        elif action_type == ActionType.PLAY_MONOPOLY and action_value is not None:
            desc = f"Play Monopoly card for {action_value}"
        elif action_type == ActionType.DISCARD:
            desc = f"Discard resources: {action_value}"
        elif action_type == ActionType.MOVE_ROBBER and action_value is not None:
            if len(action_value) >= 2:
                desc = f"Move robber to tile {action_value[0]}, steal from {action_value[1] if action_value[1] else 'no one'}"
            else:
                desc = f"Move robber to tile {action_value[0]}"
        elif action_type == ActionType.MARITIME_TRADE:
            # action_value is a tuple like ('SHEEP', 'SHEEP', 'SHEEP', 'SHEEP', 'WOOD')
            # The first 4 are what you give, the last is what you get
            if action_value and len(action_value) == 5:
                give = action_value[0]
                get = action_value[4]
                port_ratio = 4
                if action_value[3] is None:
                    port_ratio = 3
                if action_value[2] is None:
                    port_ratio = 2

                desc = f"Maritime trade: give {port_ratio} {give} for 1 {get} ({port_ratio}:1 port trade)"
            else:
                desc = "Maritime trade"
        else:
            desc = f"Action: {action_type}"

        description.append(f"{i}: {desc}")

    return "\n".join(description)
