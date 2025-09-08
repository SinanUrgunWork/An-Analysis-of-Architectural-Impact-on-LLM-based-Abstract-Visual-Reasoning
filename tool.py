import cv2
import numpy as np
from PIL import Image
import math
import os
import re

COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
COLOR_MIN = 0
COLOR_MAX = len(COLOR_VALUES) - 1

WHITE = 0
VERY_LIGHT_GRAY = 1
LIGHT_GRAY = 2
MEDIUM_LIGHT_GRAY = 3
MEDIUM_GRAY = 4
MEDIUM_DARK_GRAY = 5
DARK_GRAY = 6
VERY_DARK_GRAY = 7
ALMOST_BLACK = 8
BLACK = 9

class RAVENAnswerGenerator:
    def __init__(self, image_size=160):
        self.image_size = image_size
        self.center = (image_size // 2, self.image_size // 2)
        self.default_width = 2

    def generate_answer_panel(self, structure):
        canvas = np.ones((self.image_size, self.image_size), np.uint8) * 255
        background = np.zeros((self.image_size, self.image_size), np.uint8)
        if structure.get('type') != 'Singleton':
            structure_img = self._render_structure(structure['type'])
            background = self._layer_add(background, structure_img)
        for component in structure.get('components', []):
            component_img = self._render_component(component)
            background = self._layer_add(background, component_img)
        return canvas - background

    def _render_structure(self, structure_type):
        img = np.zeros((self.image_size, self.image_size), np.uint8)
        if structure_type == "Left_Right":
            img[:, self.image_size // 2] = 255
        elif structure_type == "Up_Down":
            img[self.image_size // 2, :] = 255
        elif structure_type == "Out_In":
            border = 3
            img[border:-border, border] = 255
            img[border:-border, -border] = 255
            img[border, border:-border] = 255
            img[-border, border:-border] = 255
        return img

    def _render_component(self, component):
        component_img = np.zeros((self.image_size, self.image_size), np.uint8)
        layout = component['layout']
        layout_type = layout['type']
        positions = self._get_layout_positions(layout_type, component.get('name'))
        if 'filled_positions' in layout:
            positions = [positions[i] for i in layout['filled_positions'] if i < len(positions)]
        entities = layout.get('entities', [])
        for i, pos in enumerate(positions):
            if i < len(entities):
                entity = entities[i]
                entity_img = self._render_entity(entity, pos, layout_type, component.get('name', None))
                component_img = self._layer_add(component_img, entity_img)
        return component_img

    def _get_layout_positions(self, layout_type, component_name=None):
        positions = []
        if layout_type == "Center_Single":
            if component_name == "Left":
                positions = [(self.image_size // 4, self.image_size // 2)]
            elif component_name == "Right":
                positions = [(3 * self.image_size // 4, self.image_size // 2)]
            elif component_name == "Up":
                positions = [(self.image_size // 2, self.image_size // 4)]
            elif component_name == "Down":
                positions = [(self.image_size // 2, 3 * self.image_size // 4)]
            elif component_name in ["In", "Inner"]:
                positions = [self.center]
            else:
                positions = [self.center]
        elif layout_type == "Distribute_Three":
            positions = [
                (self.image_size // 2, int(self.image_size * 0.25)),
                (int(self.image_size * 0.25), int(self.image_size * 0.75)),
                (int(self.image_size * 0.75), int(self.image_size * 0.75))
            ]
        elif layout_type == "Distribute_Four":
            base_positions = [(0.26, 0.26), (0.74, 0.26), (0.26, 0.74), (0.74, 0.74)]
            positions = [(int(x * self.image_size), int(y * self.image_size))
                         for x, y in base_positions]
        elif layout_type == "Distribute_Nine":
            base_positions = []
            for row in range(3):
                for col in range(3):
                    x = 0.18 + col * 0.32  
                    y = 0.18 + row * 0.32
                    base_positions.append((x, y))
            positions = [(int(x * self.image_size), int(y * self.image_size))
                         for x, y in base_positions]
        return positions

    def _render_entity(self, entity, position, layout_type, component_name=None):
        if layout_type == "Center_Single" and component_name in ["Left", "Right", "Up", "Down"]:
            size_multiplier = 0.92
            max_size = (self.image_size // 4) - 4
        elif layout_type == "Center_Single":
            size_multiplier = 1.0
            max_size = (self.image_size // 2) - 8
        elif layout_type == "Distribute_Three":
            size_multiplier = 0.92
            max_size = min(self.image_size // 2, 55)-8
        elif layout_type == "Distribute_Four":
            size_multiplier = 0.90
            max_size = min(self.image_size // 2, 54)-8
        elif layout_type == "Distribute_Nine":
            size_multiplier = 0.85
            max_size = min(self.image_size // 3, 36)-8
        else:
            size_multiplier = 1.0
            max_size = (self.image_size // 2) - 8

        return self._render_shape(
            entity['type'],
            position,
            entity['size'] * size_multiplier,
            entity['color'],
            entity.get('angle', 0),
            max_size
        )

    def _render_shape(self, shape_type, center, size, color, angle=0, max_size=50):
        img = np.zeros((self.image_size, self.image_size), np.uint8)
        actual_size = max_size * size
        actual_size = max(5, min(actual_size, max_size))
        if isinstance(color, int) and 0 <= color <= COLOR_MAX:
            color_value = COLOR_VALUES[color]
        elif isinstance(color, int) and 10 < color <= 255:
            color_value = 255 - color
        else:
            color_value = 128
        if shape_type == "none":
            return img
        elif shape_type == "circle":
            radius = int(actual_size)
            if color != 0:
                cv2.circle(img, center, radius, 255 - color_value, -1)
                cv2.circle(img, center, radius, 255, self.default_width)
            else:
                cv2.circle(img, center, radius, 255, self.default_width)
        elif shape_type == "square":
            dl = int(actual_size )  # scaling ile kareyi büyüt!
            pts = []
            for i in range(4):
                base_angle = (i * 90) - 45 - angle
                angle_rad = base_angle * math.pi / 180
                x = center[0] + int(dl * math.cos(angle_rad))
                y = center[1] + int(dl * math.sin(angle_rad))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillPoly(img, [pts], 255 - color_value)
            cv2.polylines(img, [pts], True, 255, self.default_width)
        elif shape_type == "triangle":
            dl = int(actual_size)
            pts = []
            for i in range(3):
                base_angle = (i * 120 - 90) - angle
                angle_rad = base_angle * math.pi / 180
                x = center[0] + int(dl * math.cos(angle_rad))
                y = center[1] + int(dl * math.sin(angle_rad))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillPoly(img, [pts], 255 - color_value)
            cv2.polylines(img, [pts], True, 255, self.default_width)
        elif shape_type == "pentagon":
            dl = int(actual_size)
            pts = []
            for i in range(5):
                base_angle = (i * 72 - 90) - angle
                angle_rad = base_angle * math.pi / 180
                x = center[0] + int(dl * math.cos(angle_rad))
                y = center[1] + int(dl * math.sin(angle_rad))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillPoly(img, [pts], 255 - color_value)
            cv2.polylines(img, [pts], True, 255, self.default_width)
        elif shape_type == "heptagon":
            dl = int(actual_size)
            pts = []
            for i in range(7):
                base_angle = (i * 51.43 - 90) - angle
                angle_rad = base_angle * math.pi / 180
                x = center[0] + int(dl * math.cos(angle_rad))
                y = center[1] + int(dl * math.sin(angle_rad))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillPoly(img, [pts], 255 - color_value)
            cv2.polylines(img, [pts], True, 255, self.default_width)
        elif shape_type == "hexagon":
            dl = int(actual_size)
            pts = []
            for i in range(6):
                base_angle = (i * 60 - 90) - angle
                angle_rad = base_angle * math.pi / 180
                x = center[0] + int(dl * math.cos(angle_rad))
                y = center[1] + int(dl * math.sin(angle_rad))
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillPoly(img, [pts], 255 - color_value)
            cv2.polylines(img, [pts], True, 255, self.default_width)
        elif shape_type == "line":
            length = int(actual_size)
            angle_rad = angle * math.pi / 180
            x1 = center[0] - int(length * math.cos(angle_rad) / 2)
            y1 = center[1] - int(length * math.sin(angle_rad) / 2)
            x2 = center[0] + int(length * math.cos(angle_rad) / 2)
            y2 = center[1] + int(length * math.sin(angle_rad) / 2)
            cv2.line(img, (x1, y1), (x2, y2), 255, self.default_width)
        return img

    def _layer_add(self, lower_layer, higher_layer):
        lower_layer[higher_layer > 0] = 0
        return lower_layer + higher_layer

    def save_panel(self, panel, filename):
        img = Image.fromarray(panel)
        img.save(filename)

def generate_visual_panel(config_type, objects=None, grid_layout=None, positions=None, 
                         empty_positions=None, source_filename=None):
    try:
        generator = RAVENAnswerGenerator()
        if objects:
            objects = _clean_objects(objects)
        if grid_layout:
            structure = _create_grid_structure(grid_layout, objects, positions, empty_positions)
        else:
            structure = _create_simple_structure(config_type, objects)
        panel = generator.generate_answer_panel(structure)
        output_filename = _generate_filename(source_filename, config_type)
        generator.save_panel(panel, output_filename)
        return {
            "status": "success",
            "message": f"Answer panel generated successfully",
            "filename": output_filename,
            "config_type": config_type,
            "objects_count": len(objects) if objects else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate panel: {str(e)}"
        }

def _clean_objects(objects):
    for obj in objects:
        if 'angle' in obj:
            try:
                obj['angle'] = int(float(obj['angle']) % 360)
                if obj['angle'] > 180:
                    obj['angle'] -= 360
            except:
                obj['angle'] = 0
        else:
            obj['angle'] = 0
        if 'size' in obj:
            try:
                obj['size'] = max(0.1, min(1.0, float(obj['size'])))
            except:
                obj['size'] = 0.5
        else:
            obj['size'] = 0.5
        if 'color' in obj:
            try:
                color_val = obj['color']
                if isinstance(color_val, str):
                    color_map = {
                        'white': WHITE, 'very_light_gray': VERY_LIGHT_GRAY,
                        'light_gray': LIGHT_GRAY, 'medium_light_gray': MEDIUM_LIGHT_GRAY,
                        'medium_gray': MEDIUM_GRAY, 'medium_dark_gray': MEDIUM_DARK_GRAY,
                        'dark_gray': DARK_GRAY, 'very_dark_gray': VERY_DARK_GRAY,
                        'almost_black': ALMOST_BLACK, 'black': BLACK
                    }
                    obj['color'] = color_map.get(color_val.lower(), MEDIUM_GRAY)
                else:
                    color_val = int(color_val)
                    if color_val > COLOR_MAX:
                        distances = [abs(color_val - cv) for cv in COLOR_VALUES]
                        obj['color'] = distances.index(min(distances))
                    else:
                        obj['color'] = max(0, min(COLOR_MAX, color_val))
            except:
                obj['color'] = MEDIUM_GRAY
        else:
            obj['color'] = MEDIUM_GRAY
        valid_shapes = ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'circle', 'line', 'none']
        if 'shape' not in obj or obj['shape'] not in valid_shapes:
            obj['shape'] = 'circle'
        if 'type' not in obj and 'shape' in obj:
            obj['type'] = obj['shape']
    return objects

def _create_grid_structure(grid_layout, objects, positions, empty_positions):
    layout_map = {
        '1x1': 'Center_Single',
        '1x2': 'Left_Right',
        '2x1': 'Up_Down', 
        '2x2': 'Distribute_Four',
        '3x3': 'Distribute_Nine',
        '1x3': 'Distribute_Three',
        '3x1': 'Distribute_Three'
    }
    layout_type = layout_map.get(grid_layout, 'Center_Single')
    if positions is None and objects:
        positions = list(range(len(objects)))
    elif positions is None:
        positions = [0]
    if empty_positions:
        positions = [p for p in positions if p not in empty_positions]
    entities = objects or []
    if layout_type == 'Left_Right':
        return {
            'type': 'Left_Right',
            'components': [
                {'name': 'Left', 'layout': {'type': 'Center_Single', 'entities': [entities[0]] if entities else []}},
                {'name': 'Right', 'layout': {'type': 'Center_Single', 'entities': [entities[1]] if len(entities) > 1 else []}}
            ]
        }
    elif layout_type == 'Up_Down':
        return {
            'type': 'Up_Down',
            'components': [
                {'name': 'Up', 'layout': {'type': 'Center_Single', 'entities': [entities[0]] if entities else []}},
                {'name': 'Down', 'layout': {'type': 'Center_Single', 'entities': [entities[1]] if len(entities) > 1 else []}}
            ]
        }
    else:
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {
                    'type': layout_type,
                    'entities': entities,
                    'filled_positions': positions
                }
            }]
        }

def _create_simple_structure(config_type, objects):
    if not objects:
        objects = [{'type': 'circle', 'size': 0.5, 'color': MEDIUM_GRAY, 'angle': 0, 'shape': 'circle'}]
    if config_type == "singleton_center":
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {
                    'type': 'Center_Single',
                    'entities': [{
                        'type': objects[0]['shape'],
                        'size': objects[0]['size'],
                        'color': objects[0]['color'],
                        'angle': objects[0]['angle']
                    }]
                }
            }]
        }
    elif config_type == "left_right" and len(objects) >= 2:
        return {
            'type': 'Left_Right',
            'components': [
                {'name': 'Left', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[0]['shape'],
                    'size': objects[0]['size'],
                    'color': objects[0]['color'],
                    'angle': objects[0]['angle']
                }]}},
                {'name': 'Right', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[1]['shape'],
                    'size': objects[1]['size'],
                    'color': objects[1]['color'],
                    'angle': objects[1]['angle']
                }]}}
            ]
        }
    elif config_type == "up_down" and len(objects) >= 2:
        return {
            'type': 'Up_Down',
            'components': [
                {'name': 'Up', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[0]['shape'],
                    'size': objects[0]['size'],
                    'color': objects[0]['color'],
                    'angle': objects[0]['angle']
                }]}},
                {'name': 'Down', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[1]['shape'],
                    'size': objects[1]['size'],
                    'color': objects[1]['color'],
                    'angle': objects[1]['angle']
                }]}}
            ]
        }
    elif config_type == "out_in" and len(objects) >= 2:
        return {
            'type': 'Out_In',
            'components': [
                {'name': 'Out', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[0]['shape'],
                    'size': objects[0]['size'],
                    'color': objects[0]['color'],
                    'angle': objects[0]['angle']
                }]}},
                {'name': 'In', 'layout': {'type': 'Center_Single', 'entities': [{
                    'type': objects[1]['shape'],
                    'size': objects[1]['size'],
                    'color': objects[1]['color'],
                    'angle': objects[1]['angle']
                }]}}
            ]
        }
    elif config_type in ["grid_2x2", "distribute_four"]:
        entities = [{
            'type': obj['shape'],
            'size': obj['size'],
            'color': obj['color'],
            'angle': obj['angle']
        } for obj in objects]
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {'type': 'Distribute_Four', 'entities': entities}
            }]
        }
    elif config_type in ["grid_3x3", "distribute_nine"]:
        entities = [{
            'type': obj['shape'],
            'size': obj['size'],
            'color': obj['color'],
            'angle': obj['angle']
        } for obj in objects]
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {'type': 'Distribute_Nine', 'entities': entities}
            }]
        }
    elif config_type == "distribute_three":
        entities = [{
            'type': obj['shape'],
            'size': obj['size'],
            'color': obj['color'],
            'angle': obj['angle']
        } for obj in objects[:3]]
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {'type': 'Distribute_Three', 'entities': entities}
            }]
        }
    else:
        return {
            'type': 'Singleton',
            'components': [{
                'name': 'Grid',
                'layout': {
                    'type': 'Center_Single',
                    'entities': [{
                        'type': objects[0]['shape'],
                        'size': objects[0]['size'],
                        'color': objects[0]['color'],
                        'angle': objects[0]['angle']
                    }]
                }
            }]
        }

def _generate_filename(source_filename, config_type):
    if source_filename:
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        number_match = re.search(r'\d+', base_name)
        if number_match:
            question_number = number_match.group()
            return f"question{question_number}LLMAnswer.png"
        else:
            return f"{base_name}LLMAnswer.png"
    else:
        return f"{config_type}_answer.png"


# Add this to your tool.py file - PERFECT TOOL DESCRIPTION

# Add this to your tool.py file - PERFECT TOOL DESCRIPTION

tool_functions = [
    {
        "type": "function",
        "function": {
            "name": "generate_visual_panel",
            "description": """Generate a RAVEN matrix answer panel with specified configuration and objects.

CRITICAL USAGE RULES:
1. GRID_3X3/DISTRIBUTE_NINE: Must provide exactly 9 objects (including 'none' for empty positions)
2. LEFT_RIGHT/UP_DOWN: Must provide exactly 2 objects
3. OUT_IN: Must provide exactly 2 objects (first=outer, second=inner, both centered)
4. DISTRIBUTE_FOUR/GRID_2X2: Provide up to 4 objects
5. DISTRIBUTE_THREE: Provide up to 3 objects
6. SINGLETON_CENTER: Provide exactly 1 object

GRID POSITION MAPPING (for 3x3 grids):
Position indices 0-8 map to grid locations:
0 1 2  (top row)
3 4 5  (middle row)  
6 7 8  (bottom row)

EMPTY POSITIONS: Use shape='none', color=0, size=0.4, angle=0

COLOR SYSTEM: 
0=white, 1=very_light_gray, 2=light_gray, 3=medium_light_gray, 4=medium_gray, 
5=medium_dark_gray, 6=dark_gray, 7=very_dark_gray, 8=almost_black, 9=black

SIZE SCALING: Tool automatically scales based on layout:
- singleton_center: 1.0x multiplier  
- left_right/up_down: 0.92x multiplier
- distribute_three: 0.92x multiplier
- distribute_four: 0.90x multiplier  
- distribute_nine: 0.85x multiplier

EXAMPLES:
- Single object: config_type="singleton_center", objects=[{"shape":"circle","size":0.6,"color":5,"angle":0}]
- Two side-by-side: config_type="left_right", objects=[{"shape":"triangle","size":0.5,"color":3,"angle":0},{"shape":"square","size":0.5,"color":7,"angle":45}]
- Two nested (outer+inner): config_type="out_in", objects=[{"shape":"circle","size":0.8,"color":2,"angle":0},{"shape":"triangle","size":0.4,"color":8,"angle":0}]
- 3x3 with empty spots: config_type="grid_3x3", objects=[9 objects total, use "none" for empty positions]
- Specific positions: use positions=[0,4,8] to fill only those grid locations
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_type": {
                        "type": "string",
                        "enum": [
                            "singleton_center",
                            "left_right", 
                            "up_down",
                            "out_in",
                            "distribute_three",
                            "distribute_four",
                            "grid_2x2",
                            "distribute_nine", 
                            "grid_3x3"
                        ],
                        "description": "Layout configuration type. grid_2x2=distribute_four, grid_3x3=distribute_nine"
                    },
                    "objects": {
                        "type": "array",
                        "maxItems": 9,
                        "items": {
                            "type": "object",
                            "properties": {
                                "shape": {
                                    "type": "string",
                                    "enum": ["triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"],
                                    "description": "Shape type. Use 'none' for empty grid positions"
                                },
                                "size": {
                                    "type": "number",
                                    "minimum": 0.1,
                                    "maximum": 1.0,
                                    "description": "Size multiplier (0.1-1.0). Tool applies additional scaling based on layout"
                                },
                                "color": {
                                    "type": "integer", 
                                    "minimum": 0,
                                    "maximum": 9,
                                    "description": "Grayscale color: 0=white, 5=medium_gray, 9=black"
                                },
                                "angle": {
                                    "type": "integer",
                                    "minimum": -180,
                                    "maximum": 180,
                                    "description": "Rotation angle in degrees. 0=default orientation"
                                }
                            },
                            "required": ["shape", "size", "color", "angle"],
                            "additionalProperties": False
                        },
                        "description": "Array of objects. For grid_3x3: exactly 9 objects required. For left_right/up_down/out_in: exactly 2 objects required"
                    },
                    "grid_layout": {
                        "type": "string",
                        "enum": ["1x1", "1x2", "2x1", "2x2", "3x3", "1x3", "3x1"],
                        "description": "Alternative to config_type. Maps: 1x1=singleton_center, 1x2=left_right, 2x1=up_down, 2x2=distribute_four, 3x3=distribute_nine"
                    },
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8
                        },
                        "uniqueItems": True,
                        "description": "Grid indices to fill (0-8). For 3x3: 0,1,2=top row, 3,4,5=middle, 6,7,8=bottom. Use with fewer objects than grid size"
                    },
                    "empty_positions": {
                        "type": "array", 
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8
                        },
                        "uniqueItems": True,
                        "description": "Grid indices to leave empty. Alternative to using 'none' shapes"
                    }
                },
                "required": ["config_type", "objects"],
                "additionalProperties": False
            }
        }
    }
]

def execute_tool_function(function_name, arguments, source_filename=None):
    """Execute the specified tool function with given arguments and enhanced validation"""
    if function_name == "generate_visual_panel":
        try:
            # Validate object count based on config_type
            config_type = arguments.get("config_type", "")
            objects = arguments.get("objects", [])
            
            # Strict validation for specific config types
            if config_type == "singleton_center" and len(objects) != 1:
                return {"status": "error", "message": "singleton_center requires exactly 1 object"}
            elif config_type in ["left_right", "up_down", "out_in"] and len(objects) != 2:
                return {"status": "error", "message": f"{config_type} requires exactly 2 objects"}
            elif config_type in ["grid_3x3", "distribute_nine"]:
                # For 3x3 grids, if using positions parameter, allow fewer objects
                if "positions" not in arguments and len(objects) != 9:
                    return {"status": "error", "message": "grid_3x3 requires exactly 9 objects (use 'none' for empty positions)"}
            
            # Validate shapes
            valid_shapes = ["triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"]
            for i, obj in enumerate(objects):
                if obj.get("shape") not in valid_shapes:
                    return {"status": "error", "message": f"Object {i}: invalid shape '{obj.get('shape')}'. Use: {valid_shapes}"}
                
                # Validate color range
                color = obj.get("color")
                if not isinstance(color, int) or color < 0 or color > 9:
                    return {"status": "error", "message": f"Object {i}: color must be integer 0-9, got {color}"}
                
                # Validate size range  
                size = obj.get("size")
                if not isinstance(size, (int, float)) or size < 0.1 or size > 1.0:
                    return {"status": "error", "message": f"Object {i}: size must be float 0.1-1.0, got {size}"}
                
                # Validate angle
                angle = obj.get("angle")
                if not isinstance(angle, int) or angle < -180 or angle > 180:
                    return {"status": "error", "message": f"Object {i}: angle must be integer -180 to 180, got {angle}"}
            
            # Add source filename for output naming
            if source_filename:
                arguments["source_filename"] = source_filename
                
            # Call the actual function
            result = generate_visual_panel(**arguments)
            
            #success message
            if result.get("status") == "success":
                result["validation"] = {
                    "config_type": config_type,
                    "object_count": len(objects),
                    "valid_shapes": all(obj.get("shape") in valid_shapes for obj in objects),
                    "all_parameters_valid": True
                }
            
            return result
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Tool execution failed: {str(e)}",
                "arguments_received": arguments
            }
    else:
        return {
            "status": "error", 
            "message": f"Unknown function: {function_name}. Available: generate_visual_panel"
        }
