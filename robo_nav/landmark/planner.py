"""Landmark planner prompt and answer schema, shared by the MuJoCo and Habitat navigators."""

from typing import List

from robo_nav.claude import obj_schema

PLAN_SCHEMA = obj_schema({
    "observation": {"type": "string", "description": "what the robot camera shows right now"},
    "where_am_i": {"type": "string", "description": "position relative to landmarks in the map, or 'unsure'"},
    "action": {"type": "string", "enum": ["instruct", "look_around", "arrived", "give_up"]},
    "instruction": {"type": "string", "description": "the single next instruction for the driving policy ('' unless action is instruct)"},
    "plan": {"type": "array", "items": {"type": "string"}, "description": "remaining landmark steps after this one"},
})


def planner_system(map_md: str, camera: str = "about 20 cm above the floor") -> List[dict]:
    text = (
        "You guide a small wheeled robot through a building using only landmarks. Its camera is "
        f"{camera} and faces forward. You never know its coordinates; work out where it is "
        "from what the camera shows and the landmark map below, which you wrote earlier from a video of a "
        "walk through this building.\n\n"
        "A local driving policy executes your instructions. Give it one short instruction at a time that "
        "covers one landmark or a few meters, e.g. 'Go forward past the grey couch toward the wooden "
        "doors.', 'Turn left and go through the open doorway.', 'Go to the white fridge and stop.' It "
        "cannot see behind itself: to go somewhere behind, start the instruction with 'Turn around'. It "
        "stops when it thinks the instruction is done, then you see a new camera view.\n\n"
        "Rules: describe only physical things -- furniture, appliances, doors, wall corners, colors. Never "
        "name rooms or room types. Use action 'look_around' to get eight views (45 degrees apart) when you "
        "cannot tell where the robot is, 'arrived' only when the goal is right in front of the robot "
        "(within about a meter), and 'give_up' only if the goal cannot be found.\n\n# Landmark map\n\n" + map_md)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
