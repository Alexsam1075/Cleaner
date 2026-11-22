"""Zero-shot prompt families for bulk (pile of many loose pieces) vs solid (one rigid object).
Each family: (bulk prompts, solid prompts). Families are generic (no reference to specific training objects)."""

TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor.", "a photo of {} lying on the floor."]

FAMILIES = {
    # generic wording of the concept
    "pile": (["a pile of small loose pieces", "scattered granular material", "a heap of many small items", "loose debris"],
             ["a single solid object", "one rigid item", "a single whole object", "one intact item"]),
    "many": (["many small objects", "several items", "multiple pieces", "a group of objects", "a bunch of things"],
             ["one object", "a single item", "one thing", "a single object"]),
    "count": (["a photo of several objects on the floor", "a photo of many pieces on the floor", "a photo of a pile on the floor",
               "a photo of scattered items on the floor"],
              ["a photo of one object on the floor", "a photo of a single item on the floor", "a photo of a single object lying on the floor"]),
    "mix": (["a pile of small loose pieces", "many small objects", "several items", "scattered granular material",
             "a heap of many small items", "multiple pieces", "a bunch of things"],
            ["a single solid object", "one rigid item", "one object", "a single item", "a single whole object"]),
    "multi": (["multiple separate pieces", "a collection of several loose parts", "items piled on top of each other", "a jumble of parts"],
              ["a single object", "one item", "a single piece"]),
    "crumpled": (["crumpled and scattered material", "a messy heap", "a heap of loose material", "a mess of pieces"],
                 ["a neat single object", "one clean rigid object", "a single intact item"]),
    "parts": (["spare parts", "components", "fragments", "pieces", "chips", "granules", "scraps"],
              ["a product", "a tool", "a container", "a box", "a bottle", "a block"]),
    "trash": (["a pile of trash", "scattered garbage", "spilled debris", "loose waste material"],
              ["a piece of trash", "a discarded object", "a single item of garbage"]),
    # material-specific piles vs single objects of the same material
    "matpile": (["a pile of screws", "a pile of stones", "a heap of sawdust", "a pile of wood chips", "a pile of plastic parts",
                 "several crumpled paper balls", "a stack of aluminium profiles", "a bunch of small bottles", "a pile of cables and plastic parts"],
                ["a screw", "a stone", "a wooden block", "a plastic bottle", "a cardboard box", "a crumpled paper ball", "an aluminium bar",
                 "a bottle", "a plastic part"]),
    "paperpile": (["several crumpled sheets of paper", "a pile of crumpled paper", "many wads of paper", "crumpled paper scattered around",
                   "a heap of torn paper"],
                  ["one crumpled sheet of paper", "a single wad of paper", "a cardboard box", "a paper roll", "a paper bag"]),
    "metalpile": (["a pile of screws and bolts", "several metal bars", "a bundle of aluminium profiles", "a heap of metal scraps",
                   "several metal rods lying together", "a stack of metal rails"],
                  ["a metal can", "a single metal bar", "a metal plate", "a metal box", "one aluminium profile", "a metal bracket"]),
    "plasticpile": (["a pile of plastic parts", "several plastic pieces", "many small plastic bottles", "a heap of plastic fragments",
                     "several plastic boxes stacked together", "plastic scraps"],
                    ["a plastic bottle", "a plastic box", "a plastic bag", "a plastic container", "a plastic tray", "a plastic casing"]),
    "woodpile": (["a pile of wood chips", "a heap of sawdust", "wood shavings", "several wooden sticks"],
                 ["a wooden plank", "a wooden block", "a piece of timber", "a wooden board"]),
    "stonepile": (["a pile of gravel", "a heap of sand", "several stones", "a pile of rubble"],
                  ["a rock", "a stone", "a brick", "a concrete block"]),
    # count words only
    "several": (["several", "many", "a few", "multiple", "lots of", "a pile of", "a heap of"], ["a single", "one", "a", "an individual", "a lone"]),
    "plural": (["objects", "items", "pieces", "things", "parts"], ["an object", "an item", "a piece", "a thing", "a part"]),
    # assembly / stack vs monolithic
    "stack": (["a stack of several identical items", "identical pieces stacked together", "a bundle of similar parts",
               "several copies of the same part"],
              ["a single manufactured object", "one monolithic object", "a single piece of equipment", "one whole product"]),
    "loose": (["loose material that can be swept", "material that would spill if pushed", "a loose heap that falls apart when moved"],
              ["a rigid object that can be picked up in one piece", "an object that keeps its shape when moved", "one solid body"]),
    "shape": (["an irregular heap with no definite shape", "a shapeless mound of material", "a scattered mess"],
              ["an object with a definite shape", "a compact object with clear edges", "a single geometric object"]),
    "robot": (["debris that a cleaning robot would sweep up", "bulk material to be collected with a scoop"],
              ["an item a robot would pick up with a gripper", "a single object to be grasped"]),
}

FAMILY_GROUPS = {
    "zs": ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash"),
    "zsM": ("matpile", "paperpile", "metalpile", "plasticpile", "woodpile", "stonepile"),
    "zsC": ("several", "plural", "stack", "loose", "shape", "robot"),
    "zsAll": tuple(FAMILIES),
}
