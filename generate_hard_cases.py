""" Han """
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import matplotlib.pyplot as plt
from signal import signal, SIGINT
from constants import WORKSPACE_LIMITS, PUSH_DISTANCE

def handler(signal_received, frame):
    # Handle any cleanup here
    print("SIGINT or CTRL-C detected. Exiting gracefully")
    exit(0)


""" Parameters to specify """
POLYGONS = {
    "concave.urdf": [(0.045, -0.0225), (0.045, 0.0225), (-0.045, 0.0225), (-0.045, -0.0225)],
    "triangle.urdf": [(0.0225, -0.045), (0.0225, 0.045), (-0.0225, 0)],
    "cube.urdf": [(0.0225, -0.0225), (0.0225, 0.0225), (-0.0225, 0.0225), (-0.0225, -0.0225)],
    "half-cube.urdf": [(0.0225, -0.01125), (0.0225, 0.01125), (-0.0225, 0.01125), (-0.0225, -0.01125)],
    "rect.urdf": [(0.0225, -0.045), (0.0225, 0.045), (-0.0225, 0.045), (-0.0225, -0.045)],
    "cylinder.urdf": [(0.022, -0.022), (0.022, 0.022), (-0.022, 0.022), (-0.022, -0.022)],
    # "half-cylinder.urdf": [(0.022, -0.011), (0.022, 0.011), (-0.022, 0.011), (-0.022, -0.011)],
}
CENTERS = {
    "concave.urdf": (0.0, 0.0),
    "triangle.urdf": (0.0, 0.0),
    "cube.urdf": (0.0, 0.0),
    "half-cube.urdf": (0.0, 0.0),
    "rect.urdf": (0.0, 0.0),
    "cylinder.urdf": (0.0, 0.0),
    # "half-cylinder.urdf": (0.0, 0.0),
}
heights = {
    "concave.urdf": 0.024,
    "triangle.urdf": 0.024,
    "cube.urdf": 0.024,
    "half-cube.urdf": 0.024,
    "rect.urdf": 0.024,
    "cylinder.urdf": 0.024,
    # "half-cylinder.urdf": 0.024,
}

color_space = (
    np.asarray(
        [
            [78.0, 121.0, 167.0],  # blue
            [89.0, 161.0, 79.0],  # green
            [156, 117, 95],  # brown
            [242, 142, 43],  # orange
            [237.0, 201.0, 72.0],  # yellow
            [186, 176, 172],  # gray
            [255.0, 87.0, 89.0],  # red
            [176, 122, 161],  # purple
            [118, 183, 178],  # cyan
            [255, 157, 167],  # pink
        ]
    )
    / 255.0
)


def generate_unit_scenario(shapes: list) -> np.ndarray:
    """Randomly generate challenging VPG test scenarios.
    Params:
        shapes (list): shapes of objects in the generated scene.
    Return:
        np.ndarray: each row represents the 2d pose for a shape.
    """
    # Find polygons of all shapes
    polys = [np.array(POLYGONS[s], dtype=np.float64) for s in shapes]
    centers = [np.array(CENTERS[s], dtype=np.float64) for s in shapes]
    configs = [[centers[0][0], centers[0][1], 0]]
    # We start with an initial shape and build up
    meta_poly = Polygon(polys[0])
    # Iterate through all polygons and add them to meta_poly
    for j, p in enumerate(polys):
        if j == 0:
            continue
        # Randomly find an edge on meta_poly to attach polygon
        coords = np.transpose(meta_poly.exterior.coords.xy)
        matched = False
        count = 0
        while not matched:
            count += 1
            if count > 100:
                return None
            # Looking for an edge
            index = np.random.randint(0, len(coords))
            start_pt = coords[index]
            # Looking for an arbitrary rotation
            angle = np.random.randint(0, 4) * np.pi / 2
            angle = np.random.randint(0, 2) * np.pi - np.pi
            rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Transitions to match the new polygon to the existing shape
            center = np.copy(centers[j])
            poly = np.copy(p)
            center -= poly[0]
            poly -= poly[0]
            for i in range(len(poly)):
                poly[i] = np.dot(rotation_matrix, np.transpose(poly[i]))
            center = np.dot(rotation_matrix, np.transpose(center))
            pt = np.random.randint(poly.shape[0])
            center -= poly[pt]
            poly -= poly[pt]
            poly += start_pt
            center += start_pt
            if (
                center[0, 0] + 0.5 < WORKSPACE_LIMITS[0][0] + PUSH_DISTANCE
                or center[0, 0] + 0.5 > WORKSPACE_LIMITS[0][1] - PUSH_DISTANCE
                or center[0, 1] < WORKSPACE_LIMITS[1][0] + PUSH_DISTANCE
                or center[0, 1] > WORKSPACE_LIMITS[1][1] - PUSH_DISTANCE
            ):
                continue
            # Check if the generated pose suggests a hard case
            suggested_poly = Polygon(poly)
            if meta_poly.intersects(suggested_poly):
                if (
                    type(meta_poly.intersection(suggested_poly)) is Polygon
                    and meta_poly.intersection(suggested_poly).area < 1e-15
                ):
                    meta_poly = meta_poly.union(suggested_poly)
                    configs.append([center[0, 0], center[0, 1], angle])
                    break
            if meta_poly.touches(suggested_poly):
                if (
                    type(meta_poly.intersection(suggested_poly)) is not Point
                    and meta_poly.intersection(suggested_poly).area < 1e-8
                ):
                    meta_poly = meta_poly.union(suggested_poly)
                    configs.append([center[0, 0], center[0, 1], angle])
                    break
    # Finally, a random rotation for all objects

    def my_rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return [qx, qy]

    angle = np.random.uniform(-np.pi, np.pi)
    configs = [my_rotate((0, 0), (config[0], config[1]), angle) + [config[2] + angle] for config in configs]

    # fig, ax = plt.subplots()
    # ax.plot(*meta_poly.exterior.xy)
    # ax.plot(*(np.transpose(configs)[:2]), "o")
    # ax.set_aspect(1)
    # plt.show()

    return configs


def generate(shape_list, num_scenarios, num_shapes_min, num_shapes_max, color_space):
    """Randomly generate challenging VPG test scenarios. Output to txt.
    Params:
        shape_list (list): all available shapes.
        num_scenarios (int): number of scenarios to be generated.
        num_shapes_min, num_shapes_max: the range of number of objects in a scenario
    """
    np.random.seed(0)
    num_generated = 0
    while num_generated < num_scenarios:
        print(num_generated)
        num_objects = np.random.randint(num_shapes_min, num_shapes_max + 1)
        selected_objects = np.random.choice(shape_list, size=num_objects)
        try:
            configs = generate_unit_scenario(selected_objects)
            if configs is not None:
                configs = [[round(c, 6) for c in config] for config in configs]
                with open("hard-cases/" + str(num_generated) + ".txt", "w") as out_file:
                    # color_idx = np.random.choice(len(color_space), 2)
                    # color_idx = list(color_idx) * num_objects
                    for i, obj in enumerate(selected_objects):
                        # color = color_space[color_idx[i]]
                        color = color_space[i]
                        out_file.write(
                            "%s %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e\n"
                            % (
                                obj,
                                color[0],
                                color[1],
                                color[2],
                                configs[i][0] + 0.5,
                                configs[i][1],
                                heights[obj],
                                0,
                                0,
                                configs[i][2],
                            )
                        )
                num_generated += 1
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    signal(SIGINT, handler)
    generate([x for x in POLYGONS.keys()] * 6, 200000, 3, 9, color_space)