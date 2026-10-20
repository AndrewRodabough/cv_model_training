from hashlib import sha256
import cv2
import matplotlib.pyplot as plt

SKELETON_CONNECTIONS = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (25, 23),
    (25, 24),
    (15, 25),
    (25, 26),
    (23, 24),
    (29, 27),
    (29, 28),
    (16, 29),
    (29, 30),
    (27, 28),
)

def person_color(person_id):
    digest = sha256(str(person_id).encode('utf-8')).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))

def draw_person(image, person):
    person_id = person.get('person_id', person.get('track_id', 'unknown'))
    color = person_color(person_id)
    bbox_element = person.find('bbox')
    if bbox_element is not None:
        bbox = tuple(float(bbox_element.get(name)) for name in ('x1', 'y1', 'x2', 'y2'))
    else:
        bbox = tuple(float(person.get(name)) for name in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))
    x1, y1, x2, y2 = (round(value) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        f'person_id={person_id}',
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    points = {}
    for keypoint in person.findall('keypoint'):
        point_id = int(keypoint.get('id'))
        visibility = int(keypoint.get('visibility', '0'))
        x = float(keypoint.get('x'))
        y = float(keypoint.get('y'))
        points[point_id] = (round(x), round(y), visibility)
        if visibility == 0:
            continue
        radius = 5 if visibility == 2 else 4
        cv2.circle(image, (round(x), round(y)), radius, color, -1)
        cv2.putText(
            image,
            str(point_id),
            (round(x) + 6, round(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    for first_id, second_id in SKELETON_CONNECTIONS:
        first = points.get(first_id)
        second = points.get(second_id)
        if first is None or second is None or first[2] == 0 or second[2] == 0:
            continue
        cv2.line(image, first[:2], second[:2], color, 2, cv2.LINE_AA)