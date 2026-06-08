import numpy as np
from collections import OrderedDict

class FaceTracker:
    def __init__(self, max_disappeared=15, max_distance=150):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> centroid (x, y)
        self.disappeared = OrderedDict()   # object_id -> number of frames disappeared
        
        # Identity memory: object_id -> {"name": str, "user_id": int/None, "confidence": float}
        self.identities = OrderedDict()
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, name, user_id, confidence):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        
        if name != "Unknown":
            self.identities[self.next_object_id] = {
                "name": name,
                "user_id": user_id,
                "confidence": confidence
            }
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.identities:
            del self.identities[object_id]

    def update(self, rects, current_names, current_user_ids, current_confidences):
        """
        rects: list of tuples (startX, startY, endX, endY)
        current_names: list of string names predicted by SVM
        current_user_ids: list of int IDs or None
        current_confidences: list of floats
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return [], [], []

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], current_names[i], current_user_ids[i], current_confidences[i])
            return current_names, current_user_ids, current_confidences

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Euclidean distance matrix
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        
        final_names = list(current_names)
        final_user_ids = list(current_user_ids)
        final_confidences = list(current_confidences)

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            # Identity Stabilization Logic
            curr_name = current_names[col]
            if curr_name != "Unknown":
                # Valid recognition, update tracker memory
                self.identities[object_id] = {
                    "name": curr_name,
                    "user_id": current_user_ids[col],
                    "confidence": current_confidences[col]
                }
            else:
                # Blurry/Unrecognized face -> Fallback to tracker memory if exists
                if object_id in self.identities:
                    final_names[col] = self.identities[object_id]["name"]
                    final_user_ids[col] = self.identities[object_id]["user_id"]
                    final_confidences[col] = self.identities[object_id]["confidence"]

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col], current_names[col], current_user_ids[col], current_confidences[col])

        return final_names, final_user_ids, final_confidences
