import math

class Tracker:
    def __init__(self, max_age=10, distance_threshold=80,history_len=4):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0
        # Maximum frames an object can remain undetected before being removed
        self.max_age = max_age
        # Distance threshold for matching objects
        self.distance_threshold = distance_threshold
        # Dictionary to track the age of each object
        self.object_ages = {}
        #history_track
        self.history={}#{id:[(cx,cy),(cx,cy).....]}
        self.history_len=history_len

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # If no objects are detected, clean up old tracks and return empty list
        if not objects_rect:
            self._cleanup_old_tracks()
            return objects_bbs_ids, []

        # Convert [x1, y1, x2, y2] format to [x, y, w, h] format
        detections_rect = []
        detections_centers = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections_rect.append([x1, y1, w, h])  # [x, y, w, h]
            detections_centers.append((cx, cy))    # Center points

        # Match detected objects with existing IDs
        matched_ids = set()
        for i, (cx, cy) in enumerate(detections_centers):
            min_dist = float('inf')
            closest_id = None

            # Find the closest existing object within the distance threshold
            for obj_id, (prev_cx, prev_cy) in self.center_points.items():
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist < min_dist and dist < self.distance_threshold:
                    min_dist = dist
                    closest_id = obj_id

            # If a match is found, update the center point and reset age
            if closest_id is not None:
                self.center_points[closest_id] = (cx, cy)
                self.object_ages[closest_id] = 0  # Reset age since it's detected
                objects_bbs_ids.append([*detections_rect[i], closest_id])
                matched_ids.add(closest_id)

                if closest_id in self.history:
                    self.history[closest_id].append((cx,cy))
                    if len(self.history[closest_id])>self.history_len:
                        self.history[closest_id].pop(0)
                else:
                    self.history[closest_id]=[(cx,cy)]
            else:
                # Assign a new ID if no match is found
                new_id = self.id_count
                self.center_points[new_id] = (cx, cy)
                self.object_ages[new_id] = 0  # Initialize age
                objects_bbs_ids.append([*detections_rect[i], new_id])
                self.id_count += 1

                self.history[new_id]=[(cx,cy)]

        # Increment age for unmatched tracks and clean up old ones
        self._cleanup_old_tracks(matched_ids)

        # Return objects with bounding boxes and IDs
        return objects_bbs_ids,[]

    def _cleanup_old_tracks(self, matched_ids=None):
        """Remove tracks that have exceeded the maximum age."""
        if matched_ids is None:
            matched_ids = set()

        # Increment age for unmatched tracks
        for obj_id in list(self.center_points.keys()):
            if obj_id not in matched_ids:
                self.object_ages[obj_id] += 1
                if self.object_ages[obj_id] > self.max_age:
                    del self.center_points[obj_id]
                    del self.object_ages[obj_id]
    
    def get_direction(self,obj_id):
        if obj_id not in self.history or len(self.history[obj_id])<2:
            return None
        
        oldest_pt=self.history[obj_id][0]#1st point
        recent_pt=self.history[obj_id][-1]#last point

        dx=recent_pt[0]-oldest_pt[0]#difference in x
        dy=recent_pt[1]-oldest_pt[1]#difference in y

        #(dx,dy)...both direction and magnitude (movement vector)
        #if dx is positive->left to right, negative->right to left
        #if dy is positive->up to down ,negative->down to up

        displacement = math.hypot(dx, dy)

        if displacement >=20:
            return dx, dy
        else:
            return None