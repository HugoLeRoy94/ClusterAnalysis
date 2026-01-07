import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def trackmate_xml_to_df(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    model = root.find('Model')
    
    # 1. Get the list of 'Valid' Track IDs (The ones visible in Fiji)
    # TrackMate saves the IDs of tracks that passed the filter here
    valid_track_ids = set()
    filtered_tracks_elem = model.find('FilteredTracks')
    
    if filtered_tracks_elem is not None:
        for tid in filtered_tracks_elem.findall('TrackID'):
            valid_track_ids.add(int(tid.get('TRACK_ID')))
    else:
        print("Warning: No filter list found. Returning all tracks.")
    
    # 2. Parse Tracks to link Spot_IDs to Track_IDs
    # We only care about tracks that are in our 'valid_track_ids' set
    spot_to_track = {}
    tracks_element = model.find('AllTracks')
    
    if tracks_element is not None:
        for track in tracks_element.findall('Track'):
            track_id = int(track.get('TRACK_ID'))
            
            # SKIP this track if it is not in the valid list
            if track_id not in valid_track_ids:
                continue

            for edge in track.findall('Edge'):
                source_id = int(edge.get('SPOT_SOURCE_ID'))
                target_id = int(edge.get('SPOT_TARGET_ID'))
                spot_to_track[source_id] = track_id
                spot_to_track[target_id] = track_id

    # 3. Parse Spots
    spots_data = []
    # Optimization: We only need spots that are actually in a valid track
    for spots_in_frame in model.find('AllSpots'):
        for spot in spots_in_frame.findall('Spot'):
            spot_id = int(spot.get('ID'))
            
            # If this spot isn't part of a valid track, skip it
            if spot_id not in spot_to_track:
                continue
                
            attr = spot.attrib
            spots_data.append({
                'Spot_ID': spot_id,
                'Track_ID': spot_to_track[spot_id],
                'Frame': int(attr.get('FRAME')),
                'X': float(attr.get('POSITION_X')),
                'Y': float(attr.get('POSITION_Y')),
                'Z': float(attr.get('POSITION_Z')),
            })
    
    df = pd.DataFrame(spots_data)
    return df.sort_values(by=['Track_ID', 'Frame'])