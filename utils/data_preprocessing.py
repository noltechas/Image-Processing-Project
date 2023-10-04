import pandas as pd


def load_keypoints_from_csv(csv_path):
    """
    Load keypoints from a given CSV file and return them grouped by filename.

    Args:
    - csv_path (str): Path to the CSV file.

    Returns:
    - dict: A dictionary where each key is a filename and the value is another dictionary containing the x and y coordinates of the keypoints and their corresponding joint names.
    """
    data = pd.read_csv(csv_path)

    # Extract relevant columns
    filenames = data['filename'].tolist()

    # Check for existence of 'cx' and 'cy' keys before extracting
    x_coordinates = [attr.get('cx', None) for attr in data['region_shape_attributes'].apply(eval)]
    y_coordinates = [attr.get('cy', None) for attr in data['region_shape_attributes'].apply(eval)]

    # Handle 'undefined' joint attribute without evaluating the string
    def get_joint(attr_str):
        if '"joint":undefined' in attr_str:
            return 'Lhand'
        attr = eval(attr_str)
        return attr.get('joint', 'Lhand')

    joints = [get_joint(attr_str) for attr_str in data['region_attributes']]

    # Filter out None values (rows without 'cx' or 'cy' keys)
    valid_indices = [i for i, x in enumerate(x_coordinates) if x is not None and y_coordinates[i] is not None]
    filenames = [filenames[i] for i in valid_indices]
    x_coordinates = [x_coordinates[i] for i in valid_indices]
    y_coordinates = [y_coordinates[i] for i in valid_indices]
    joints = [joints[i] for i in valid_indices]

    # Group data by filename
    grouped_data = {}
    for i, filename in enumerate(filenames):
        if filename not in grouped_data:
            grouped_data[filename] = {'x': [], 'y': [], 'joints': []}
        grouped_data[filename]['x'].append(x_coordinates[i])
        grouped_data[filename]['y'].append(y_coordinates[i])
        grouped_data[filename]['joints'].append(joints[i])

    return grouped_data
