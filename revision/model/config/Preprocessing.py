import os

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons
        self.training_base_dir = 'Preprocessing/input'
        self.training_area_fn = 'area/surrey_training_area_185_028.shp'
        self.training_polygon_fn = 'polygons/surrey_training_polygons_185_028.shp'
      
        # For reading the VHR images
        self.bands = [0, 1]
        self.raw_image_base_dir ='raw_images'
        self.raw_image_file_type = '.tif'
        self.raw_image_prefix = 'LN_surrey_102003_GE01-QB02-WV02-WV03_PV_185_028_mosaic'
        self.raw_ndvi_image_prefix = 'ndvi' 
        self.raw_pan_image_prefix = 'pan'

        # For writing the extracted images and their corresponding annotations and boundary file
        self.path_to_write = 'Preprocessing/output'
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
        self.extracted_ndvi_filename = 'ndvi'
        self.extracted_pan_filename = 'pan'
        self.extracted_annotation_filename = 'annotation'
        self.extracted_boundary_filename = 'boundary'
        

        # Path to write should be a valid directory
        assert os.path.exists(self.path_to_write)

        if not len(os.listdir(self.path_to_write)) == 0:
            print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')