from loglizer.BGL_workflow import csv_extractor, BGL_dropper, event_sequence_generator

folder = 'BGL/'
origin_log_file = folder + 'BGL.log'
col_header_file = folder + 'BGL_templates.csv'
csv_log_file = folder + 'BGL.csv'
csv_slim_log_file = folder + 'BGL_slim.csv'
event_sequence_file = folder + 'event_sequence.csv'

# STEP 1
# csv_extractor.extract_csv(origin_log_file, col_header_file, csv_log_file)

# STEP 2
# BGL_dropper.drop_csv(csv_log_file, csv_slim_log_file)

# STEP 3
event_sequence_generator.generate_event_sequence(csv_slim_log_file, event_sequence_file)
