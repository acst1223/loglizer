import csv_extractor
import csv_block_extractor
import splitter
import classifier
import time_transformer
import valid_template
import mean_std_stater

origin_log_file = 'HDFS/HDFS_1e6.log'
col_header_file = 'HDFS/col_header.txt'
label_file = 'HDFS/label.csv'
valid_template_file = 'HDFS/valid_template.pkl'
train_min = 1
validate_min = 1
test_min = 1
train_ratio = 0.6
test_ratio = 0.3
tot_syms = 29

csv_log_file = origin_log_file[: -4] + '.csv'
csv_extracted_file = csv_log_file[: -4] + '_extracted.csv'
train_file = csv_log_file[: -4] + '_train.csv'
validate_file = csv_log_file[: -4] + '_validate.csv'
test_file = csv_log_file[: -4] + '_test.csv'

csv_block_extractor.load_label_file(label_file)
valid_templates = valid_template.load_valid_template(valid_template_file)

# STEP 1
csv_extractor.csv_extracting(origin_log_file, col_header_file, label_file)

# STEP 2
csv_block_extractor.csv_block_extracting(csv_log_file, csv_extracted_file)

# STEP 3
splitter.split(csv_extracted_file, train_file, validate_file, test_file, train_ratio, test_ratio)

# STEP 4
valid_templates = classifier.classify(train_file, validate_file, test_file, tot_syms,
                                      train_min, validate_min, test_min)
valid_template.save_valid_template(valid_template_file, valid_templates)

# STEP 6
time_transformer.transform(train_file, validate_file, test_file, valid_templates)

# STEP 7
mean_std_stater.mean_std_stat(train_file, valid_templates)
