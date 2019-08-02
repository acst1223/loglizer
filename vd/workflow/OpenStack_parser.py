from loglizer.vd_workflow import (csv_extractor,
                                  csv_block_extractor,
                                  splitter,
                                  classifier,
                                  time_transformer,
                                  valid_template,
                                  mean_std_stater,
                                  decimal_integrator)

origin_normal_file1 = 'OpenStack/openstack_normal1.log'
origin_normal_file2 = 'OpenStack/openstack_normal2.log'
origin_mixed_file = 'OpenStack/openstack_abnormal.log'
col_header_file = 'OpenStack/OpenStack_2k.log_templates.csv'
valid_template_file = 'OpenStack/valid_template.pkl'
label_file = 'anomaly_labels.txt'
train_min = 10
validate_min = 10
test_min = 10
tot_syms = 23

train_file = origin_normal_file2[: -4] + '.csv'
validate_file = origin_normal_file1[: -4] + '.csv'
test_file = origin_mixed_file[: -4] + '.csv'

# csv_block_extractor.load_label_file(label_file)
valid_templates = valid_template.load_valid_template(valid_template_file)

# STEP 1
# csv_extractor.csv_extracting(origin_normal_file1, col_header_file, None, 'OpenStack')
# csv_extractor.csv_extracting(origin_normal_file2, col_header_file, None, 'OpenStack')
# csv_extractor.csv_extracting(origin_mixed_file, col_header_file, None, 'OpenStack')

# STEP 2
# valid_templates = classifier.classify(train_file, validate_file, test_file, tot_syms,
#                                       train_min, validate_min, test_min, 'OpenStack')
# valid_template.save_valid_template(valid_template_file, valid_templates)

# STEP 3
# time_transformer.transform(train_file, validate_file, test_file, valid_templates, 'OpenStack')

# STEP 4
# decimal_integrator.integrate_decimal(train_file, validate_file, test_file, valid_templates)

# STEP 5
mean_std_stater.mean_std_stat(train_file, valid_templates, 'OpenStack')
