import os
import pandas as pd


class Collector(object):

    instance_types = ['tp', 'tn', 'fp', 'fn']

    def __init__(self, folder, collect_options, save_original_template=False, template_file=None, sample=None):
        '''
        :param folder: Where to keep the result of collection.
        :param collect_options: A tuple of booleans (collect_tp, collect_tn, collect_fp, collect_fn).
        :param save_original_template: Whether to save the results in the form of original template.
                                       In this case, "template_file" must be specified.
        :param template_file: File of original templates. Should be specified when "save_original_template" is True.
        :param sample: If "sample" is not None, then keep the first "sample" instances for each type.
        '''
        assert not save_original_template or template_file
        self.save_original_template = save_original_template
        if save_original_template:
            template_df = pd.read_csv(template_file).set_index('EventId')
            self.template_dict = template_df.to_dict()['EventTemplate']

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder

        assert len(collect_options) == 4
        self.collect_options = {Collector.instance_types[i]: collect_options[i] for i in range(4)}
        self.collections = {Collector.instance_types[i]: [] for i in range(4)}
        self.sample = sample

    def clear(self):
        self.collections = {Collector.instance_types[i]: [] for i in range(4)}

    def add_instance(self, instance, instance_type):
        if self.collect_options[instance_type]:
            if not self.sample or len(self.collections[instance_type]) < self.sample:
                if self.save_original_template:
                    self.collections[instance_type].append([t + ' ' + self.template_dict[t] for t in instance])
                else:
                    self.collections[instance_type].append(instance)

    def write_collections(self):
        for t in Collector.instance_types:
            if self.collections[t]:
                file_name = os.path.join(self.folder, 'result_%s.txt' % t)
                with open(file_name, 'w') as f:
                    if self.save_original_template:
                        for c in self.collections[t]:
                            f.write('\n'.join(c))
                            f.write('\n\n')
                    else:
                        for c in self.collections[t]:
                            f.write(' '.join(c))
                            f.write('\n')
