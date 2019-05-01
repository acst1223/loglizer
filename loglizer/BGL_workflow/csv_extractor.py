import re
import pandas as pd


def extract_csv(log_file, col_header_file, output_file):
    '''
    :param log_file: txt/log
    :param col_header_file: txt
    :param output_file: csv
    :return: None
    '''
    print("== STEP 1 ==")
    assert log_file.endswith('.txt') or log_file.endswith('.log')

    idx = 0

    fw = open(output_file, 'w')
    fw.write('LineId,Label,Timestamp,Date,Node,Time,NodeRepeat,Type,Component,Level,Content,EventId,EventTemplate\n')
    fw.close()

    col_df = pd.read_csv(col_header_file)
    raw_headers = col_df['EventTemplate'].tolist()
    headers = [t.replace(r'[', r'\[') for t in raw_headers]
    headers = [t.replace(r']', r'\]') for t in headers]
    headers = [t.replace(r'.', r'\.') for t in headers]
    headers = [t.replace(r'(', r'\(') for t in headers]
    headers = [t.replace(r')', r'\)') for t in headers]
    headers = [t.replace(r'<*>', r'(.*)') for t in headers]
    patterns = [re.compile(t) for t in headers]  # patterns according to headers

    node_pattern = re.compile('[A-Z]\d{2}-[A-Z]\d-[A-Z]\d-[A-Z]:[A-Z]\d{2}-[A-Z]\d{2}')

    f = open(log_file, 'r')
    st = ''  # buffer

    while True:
        st2 = ''  # buffer in loop
        try:
            l = f.readline()
        except Exception:
            print('Exception met before index %d' % idx)
            continue
        if not l:
            break
        l = l[: -1].split(' ')
        if len(l) < 10:
            print('Line format illegal before index %d' % idx)
            continue
        if not node_pattern.match(l[3]):
            continue
        idx += 1
        st2 += '%d,' % idx  # LineId
        st2 += '%s,' % l[0]  # Label
        st2 += '%s,' % l[1]  # Timestamp
        st2 += '%s,' % l[2]  # Date
        st2 += '%s,' % l[3]  # Node
        st2 += '%s,' % l[4]  # Time
        st2 += '%s,' % l[5]  # NodeRepeat
        st2 += '%s,' % l[6]  # Type
        st2 += '%s,' % l[7]  # Component
        st2 += '%s,' % l[8]  # Level
        content = ' '.join(l[9:])
        hit = False
        for i in range(len(headers)):
            m = patterns[i].match(content)
            if m:
                if content.find(',') == -1:
                    st2 += '%s,' % content  # Content
                else:
                    st2 += '"%s",' % content  # Content
                st2 += 'E%d,' % (i + 1)  # EventId
                if content.find(',') == -1:
                    st2 += '%s' % raw_headers[i]  # EventTemplate
                else:
                    st2 += '"%s"' % raw_headers[i]  # EventTemplate
                hit = True
                break
        if not hit:
            idx -= 1
            continue
        st2 += '\n'
        st += st2
        if idx % 10000 == 0:
            fw = open(output_file, 'a')
            fw.write(st)
            fw.close()
            st = ''
            print('%d logs converted' % idx)

    f.close()

    if idx % 10000 != 0:
        fw = open(output_file, 'a')
        fw.write(st)
        fw.close()
        st = ''
        print('%d logs converted' % idx)
